"""7단계. 최종 회수 시뮬레이터 (HTML) 만들기 — 새 모델(① 매각확률 × ② 낙찰가 배수 + 경매 보정) + 배당 엔진

기존 pipeline/build_dashboard.py 를 바탕으로 수정. 화면 템플릿은 dashboard/ 를 씀.

화면에서 하는 계산 (파이썬 4·6단계와 같은 규칙)
  입력: 감정가·용도·매각(예정)월·위치·토지·건물 + ★ 현재 최저가(또는 지금까지 유찰 횟수) + 권리관계·임차인·신보 채권
  1) 회차마다 최저가 = 현재 최저가 × 0.7^k
     ① 이번 회차 매각확률 = 공매 transfer 모델 → 경매 재보정(Platt)
     ② 팔린다면 낙찰가 = 최저가 × exp(log 배수 P10/P50/P90 + CQR + 경매 용도군 이동폭 + 구간 보정)
  2) 회차별 몬테카를로 → 최종 낙찰가율 분포 (팔린다면), 예상 회차, 8회 안에 안 팔릴 확률
  3) 그 낙찰가율 분포로 배당 엔진 1만 번 → 신보 회수액 분포

들어가는 것
  - 2단계 sell_transfer.txt / sell_meta.json, 3단계 mult_q10·50·90.txt / mult_meta.json  (2026-09 까지 학습한 최종 모델)
  - 4단계 auction_shift.json (경매 보정 실사용값), auction_predictions.csv (경고 문구용 검증 숫자)
  - 1단계 02_배수_학습표.csv (최근 공매 낙찰 지표 — 매각월 직전 3개월)
  - 지가변동률, 배당 수집양식(소액임차 기준표·지역 구분·사례 50건), settings.json
결과: output/07_시뮬레이터/ 회수시뮬레이터_최종.html (더블클릭해서 크롬으로 열기, 인터넷 없이 동작)
      simulator_data.json, simulator_ref.json (파이썬 정답값 — dashboard/test_engine.js 로 JS 계산이 같은지 확인)
"""
import json
import math
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd

import dividend_engine as s3
import step6_시뮬레이션 as S6
from common import (H, QS, AUCTION_DROP, ROOT, OUT_ROOT, PREP_DIR, SELL_DIR, MULT_DIR, CALIB_DIR, SEED,
                    RONE_PUB_LAG, rone_tables, nfc, read_booster)
from step4_경매시험 import load_cases

warnings.filterwarnings("ignore")
SIM_TPL = ROOT / "dashboard"                       # app.html · engine.js · doc_parser.js · vendor/
OUT_DIR = OUT_ROOT / "07_시뮬레이터"
OUT_DIR.mkdir(parents=True, exist_ok=True)
N_SIM_WEB = 20000                                  # 화면 몬테카를로 횟수 (회차별)
REF_N = 20000                                      # 파이썬 정답값 몬테카를로 횟수


def clean(v):
    """NaN/NaT → None, numpy → python"""
    if v is None:
        return None
    if isinstance(v, pd.Timestamp):
        return None if pd.isna(v) else str(v.date())
    if isinstance(v, (float, np.floating)):
        return None if math.isnan(v) else float(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return v


def flatten_tree(ts):
    """LightGBM dump 나무 1개 → 배열들 (기존 build_dashboard 와 같음)"""
    T = {k: [] for k in ("f", "th", "c", "dl", "mt", "l", "r", "v")}
    MT = {"None": 0, "Zero": 1, "NaN": 2}

    def rec(n):
        if "leaf_value" in n:
            T["v"].append(n["leaf_value"])
            return ~(len(T["v"]) - 1)
        i = len(T["f"])
        T["f"].append(n["split_feature"])
        if n["decision_type"] == "==":
            T["th"].append(None)
            T["c"].append([int(c) for c in str(n["threshold"]).split("||")])
        else:
            T["th"].append(n["threshold"])
            T["c"].append(None)
        T["dl"].append(1 if n["default_left"] else 0)
        T["mt"].append(MT[n["missing_type"]])
        T["l"].append(None); T["r"].append(None)
        T["l"][i] = rec(n["left_child"])
        T["r"][i] = rec(n["right_child"])
        return i

    rec(ts)
    return T


def dump_trees(path, num, cats):
    d = read_booster(path).dump_model()
    assert len(d["feature_names"]) == len(num + cats), "모델 파일의 피처 수가 meta와 다름"
    return [flatten_tree(t["tree_structure"]) for t in d["tree_info"]]


def recent_tables(prpt):
    """매각월 M → (M−H−2 ~ M−H) 월 공매 낙찰의 용도별 낙찰가율 중앙값(r1), 같은 재산유형의 log 배수 중앙값(r2) — 4단계 recent_market 과 같은 정의"""
    g = pd.read_csv(PREP_DIR / "02_배수_학습표.csv", low_memory=False, usecols=["월", "usage6", "prpt", "y_ratio", "log_mult"])
    last_data = pd.Period(g["월"].max(), "M")
    out = {}
    for M in pd.period_range("2025-01", last_data + H, freq="M"):
        last = M - H
        months = [str(last - k) for k in range(3)]
        s = g[g["월"].isin(months)]
        out[str(M)] = {"months": months,
                       "r1": {k: float(v) for k, v in s.groupby("usage6")["y_ratio"].median().items()},
                       "r2": {k: float(v) for k, v in s[s["prpt"] == prpt].groupby("usage6")["log_mult"].median().items()}}
    return {"byMonth": out, "lastData": str(last_data)}


def export_model():
    sm = json.load(open(SELL_DIR / "sell_meta.json", encoding="utf-8"))
    tr = sm["transfer"]
    mm = json.load(open(MULT_DIR / "mult_meta.json", encoding="utf-8"))
    sh = json.load(open(CALIB_DIR / "auction_shift.json", encoding="utf-8"))
    prpt = sh["prpt_for_auction"]
    sell = {"features": tr["num"] + tr["cats"], "num": tr["num"], "cat": tr["cats"], "categories": tr["categories"],
            "trees": dump_trees(SELL_DIR / "sell_transfer.txt", tr["num"], tr["cats"])}
    mult = {"features": mm["num"] + mm["cats"], "num": mm["num"], "cat": mm["cats"], "categories": mm["categories"],
            "trees": {str(q): dump_trees(MULT_DIR / f"mult_q{int(q * 100)}.txt", mm["num"], mm["cats"]) for q in QS},
            "cqr_all": mm["cqr_all"], "cqr_by_usage": mm["cqr_by_usage"]}
    shift = {"groups": sh["shift_groups"], "logByGroup": sh["shift_log_by_group"], "widen": sh["widen_log"],
             "platt": sh["sell_platt"]}
    # 경고 문구용: 4단계 사건 교차 추정 예측으로 낸 경매 검증 숫자 (예측을 보정하는 데 쓰지 않음)
    ap = pd.read_csv(CALIB_DIR / "auction_predictions.csv")
    ap = ap[ap["y"].notna()]
    ev = {"period": "2026년 10월 대구·경북 경매 (사건 교차 추정)", "n": int(len(ap)),
          "mae": float(np.mean(np.abs(ap["p50"] - ap["y"])) * 100), "below_p50": float(np.mean(ap["y"] < ap["p50"])),
          "below_p10": float(np.mean(ap["y"] < ap["p10"])), "above_p90": float(np.mean(ap["y"] > ap["p90"])),
          "coverage": float(np.mean((ap["y"] >= ap["p10"]) & (ap["y"] <= ap["p90"]))),
          "bias_by_usage": {u: float(np.mean(s["p50"] - s["y"]) * 100) for u, s in ap.groupby("usage6")}}
    return {"kind": "final_v1", "prpt": prpt, "drop": AUCTION_DROP, "maxRounds": S6.MAX_ROUNDS, "nSim": N_SIM_WEB,
            "lag": H + RONE_PUB_LAG, "sell": sell, "mult": mult, "shift": shift, "recent": recent_tables(prpt),
            # 기존 화면 코드가 쓰는 항목 (구조·주용도 목록, PDF 읽기, 바닥글)
            "categories": mm["categories"], "train_rows": mm["train_rows"], "train_until": mm["train_until"],
            "method": "① 매각확률 Platt 재보정 · ② 용도군 이동폭 + 구간 보정 (10월 경매 388건)", "eval": ev}


def export_rone():
    rate_d, idx_d = rone_tables()
    start = pd.Period("2024-06", "M")                 # 12개월 누적(lr_12m)까지 계산되도록
    rate = {f"{a}|{b}|{c}|{ym}": v for (a, b, c, ym), v in rate_d.items() if ym >= start and pd.notna(v)}
    idx = {f"{a}|{b}|{c}|{ym}": v for (a, b, c, ym), v in idx_d.items() if ym >= start and pd.notna(v)}
    last = str(max(k[3] for k in rate_d))
    sgg = {}
    for (a, b, c, _) in rate_d:
        if a == b:
            continue
        sgg.setdefault(a, set()).add(b if b == c else f"{b} {c}")
    return {"rate": rate, "idx": idx, "last": last, "sgg": {k: sorted(v) for k, v in sgg.items()}}


def prop_of(f):
    """4단계 경매 시험셋 한 줄 → 화면 입력값"""
    return {"usage6": f["usage6"], "sido": f["sido"], "sgg": clean(f["sgg"]), "month": "2026-10",
            "land": clean(f["landSqms"]), "bld": clean(f["bldSqms"]), "lp": clean(f["land_price"]),
            "age": clean(f["bld_age_c"]), "flr": clean(f["bld_grnd_flr"]), "tot": clean(f["bld_tot_area"]),
            "strct": clean(f["bld_strct"]), "purps": clean(f["bld_purps"]),
            "apsl": float(f["apslEvlAmt"]), "minPrice": clean(float(f["apslEvlAmt"]) * float(f["rto"]) / 100),
            "fails": clean(f["경매_유찰횟수"])}


def main():
    print("=" * 70, "\n7단계. 최종 회수 시뮬레이터 (HTML)\n" + "=" * 70)
    model = export_model()
    rone = export_rone()
    shift = json.load(open(CALIB_DIR / "auction_shift.json", encoding="utf-8"))
    prpt = shift["prpt_for_auction"]
    D = s3.load_inputs()
    case = load_cases()
    case["key"] = case["사건"].str.split(" ", n=1).str[1]
    sim = pd.read_csv(OUT_ROOT / "06_시뮬레이션" / "simulation_50.csv")
    sim["key"] = sim["사건"].str.split(" ", n=1).str[1]
    res = pd.read_csv(OUT_ROOT / "06_시뮬레이션" / "진행중" / "dividend_results.csv")
    tbl = pd.read_csv(OUT_ROOT / "06_시뮬레이션" / "진행중" / "dividend_tables.csv")

    small = [{"kind": r["구분"], "region": r["지역"], "start": str(pd.Timestamp(r["시작일"]).date()),
              "end": "9999-12-31" if str(r["종료일"]).startswith("9999") else str(pd.Timestamp(r["종료일"]).date()),
              "limit": float(r["소액 보증금 상한"]), "amount": float(r["최우선변제액"]), "ratio": float(r["한도 비율"])}
             for _, r in D["small_tbl"].iterrows()]
    region_tbl = {nfc(r["시·군·구"]): r["기준 지역"] for _, r in D["region_tbl"].iterrows()}

    cases, ref_cases, ref_pred = [], [], []
    have = set(D["reg"]["key"])
    rng = np.random.default_rng(SEED)
    S6.N_SIM = REF_N
    for _, row in D["target"].iterrows():
        k = row["key"]
        f = case[case.key == k]
        if k not in have or f.empty:
            continue
        f = f.iloc[0]
        R = D["reg"][D["reg"].key == k]
        T = D["ten"][D["ten"].key == k]
        _, info = s3.build_case(row, D)
        prop = prop_of(f)
        cases.append({
            "key": k, "court": row["법원"], "caseNo": row["사건번호"], "item": int(row["물건번호"]),
            "use": row["용도"], "addr": row["소재지"], "apsl": float(row["감정가(원)"]), "sale": clean(float(row["매각가(원)"])),
            "baseDate": clean(s3.todate(row["최선순위 설정일"])), "region": info["지역"],
            "landApsl": clean(s3.tonum(row[s3.LAND_COL])) if s3.LAND_COL in row.index else None,
            "shares": ({"토지": s3.tonum(row[s3.LAND_COL]), "건물": float(row["감정가(원)"]) - s3.tonum(row[s3.LAND_COL])}
                       if s3.LAND_COL in row.index and s3.isnum(s3.tonum(row[s3.LAND_COL])) else None),
            "prop": {kk: v for kk, v in prop.items() if kk != "apsl"},
            "rights": [{"gu": clean(r["갑구/을구"]), "rank": (f"을{int(r['순위번호'])}" if pd.notna(r["순위번호"]) and str(r["갑구/을구"]).startswith("을") else None),
                        "book": clean(r[s3.BOOK_COL]) if s3.BOOK_COL in R.columns else None,
                        "type": clean(r["등기목적"]), "who": clean(r["권리자 유형"]),
                        "name": clean(r["권리자"]), "date": clean(r["접수일"]), "amount": clean(r["금액"]),
                        "valid": bool(r["유효"]), "memo": clean(r["메모"])} for _, r in R.iterrows()],
            "tenants": [{"no": clean(t["임차인 번호"]), "kind": clean(t["구분"]), "deposit": clean(s3.tonum(t["보증금(원)"])),
                         "rent": clean(s3.tonum(t["월세(원)"])), "moved": clean(t["전입일 (상가는 사업자등록일)"]),
                         "confirmed": clean(t["확정일자"]),
                         "claim": "Y" if str(t["배당요구 여부"]).strip().upper() == "Y" else "N",
                         "memo": clean(t["메모"])} for _, t in T.iterrows()],
        })
        # 파이썬 정답값: 회차표(결정적) + 몬테카를로(난수만 다름) + 배당표
        rto_k, p_k, P_k = S6.round_table(f, f["rto"], prpt, shift)
        mc = S6.simulate(f, f["rto"], prpt, shift, rng)
        ref_pred.append({"key": k, "prop": prop, "rto": rto_k.tolist(), "p": p_k.tolist(), "P": P_k.tolist(),
                         "mc": [mc["p10"], mc["p50"], mc["p90"]], "unsold": mc[f"{S6.MAX_ROUNDS}회 안에 미매각"],
                         "expRound": mc["예상회차"]})
        rr = res[res.key == k]
        if not rr.empty:
            rr = rr.iloc[0]
            sm = sim[sim.key == k].iloc[0]
            ref_cases.append({"key": k, "p": [sm["진행중_p10"], sm["진행중_p50"], sm["진행중_p90"]],
                              "sinbo_actual": rr["실제가_신보배당"], "sinbo_claim": rr["신보채권액"],
                              "table": tbl[tbl.key == k][["순위그룹", "청구액", "배당액"]].fillna(-1).values.tolist(),
                              "mc_median": rr["회수_중앙값"], "mc_zero": rr["영원확률"]})

    S = s3._S
    settings = {"sinboMode": S["SINBO_MODE"], "sinboRate": S["SINBO_AMOUNT_RATE"], "costRate": S["COST_RATE"],
                "mortgageRatio": S["MORTGAGE_RATIO"], "taxScenarios": S["TAX_SCENARIOS"], "nSim": S["N_SIM"],
                "useActualSinbo": S["USE_ACTUAL_SINBO"], "includeAuctionClaim": S["INCLUDE_AUCTION_CLAIM"],
                "useAuctionClaim": S["USE_AUCTION_CLAIM"], "absorb": S["ABSORB"], "taxFirst": S["TAX_FIRST"],
                "partialSubrogation": S["PARTIAL_SUBROGATION"], "splitBooks": S.get("SPLIT_BOOKS", True)}
    data = {"model": model, "rone": rone, "small": small, "regionTbl": region_tbl,
            "regions": sorted({r["region"] for r in small}), "cases": cases, "settings": settings,
            "built": pd.Timestamp.today().strftime("%Y-%m-%d")}
    js = json.dumps(data, ensure_ascii=False, separators=(",", ":"), default=clean)
    json.dump({"pred": ref_pred, "cases": ref_cases}, open(OUT_DIR / "simulator_ref.json", "w", encoding="utf-8"),
              ensure_ascii=False, default=clean)
    (OUT_DIR / "simulator_data.json").write_text(js, encoding="utf-8")

    tpl = (SIM_TPL / "app.html").read_text(encoding="utf-8")
    engine = (SIM_TPL / "engine.js").read_text(encoding="utf-8")
    parser = (SIM_TPL / "doc_parser.js").read_text(encoding="utf-8")
    body = (tpl.replace("/*__ENGINE__*/", engine).replace("/*__PARSER__*/", parser)
            .replace("/*__DATA__*/null", js.replace("</", "<\\/")))
    cdn = "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/"
    online = f'<script src="{cdn}pdf.min.js"></script>\n<script src="{cdn}pdf.worker.min.js"></script>'
    vend = SIM_TPL / "vendor"
    esc_script = "<\\/script"
    offline = "\n".join("<script>" + (vend / n).read_text(encoding="utf-8").replace("</script", esc_script) + "</script>"
                        for n in ("pdf.min.js", "pdf.worker.min.js"))
    (OUT_DIR / "회수시뮬레이터_최종_body.html").write_text(body.replace("<!--__PDFJS__-->", online), encoding="utf-8")
    body = body.replace("<!--__PDFJS__-->", offline)
    full = ('<!doctype html>\n<html lang="ko">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            '</head>\n<body>\n' + body + '\n</body>\n</html>\n')
    out = OUT_DIR / "회수시뮬레이터_최종.html"
    out.write_text(full, encoding="utf-8")
    n_trees = len(model["sell"]["trees"]) + sum(len(v) for v in model["mult"]["trees"].values())
    print(f"사례 {len(cases)}건, 나무 {n_trees}그루 (① {len(model['sell']['trees'])} + ② {n_trees - len(model['sell']['trees'])}), "
          f"파일 {out.stat().st_size / 1e6:.1f}MB")
    print(f"저장: {nfc(out)}")
    print("확인: node dashboard/test_engine.js  (파이썬 = JS 계산 일치)")


if __name__ == "__main__":
    main()
