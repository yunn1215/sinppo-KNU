"""2단계. 공매 → 경매 보정 + 경매 220건 평가 (원본: 코드/auction_calibration_colab.ipynb)

보정 계수 = 경매 매각가율 ÷ 공매 낙찰가율 (전국·용도별·금액 가중, 건수 적으면 전체 계수 쪽으로 축소 k=50)
후보(A 압류 / B 기타일반 / C 전체 / 보정 안 함)를 9월 매각통계(val)로 고르고, 10월에 채점.

결과: output/calib_out/calibration.json, factors_selected.csv, auction_predictions.csv
옵션: --team-model  → 1단계 대신 팀이 코랩에서 저장한 data/model_out 모델로 평가 (재현 확인용)
"""
import json
import sys
import warnings

import numpy as np
import pandas as pd

from common import (LAND_BLD_AS_ZERO, OUT_ROOT, QS, USAGES, LAND_USAGES as LAND, BLDG_USAGES as BLDG, Y_RANGE, DATA_DIR, STAT_DIR,
                    MODEL_DIR, CALIB_DIR, nfc, sub, load_gongmae, attach_market, AuctionModel, pinball)

warnings.filterwarnings("ignore")

CAL_END = "2026-08-31"          # 보정 기간 = 공매 train 기간 (2024.11 ~ 2026.08)
SHRINK_K = 50

# ───────── 1. 경매 매각가율 (법원 매각통계) ─────────
STAT_MAP = {"아파트": [("아파트", None)],
            "기타주거": [("단독주택,다가구주택", "소계"), ("연립주택,다세대", None)],
            "상가·업무": [("상가,오피스텔,근린시설", "소계")],
            "대지": [("대지,임야,전답", "대지")],
            "농지·임야": [("대지,임야,전답", "임야"), ("대지,임야,전답", "전답")],
            "공장·창고": [("기타", None)]}          # 전국 파일에만 있음


def _num(x):
    return pd.to_numeric(str(x).replace(",", "").replace("%", ""), errors="coerce")


def read_stat(path):
    d = pd.read_csv(path, encoding="utf-8-sig")
    d.columns = ["대", "소", "경매건수", "매각건수", "감정가", "매각가", "매각율", "매각가율"]
    d["대"] = d["대"].ffill()
    for c in ["경매건수", "매각건수", "감정가", "매각가"]:
        d[c] = d[c].map(_num)
    return d


def stat_by_usage(files):
    acc = {u: [0.0, 0.0, 0] for u in USAGES}
    found = {u: False for u in USAGES}
    for f in files:
        d = read_stat(f)
        for u, keys in STAT_MAP.items():
            for a, b in keys:
                r = d[(d["대"] == a) & ((d["소"] == b) if b else d["소"].isna())]
                if len(r):
                    acc[u][0] += r["감정가"].sum(); acc[u][1] += r["매각가"].sum(); acc[u][2] += int(r["매각건수"].sum())
                    found[u] = True
    return pd.DataFrame({u: {"경매율": (v[1] / v[0]) if v[0] > 0 else np.nan, "경매_매각건수": v[2]}
                         for u, v in acc.items() if found[u]}).T


def stat_files(period, regions):
    return sorted(f for f in STAT_DIR.glob("*.csv")
                  if nfc(f.name).startswith(period + " ") and any(f" {r} " in nfc(f.name) for r in regions))


def auction_rates():
    AUC = {}
    for label, periods in [("보정", ["24.11~25.10", "25.11~26.08"]), ("9월", ["26.9"]), ("10월", ["26.10"])]:
        for reg_label, regs in [("전국", ["전국"]), ("대구경북", ["대구", "경북"])]:
            files = sum([stat_files(p, regs) for p in periods], [])
            if not files:
                raise FileNotFoundError(f"{label} {reg_label} 매각통계 파일이 {STAT_DIR} 에 없어요.")
            AUC[(label, reg_label)] = stat_by_usage(files)
    return AUC


# ───────── 2. 공매 낙찰가율 ─────────
PERIODS = {"보정": ("2024-11-01", CAL_END), "9월": ("2026-09-01", "2026-09-30"), "10월": ("2026-10-01", "2026-10-31")}
OPTIONS = {"A 압류 기준": lambda f: f[f.prpt == "압류"],
           "B 기타일반 기준": lambda f: f[f.prpt == "기타일반"],
           "C 전체": lambda f: f}


def gongmae_rate(frame):
    t = frame.groupby("usage6").agg(감정가=("apslEvlAmt", "sum"), 낙찰가=("낙찰가", "sum"), n=("y_ratio", "size"))
    t["공매율"] = t["낙찰가"] / t["감정가"]
    return t[["공매율", "n"]].reindex(USAGES)


def period_mask(frame, start, end):
    return (frame["opbd_date"] >= pd.Timestamp(start)) & (frame["opbd_date"] <= pd.Timestamp(end))


# ───────── 3~4. 보정 계수 + val 선택 ─────────
def make_factors(GONG, auc_cal, opt):
    gr = GONG[("보정", opt)]
    both = gr.join(auc_cal, how="left")
    ok = both.dropna(subset=["공매율", "경매율"])
    w = ok["경매_매각건수"]
    f_all = (ok["경매율"] * w).sum() / (ok["공매율"] * w).sum()
    out = {}
    for u in USAGES:
        r = both.loc[u]
        if pd.notna(r["공매율"]) and pd.notna(r["경매율"]) and r["n"] > 0:
            raw = r["경매율"] / r["공매율"]
            out[u] = {"원계수": raw, "공매n": int(r["n"]), "보정계수": (r["n"] * raw + SHRINK_K * f_all) / (r["n"] + SHRINK_K)}
        else:
            out[u] = {"원계수": np.nan, "공매n": 0, "보정계수": f_all}
    t = pd.DataFrame(out).T
    t.loc["(전체)"] = [f_all, int(gr["n"].sum()), f_all]
    return t


def check(GONG, AUC, period, cand, auc_region="전국", usages=USAGES):
    rate_opt, fac, _ = cand
    gr, au_ = GONG[(period, rate_opt)], AUC[(period, auc_region)]
    rows = []
    for u in usages:
        if u not in au_.index or pd.isna(au_.loc[u, "경매율"]) or pd.isna(gr.loc[u, "공매율"]) or gr.loc[u, "n"] < 5:
            continue
        pred = gr.loc[u, "공매율"] * float(fac[u])
        rows.append({"용도": u, "공매n": int(gr.loc[u, "n"]), "공매율": gr.loc[u, "공매율"], "계수": float(fac[u]),
                     "예측 경매율": pred, "실제 경매율": au_.loc[u, "경매율"], "경매 매각건수": au_.loc[u, "경매_매각건수"]})
    t = pd.DataFrame(rows)
    if t.empty:
        return t, np.nan
    t["오차(%p)"] = (t["예측 경매율"] - t["실제 경매율"]) * 100
    return t, np.average(np.abs(t["오차(%p)"]), weights=t["경매 매각건수"])


def choose(GONG, AUC, cands, usages, title):
    common = [u for u in usages if all(GONG[("9월", c[0])].loc[u, "n"] >= 5 for c in cands.values())]
    rows = [{"후보": n, "9월 가중 오차(%p)": check(GONG, AUC, "9월", c, usages=common)[1]} for n, c in cands.items()]
    sm_ = pd.DataFrame(rows).sort_values("9월 가중 오차(%p)")
    print(f"\n[val 9월] {title} — 비교 용도 {common}"); print(sm_.round(2).to_string(index=False))
    return sm_.iloc[0]["후보"], common


# ───────── 6. 경매 220건 ─────────
def auction_usage(u, court_u):
    if u in ("공장", "창고"): return "공장·창고"
    if u == "상가": return "상가·업무"
    if court_u == "공장용지": return "공장·창고"
    if court_u in ("대지", "잡종지"): return "대지"
    return "농지·임야"


def split_addr(a):
    p = str(a).split()
    sido = p[0] if p else None
    sgg = None
    if len(p) > 1:
        sgg = p[1]
        if len(p) > 2 and p[1].endswith("시") and p[2].endswith("구"):
            sgg = f"{p[1]} {p[2]}"
    return sido, sgg


def load_auction_cases():
    xl = sub(DATA_DIR, "경매_매각사례_시험셋.xlsx")
    cases = pd.read_excel(xl, sheet_name="② 사례 기록", header=4)
    cases = cases[cases["★ 사건번호"].notna()].copy()
    notice = pd.read_excel(xl, sheet_name="① 공고 목록", header=4)
    key = ["★ 법원", "★ 사건번호", "★ 물건번호"]
    notice = notice.rename(columns={"법원": "★ 법원", "사건번호": "★ 사건번호", "물건번호": "★ 물건번호"})
    cases = cases.merge(notice[key + ["법원 표기 용도"]].drop_duplicates(key), on=key, how="left")
    au = pd.DataFrame({
        "사건": cases["★ 법원"].astype(str) + " " + cases["★ 사건번호"].astype(str) + "-" + cases["★ 물건번호"].astype(int).astype(str),
        "용도": cases["★ 용도"], "usage6": [auction_usage(u, c) for u, c in zip(cases["★ 용도"], cases["법원 표기 용도"])],
        "소재지": cases["★ 소재지 (지번까지)"],
        "apslEvlAmt": pd.to_numeric(cases["★ 감정가(원)"], errors="coerce"),
        "매각가": pd.to_numeric(cases["★ 매각가(원)"], errors="coerce"),
        "landSqms": pd.to_numeric(cases["토지면적(㎡)"], errors="coerce"),
        "bldSqms": pd.to_numeric(cases["건물면적(㎡)"], errors="coerce"),
        "opbd_date": pd.to_datetime(cases["★ 매각기일"], errors="coerce"),
        "다필지": cases["★ 소재지 (지번까지)"].astype(str).str.contains("외"),
        "우선표본": cases["메모"].astype(str).str.contains("우선"),
    })
    au[["sido", "sgg"]] = [split_addr(a) for a in cases["★ 소재지 (지번까지)"]]
    au["y"] = au["매각가"] / au["apslEvlAmt"]
    au["log_apsl"] = np.log(au["apslEvlAmt"])
    au["has_building"] = au["bldSqms"].notna()
    au["is_daegu_gb"] = True
    au["sido_grp"] = au["sido"]
    au["opbd_qtr"] = au["opbd_date"].dt.quarter
    for c in ["log_land_price", "apsl_to_official", "bld_age_c", "bld_grnd_flr", "bld_tot_area", "bld_strct", "bld_purps"]:
        au[c] = np.nan
    feat_path = OUT_ROOT / "auction_features.csv"           # 0단계를 돌렸으면 그 결과, 아니면 팀이 준 data/ 파일
    if not feat_path.exists():
        feat_path = DATA_DIR / "auction_features.csv"
    if feat_path.exists():
        ff = pd.read_csv(feat_path)
        au = au.drop(columns=[c for c in ff.columns if c in au.columns and c != "사건"]).merge(ff, on="사건", how="left")
        print(f"{feat_path.relative_to(feat_path.parent.parent)} 사용 (공시지가·건축물대장 포함)")
    else:
        print("⚠️ auction_features.csv 없음 → 공시지가·건축물대장 없이 간이 예측")
    if LAND_BLD_AS_ZERO:
        au.loc[~au["has_building"], "bldSqms"] = 0.0     # 학습 데이터와 같은 형식
    au = attach_market(au)
    n_all = len(au)
    au = au[au["y"].between(*Y_RANGE)].copy()
    print(f"경매 {n_all}건 → 낙찰가율 {Y_RANGE[0]}~{Y_RANGE[1]} 밖 {n_all - len(au)}건 제외 → {len(au)}건 평가")
    return au


def main():
    model_dir = DATA_DIR / "model_out" if "--team-model" in sys.argv else MODEL_DIR
    print("=" * 70, f"\n2단계. 공매 → 경매 보정 + 경매 평가  (모델: {model_dir})\n" + "=" * 70)

    AUC = auction_rates()
    auc_tbl = pd.concat({f"{k[0]} {k[1]}": v["경매율"] for k, v in AUC.items()}, axis=1).reindex(USAGES)
    print("경매 매각가율 (금액 가중, %)"); print((auc_tbl * 100).round(1).to_string())

    g = load_gongmae(verbose=False)
    g["낙찰가"] = g["y_ratio"] * g["apslEvlAmt"]
    GONG = {(per, opt): gongmae_rate(sel(g[period_mask(g, *PERIODS[per])])) for per in PERIODS for opt, sel in OPTIONS.items()}

    FACT = {opt: make_factors(GONG, AUC[("보정", "전국")], opt) for opt in OPTIONS}
    for opt, t in FACT.items():
        print(f"\n[{opt}] 계수"); print(t.astype(float).round(2).to_string())

    ONES = pd.Series(1.0, index=USAGES + ["(전체)"])
    CAND_BLDG = {"A 압류 기준": ("A 압류 기준", FACT["A 압류 기준"]["보정계수"], "압류"),
                 "B 기타일반 기준": ("B 기타일반 기준", FACT["B 기타일반 기준"]["보정계수"], "기타일반"),
                 "C 전체": ("C 전체", FACT["C 전체"]["보정계수"], "기타일반"),
                 "보정 안 함": ("B 기타일반 기준", ONES, "기타일반")}
    CAND_LAND = {"A 압류 기준": ("A 압류 기준", FACT["A 압류 기준"]["보정계수"], "압류"),
                 "보정 안 함": ("A 압류 기준", ONES, "압류")}
    BEST_B, COMMON_B = choose(GONG, AUC, CAND_BLDG, BLDG, "건물")
    BEST_L, COMMON_L = choose(GONG, AUC, CAND_LAND, LAND, "토지")
    FINAL_FACTOR = {u: float(CAND_BLDG[BEST_B][1][u]) for u in BLDG}
    FINAL_FACTOR.update({u: float(CAND_LAND[BEST_L][1][u]) for u in LAND})
    FINAL_PRPT = {u: CAND_BLDG[BEST_B][2] for u in BLDG}
    FINAL_PRPT.update({u: CAND_LAND[BEST_L][2] for u in LAND})
    BEST = f"건물 {BEST_B} / 토지 {BEST_L}"
    print(f"\n★ 선택: {BEST}")
    print(pd.DataFrame({"재산유형 고정": FINAL_PRPT, "보정계수": FINAL_FACTOR}).reindex(USAGES).round(3).to_string())

    # 저장 (아래 경매 예측이 이 파일을 읽음)
    calib = {"building_method": BEST, "fix_prpt_by_usage": FINAL_PRPT,
             "factors": FINAL_FACTOR, "shrink_k": SHRINK_K,
             "note": "경매 예측 = 공매 모델(용도별 재산유형 고정) P10/P50/P90 × 용도별 보정계수"}
    json.dump(calib, open(CALIB_DIR / "calibration.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    pd.DataFrame({"재산유형 고정": FINAL_PRPT, "보정계수": FINAL_FACTOR}).reindex(USAGES).to_csv(
        CALIB_DIR / "factors_selected.csv", encoding="utf-8-sig")

    # test ① 10월 매각통계
    rows = []
    for reg in ["전국", "대구경북"]:
        for name, c in CAND_BLDG.items():
            rows.append({"구분": f"건물 · {name}" + (" ★" if name == BEST_B else ""), "경매 지역": reg,
                         "10월 가중 오차(%p)": check(GONG, AUC, "10월", c, reg, usages=COMMON_B)[1]})
        for name, c in CAND_LAND.items():
            rows.append({"구분": f"토지 · {name}" + (" ★" if name == BEST_L else ""), "경매 지역": reg,
                         "10월 가중 오차(%p)": check(GONG, AUC, "10월", c, reg, usages=COMMON_L)[1]})
    print("\n[test ① 10월 매각통계]"); print(pd.DataFrame(rows).round(2).to_string(index=False))

    # test ② 경매 220건
    print()
    au = load_auction_cases()
    model = AuctionModel(model_dir, CALIB_DIR)
    y = au["y"].values; w = au["apslEvlAmt"].values

    def score(name, P):
        P = np.sort(P, axis=1)
        return {"방식": name, "n": len(y), "MAE(P50, %p)": np.mean(np.abs(y - P[:, 1])) * 100,
                "포함률(P10~P90)": np.mean((y >= P[:, 0]) & (y <= P[:, 2])),
                "구간폭(%p)": np.mean(P[:, 2] - P[:, 0]) * 100,
                "pinball": np.mean([pinball(y, P[:, i], q) for i, q in enumerate(QS)]),
                "예측 Σ매각가/Σ감정가": (P[:, 1] * w).sum() / w.sum(), "실제": (y * w).sum() / w.sum()}

    P_final = model.predict_auction(au)
    P0_fix = np.zeros((len(au), 3))
    for u in au["usage6"].unique():
        m = (au["usage6"] == u).values
        P0_fix[m] = model.predict_gongmae(au[m], FINAL_PRPT[u])
    P_a = np.clip(model.predict_gongmae(au, "압류") * au["usage6"].map(FACT["A 압류 기준"]["보정계수"].astype(float)).values[:, None], 0, None)
    dg, nat = AUC[("보정", "대구경북")]["경매율"], AUC[("보정", "전국")]["경매율"]
    base = au["usage6"].map(dg).fillna(au["usage6"].map(nat)).values
    results = [score("보정 없음 (공매 모델 그대로)", P0_fix), score("★ 보정 (선택한 방식)", P_final),
               score("참고: 전부 A(압류 기준)", P_a),
               {"방식": "기준선 (대구경북 경매 평균율)", "n": len(y), "MAE(P50, %p)": np.mean(np.abs(y - base)) * 100,
                "예측 Σ매각가/Σ감정가": (base * w).sum() / w.sum(), "실제": (y * w).sum() / w.sum()}]
    print("\n[test ② 대구·경북 10월 경매]"); print(pd.DataFrame(results).round(3).to_string(index=False))

    r = au.assign(p10=P_final[:, 0], p50=P_final[:, 1], p90=P_final[:, 2])
    print("\n[★ 보정] 용도별")
    print(r.groupby("usage6").apply(lambda s: pd.Series({
        "n": len(s), "MAE(%p)": np.mean(np.abs(s.y - s.p50)) * 100, "포함률": np.mean((s.y >= s.p10) & (s.y <= s.p90)),
        "실제 중앙값": s.y.median(), "예측 P50 중앙값": s.p50.median()}), include_groups=False).round(3).to_string())

    # 시뮬레이터 화면 경고용 진단값 (시험셋 채점 결과 — 이 값으로 예측을 보정하면 안 됨: 보정값은 다른 기간에서 정해야 함)
    bias = r.assign(e=(r.p50 - r.y) * 100).groupby("usage6")["e"].agg(["mean", "size"])
    diag = {"n": int(len(r)), "mae": float(np.mean(np.abs(r.y - r.p50)) * 100),
            "coverage": float(np.mean((r.y >= r.p10) & (r.y <= r.p90))),
            "below_p50": float(np.mean(r.y < r.p50)), "below_p10": float(np.mean(r.y < r.p10)), "above_p90": float(np.mean(r.y > r.p90)),
            "bias_by_usage": {u: round(float(v["mean"]), 1) for u, v in bias.iterrows() if v["size"] >= 10},
            "period": "2026-10 대구·경북 법원경매"}
    print(f"\n[진단] 실제가 P50 아래 {diag['below_p50']:.1%} · P10 아래 {diag['below_p10']:.1%} · P90 위 {diag['above_p90']:.1%} | 용도별 평균 과대(%p) {diag['bias_by_usage']}")
    if model_dir == MODEL_DIR:
        calib["eval_auction"] = diag
        json.dump(calib, open(CALIB_DIR / "calibration.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        r[["사건", "용도", "usage6", "apslEvlAmt", "매각가", "y", "p10", "p50", "p90", "다필지", "우선표본"]].to_csv(
            CALIB_DIR / "auction_predictions.csv", index=False, encoding="utf-8-sig")
        # 대시보드 v2(새 물건 예측)용: 경매 220건 피처 전체도 저장
        r.to_csv(CALIB_DIR / "auction_features_with_pred.csv", index=False, encoding="utf-8-sig")
    print(f"\n저장: {CALIB_DIR}/calibration.json, factors_selected.csv, auction_predictions.csv")


if __name__ == "__main__":
    main()
