"""4단계. HTML 회수 시뮬레이터 만들기

학습된 LightGBM 나무 3세트 + CQR + 보정 계수 + 지가변동률 + 소액임차 기준표 + 사례 49건(등기부·임차인 원자료)을
JSON으로 묶어 dashboard/app.html 템플릿에 넣고, 인터넷 없이 열리는 HTML 파일 하나를 만든다.

결과: output/회수시뮬레이터.html  (더블클릭해서 크롬으로 열면 됨)
      output/dashboard_ref.json   (파이썬 정답값 — dashboard/test_engine.js 로 JS 계산이 같은지 확인)
"""
import json
import math
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd

from common import LAND_BLD_AS_ZERO, ROOT, OUT_ROOT, MODEL_DIR, CALIB_DIR, DIV_DIR, QS, nfc
import step3_dividend as s3

warnings.filterwarnings("ignore")
DASH_DIR = ROOT / "dashboard"


def clean(v):
    """NaN/NaT → None, numpy → python"""
    if v is None:
        return None
    if isinstance(v, (pd.Timestamp,)):
        return None if pd.isna(v) else str(v.date())
    if isinstance(v, (float, np.floating)):
        return None if math.isnan(v) else float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return v


def flatten_tree(ts):
    """LightGBM dump 나무 1개 → 배열들 (f 피처, th 기준값, c 범주 목록, dl 결측 왼쪽, mt 결측 종류, l/r 자식, v 잎)"""
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


def export_model():
    meta = json.load(open(MODEL_DIR / "model_meta.json", encoding="utf-8"))
    calib = json.load(open(CALIB_DIR / "calibration.json", encoding="utf-8"))
    trees, feat_names, pandas_cat = {}, None, None
    for q in QS:
        d = lgb.Booster(model_file=str(MODEL_DIR / f"lgb_q{int(q * 100)}.txt")).dump_model()
        feat_names = d["feature_names"]
        pandas_cat = d["pandas_categorical"]
        trees[str(q)] = [flatten_tree(t["tree_structure"]) for t in d["tree_info"]]
    cats = meta["categories"]          # 범주 번호 = 이 목록에서의 위치 (파이썬 make_X 와 같음)
    if pandas_cat:                      # 예전 방식(pandas Categorical)으로 학습한 모델이면 순서가 같은지 확인
        assert dict(zip(meta["features_cat"], pandas_cat)) == cats, "모델 파일의 범주 순서가 meta와 다름"
    assert feat_names == meta["features_num"] + meta["features_cat"]
    return {"features": feat_names, "num": meta["features_num"], "cat": meta["features_cat"], "categories": cats,
            "trees": trees, "cqr_delta": meta["cqr_delta"], "cqr_by_usage": meta["cqr_by_usage"],
            "factors": calib["factors"], "fix_prpt": calib["fix_prpt_by_usage"], "method": calib["building_method"],
            "train_rows": meta.get("train_rows"), "train_until": meta.get("train_until"), "land_bld_zero": LAND_BLD_AS_ZERO,
            "eval": calib.get("eval_auction")}


def export_rone():
    from common import rone_tables
    rate_d, idx_d = rone_tables()
    start = pd.Period("2025-06", "M")
    rate = {f"{a}|{b}|{c}|{ym}": v for (a, b, c, ym), v in rate_d.items() if ym >= start and pd.notna(v)}
    idx = {f"{a}|{b}|{c}|{ym}": v for (a, b, c, ym), v in idx_d.items() if ym >= start and pd.notna(v)}
    last = str(max(k[3] for k in rate_d))
    sgg = {}
    for (a, b, c, _) in rate_d:
        if a == b:
            continue
        sgg.setdefault(a, set()).add(b if b == c else f"{b} {c}")
    return {"rate": rate, "idx": idx, "last": last, "sgg": {k: sorted(v) for k, v in sgg.items()}}


def main():
    print("=" * 70, "\n4단계. HTML 회수 시뮬레이터 만들기\n" + "=" * 70)
    model = export_model()
    rone = export_rone()
    D = s3.load_inputs()
    feats = pd.read_csv(CALIB_DIR / "auction_features_with_pred.csv")
    feats["key"] = feats["사건"].str.split(" ", n=1).str[1]
    res = pd.read_csv(DIV_DIR / "dividend_results.csv")
    tbl = pd.read_csv(DIV_DIR / "dividend_tables.csv")

    small = [{"kind": r["구분"], "region": r["지역"], "start": str(pd.Timestamp(r["시작일"]).date()),
              "end": "9999-12-31" if str(r["종료일"]).startswith("9999") else str(pd.Timestamp(r["종료일"]).date()),
              "limit": float(r["소액 보증금 상한"]), "amount": float(r["최우선변제액"]), "ratio": float(r["한도 비율"])}
             for _, r in D["small_tbl"].iterrows()]
    region_tbl = {nfc(r["시·군·구"]): r["기준 지역"] for _, r in D["region_tbl"].iterrows()}

    cases, ref_cases = [], []
    have = set(D["reg"]["key"])
    for _, row in D["target"].iterrows():
        k = row["key"]
        f = feats[feats.key == k]
        if k not in have or f.empty:
            continue
        f = f.iloc[0]
        R = D["reg"][D["reg"].key == k]
        T = D["ten"][D["ten"].key == k]
        _, info = s3.build_case(row, D)
        cases.append({
            "key": k, "court": row["법원"], "caseNo": row["사건번호"], "item": int(row["물건번호"]),
            "use": row["용도"], "addr": row["소재지"], "apsl": float(row["감정가(원)"]), "sale": clean(float(row["매각가(원)"])),
            "baseDate": clean(s3.todate(row["최선순위 설정일"])), "region": info["지역"],
            "landApsl": clean(s3.tonum(row[s3.LAND_COL])) if s3.LAND_COL in row.index else None,
            "shares": ({"토지": s3.tonum(row[s3.LAND_COL]), "건물": float(row["감정가(원)"]) - s3.tonum(row[s3.LAND_COL])}
                       if s3.LAND_COL in row.index and s3.isnum(s3.tonum(row[s3.LAND_COL])) else None),
            "prop": {"usage6": f["usage6"], "sido": f["sido"], "sgg": clean(f["sgg"]), "month": str(f["opbd_date"])[:7],
                     "land": clean(f["landSqms"]), "bld": clean(f["bldSqms"]) or None, "lp": clean(f["land_price"]),
                     "age": clean(f["bld_age_c"]), "flr": clean(f["bld_grnd_flr"]), "tot": clean(f["bld_tot_area"]),
                     "strct": clean(f["bld_strct"]), "purps": clean(f["bld_purps"])},
            "rights": [{"gu": clean(r["갑구/을구"]), "rank": (f"을{int(r['순위번호'])}" if pd.notna(r["순위번호"]) and str(r["갑구/을구"]).startswith("을") else None),
                        "book": clean(r[s3.BOOK_COL]) if s3.BOOK_COL in R.columns else None,
                        "type": clean(r["등기목적"]), "who": clean(r["권리자 유형"]),
                        "name": clean(r["권리자"]), "date": clean(r["접수일"]), "amount": clean(r["금액"]),
                        "valid": bool(r["유효"]), "memo": clean(r["메모"])} for _, r in R.iterrows()],
            "tenants": [{"no": clean(t["임차인 번호"]), "kind": clean(t["구분"]), "deposit": clean(tonum_raw(t["보증금(원)"])),
                         "rent": clean(tonum_raw(t["월세(원)"])), "moved": clean(t["전입일 (상가는 사업자등록일)"]),
                         "confirmed": clean(t["확정일자"]),
                         "claim": "Y" if str(t["배당요구 여부"]).strip().upper() == "Y" else "N",
                         "memo": clean(t["메모"])} for _, t in T.iterrows()],
        })
        rr = res[res.key == k].iloc[0]
        ref_cases.append({"key": k, "p": [f["p10"], f["p50"], f["p90"]], "sinbo_actual": rr["실제가_신보배당"], "sinbo_claim": rr["신보채권액"],
                          "table": tbl[tbl.key == k][["순위그룹", "청구액", "배당액"]].fillna(-1).values.tolist(),
                          "mc_median": rr["회수_중앙값"], "mc_zero": rr["영원확률"]})

    nf = lambda v: None if pd.isna(v) else v
    ref_pred = [{"key": r["사건"], "p": [r["p10"], r["p50"], r["p90"]],
                 "prop": {"usage6": r["usage6"], "sido": r["sido"], "sgg": nf(r["sgg"]), "month": str(r["opbd_date"])[:7],
                          "apsl": r["apslEvlAmt"], "land": nf(r["landSqms"]), "bld": nf(r["bldSqms"]) or None, "lp": nf(r["land_price"]),
                          "age": nf(r["bld_age_c"]), "flr": nf(r["bld_grnd_flr"]), "tot": nf(r["bld_tot_area"]),
                          "strct": nf(r["bld_strct"]), "purps": nf(r["bld_purps"])}}
                for _, r in feats.iterrows()]

    S = s3._S        # settings.json (step3와 같은 값)
    settings = {"sinboMode": S["SINBO_MODE"], "sinboRate": S["SINBO_AMOUNT_RATE"], "costRate": S["COST_RATE"],
                "mortgageRatio": S["MORTGAGE_RATIO"], "taxScenarios": S["TAX_SCENARIOS"], "nSim": S["N_SIM"],
                "useActualSinbo": S["USE_ACTUAL_SINBO"], "includeAuctionClaim": S["INCLUDE_AUCTION_CLAIM"],
                "useAuctionClaim": S["USE_AUCTION_CLAIM"], "absorb": S["ABSORB"], "taxFirst": S["TAX_FIRST"],
                "partialSubrogation": S["PARTIAL_SUBROGATION"], "splitBooks": S.get("SPLIT_BOOKS", True)}
    data = {"model": model, "rone": rone, "small": small, "regionTbl": region_tbl,
            "regions": sorted({r["region"] for r in small}), "cases": cases, "settings": settings,
            "built": pd.Timestamp.today().strftime("%Y-%m-%d")}
    js = json.dumps(data, ensure_ascii=False, separators=(",", ":"), default=clean)
    json.dump({"pred": ref_pred, "cases": ref_cases}, open(OUT_ROOT / "dashboard_ref.json", "w", encoding="utf-8"),
              ensure_ascii=False, default=clean)
    (OUT_ROOT / "dashboard_data.json").write_text(js, encoding="utf-8")

    tpl = (DASH_DIR / "app.html").read_text(encoding="utf-8")
    engine = (DASH_DIR / "engine.js").read_text(encoding="utf-8")
    parser = (DASH_DIR / "doc_parser.js").read_text(encoding="utf-8")
    body = (tpl.replace("/*__ENGINE__*/", engine).replace("/*__PARSER__*/", parser)
            .replace("/*__DATA__*/null", js.replace("</", "<\\/")))
    # PDF 읽기(pdf.js 3.11.174): 아티팩트는 CDN, 오프라인 파일은 파일 안에 넣음 (인터넷 없이 동작)
    cdn = "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/"
    online = f'<script src="{cdn}pdf.min.js"></script>\n<script src="{cdn}pdf.worker.min.js"></script>'
    vend = DASH_DIR / "vendor"
    offline = "\n".join(f"<script>{(vend / n).read_text(encoding='utf-8').replace('</script', '<\\/script')}</script>"
                         for n in ("pdf.min.js", "pdf.worker.min.js"))
    # 아티팩트용(본문만) / 오프라인용(완전한 HTML 문서) 두 가지
    (OUT_ROOT / "회수시뮬레이터_body.html").write_text(body.replace("<!--__PDFJS__-->", online), encoding="utf-8")
    body = body.replace("<!--__PDFJS__-->", offline)
    full = ('<!doctype html>\n<html lang="ko">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            '</head>\n<body>\n' + body + '\n</body>\n</html>\n')
    out = OUT_ROOT / "회수시뮬레이터.html"
    out.write_text(full, encoding="utf-8")
    print(f"사례 {len(cases)}건, 나무 {sum(len(v) for v in model['trees'].values())}그루, 파일 {out.stat().st_size / 1e6:.1f}MB")
    print("저장:", out)


def tonum_raw(v):
    return s3.tonum(v)


if __name__ == "__main__":
    main()
