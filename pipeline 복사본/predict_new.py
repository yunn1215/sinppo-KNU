"""새 경매 물건 하나의 낙찰가율 예측 (경매 기준 P10 / P50 / P90)

사용 예:
  python predict_new.py --감정가 8462007760 --용도 공장·창고 --시도 대구광역시 --시군구 달성군 \
      --토지면적 11031.3 --건물면적 4434.6 --매각월 2026-10

  선택 입력: --공시지가 (㎡당 원), --사용승인연도, --지상층수, --연면적, --구조, --주용도
  모르면 비워두면 됨 (모델이 결측으로 처리).

파이썬에서:
  from predict_new import predict_one
  predict_one({"감정가": 8.46e9, "용도": "공장·창고", "시도": "대구광역시", "시군구": "달성군", ...})
"""
import argparse

import numpy as np
import pandas as pd

from common import LAND_BLD_AS_ZERO, USAGES, AuctionModel, rone_tables, region_keys

SIDO_GRP = {"광주광역시": "광주·전남", "전라남도": "광주·전남", "전남광주통합특별시": "광주·전남",
            "강원도": "강원특별자치도", "전라북도": "전북특별자치도"}


def market_rates(sido, sgg, sale_month, lag=2):
    """매각월 2개월 전 지가변동률·6개월 누적. R-ONE 표보다 미래면 가장 최근 월 값을 씀"""
    rate_d, idx_d = rone_tables()
    if not rate_d:
        return np.nan, np.nan, None
    last = max(k[3] for k in rate_d)
    m = min(pd.Period(sale_month, "M") - lag, last)
    v1 = v6 = None
    for key in region_keys(sido, sgg):
        if v1 is None and key + (m,) in rate_d:
            v1 = rate_d[key + (m,)]
        if v6 is None and key + (m,) in idx_d and key + (m - 6,) in idx_d:
            v6 = (idx_d[key + (m,)] / idx_d[key + (m - 6,)] - 1) * 100
    return (np.nan if v1 is None else v1), (np.nan if v6 is None else v6), str(m)


def build_features(x):
    """입력 dict → 모델 피처 1행 (경매 220건 평가 때와 같은 규칙)"""
    sale = pd.Timestamp(str(x.get("매각월") or pd.Timestamp.today().strftime("%Y-%m")) + "-15")
    usage = x["용도"]
    if usage not in USAGES:
        raise ValueError(f"용도는 {USAGES} 중 하나")
    apsl = float(x["감정가"])
    land = x.get("토지면적")
    bld = x.get("건물면적")
    bld = None if bld in (None, "", 0) else float(bld)
    lp = x.get("공시지가")
    lp = None if lp in (None, "", 0) else float(lp)
    is_land = bld is None
    age = np.nan
    if not is_land and x.get("건물나이") not in (None, ""):        # 나이를 직접 아는 경우 (검증용)
        age = float(np.clip(float(x["건물나이"]), 0, 80))
    elif not is_land and x.get("사용승인연도"):
        age = float(np.clip(sale.year + (sale.month - 7) / 12 - float(x["사용승인연도"]), 0, 80))
    lr1, lr6, used_m = market_rates(x["시도"], x.get("시군구"), sale.strftime("%Y-%m"))
    row = {
        "usage6": usage, "sido": x["시도"], "sgg": x.get("시군구"),
        "sido_grp": SIDO_GRP.get(x["시도"], x["시도"]),
        "is_daegu_gb": x["시도"] in ("대구광역시", "경상북도"),
        "log_apsl": np.log(apsl), "apslEvlAmt": apsl,
        "landSqms": np.nan if land in (None, "") else float(land),
        "bldSqms": (0.0 if LAND_BLD_AS_ZERO else np.nan) if is_land else bld, "has_building": not is_land,
        "log_land_price": np.log(lp) if lp else np.nan,
        "apsl_to_official": apsl / (lp * float(land)) if lp and land else np.nan,
        "bld_age_c": age,
        "bld_grnd_flr": np.nan if is_land or x.get("지상층수") in (None, "") else float(x["지상층수"]),
        "bld_tot_area": np.nan if is_land or x.get("연면적") in (None, "") else float(x["연면적"]),
        "bld_strct": np.nan if is_land else (x.get("구조") or np.nan),
        "bld_purps": np.nan if is_land else (x.get("주용도") or np.nan),
        "opbd_qtr": sale.quarter, "lr_1m": lr1, "lr_6m": lr6,
    }
    return pd.DataFrame([row]), used_m


_MODEL = None


def predict_one(x, model=None):
    global _MODEL
    model = model or _MODEL or AuctionModel()
    _MODEL = model
    f, used_m = build_features(x)
    p10, p50, p90 = model.predict_auction(f)[0]
    apsl = float(x["감정가"])
    return {"P10": p10, "P50": p50, "P90": p90, "낙찰가_P10": apsl * p10, "낙찰가_P50": apsl * p50,
            "낙찰가_P90": apsl * p90, "지가변동률_기준월": used_m}


def main():
    ap = argparse.ArgumentParser(description="새 경매 물건 낙찰가율 예측")
    ap.add_argument("--감정가", type=float, required=True)
    ap.add_argument("--용도", required=True, choices=USAGES)
    ap.add_argument("--시도", required=True, help="예: 대구광역시, 경상북도, 경기도")
    ap.add_argument("--시군구", default=None, help="예: 달성군, 포항시 북구")
    ap.add_argument("--토지면적", type=float)
    ap.add_argument("--건물면적", type=float, help="토지만 있는 물건이면 생략")
    ap.add_argument("--공시지가", type=float, help="㎡당 개별공시지가(원)")
    ap.add_argument("--사용승인연도", type=int)
    ap.add_argument("--지상층수", type=float)
    ap.add_argument("--연면적", type=float)
    ap.add_argument("--구조", help="예: 일반철골구조, 철근콘크리트구조")
    ap.add_argument("--주용도", help="예: 공장, 창고시설, 제2종근린생활시설")
    ap.add_argument("--매각월", default=None, help="YYYY-MM (기본: 이번 달)")
    a = vars(ap.parse_args())
    r = predict_one(a)
    eok = lambda v: f"{v / 1e8:,.1f}억"
    print(f"\n경매 낙찰가율 예측 ({a['용도']}, 감정가 {eok(a['감정가'])})")
    print(f"  나쁠 때 P10 {r['P10']:.1%}  →  {eok(r['낙찰가_P10'])}")
    print(f"  보통   P50 {r['P50']:.1%}  →  {eok(r['낙찰가_P50'])}")
    print(f"  좋을 때 P90 {r['P90']:.1%}  →  {eok(r['낙찰가_P90'])}")
    print(f"  (지가변동률 기준월 {r['지가변동률_기준월']})")


if __name__ == "__main__":
    main()
