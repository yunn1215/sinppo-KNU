"""1단계. 데이터 정리 — ① 매각확률 회차표 + ② 낙찰가 배수 학습표

입력 (data/)
  - train_features_20261105_유찰이력추가.csv   : 공매 낙찰 4,772건 + 유찰 이력 열 (기존 학습 데이터 + 추가 열)
  - onbid_회차결과_압류_기타일반_지분제외.csv.gz : 온비드 입찰결과 전체(낙찰·유찰), 압류재산+기타일반재산, 지분 N

처리 (어제 정한 분석 프로세스)
  - 공통 제외: 지분 물건, 재공매 중복(같은 물건은 마지막 낙찰만), 낙찰가율 0.05 미만·1.5 초과, 취소 행
  - 현재 회차 최저가율: API 값, 없으면 최저입찰가 ÷ 감정가 (결측 498건 중 483건 복원)
  - 시점 맞추기: 지가변동률은 매각월 − (H + 공표시차) 개월까지만, 최근 시장 지표도 예측 시점 이전 매각만으로
  - 누수 열(낙찰가·입찰자 수·낙찰가율)은 피처로 쓰지 않음

출력 (output/01_데이터정리)
  - 02_배수_학습표.csv        : ② 낙찰가 배수 모델용 (낙찰 1건 = 1행)
  - 01_매각확률_회차표.csv.gz : ① 매각확률 모델용 (입찰 회차 1번 = 1행, 끝까지 안 팔린 물건 포함)
"""
import warnings

import numpy as np
import pandas as pd

from common import (H, Y_RANGE, MULT_RANGE, PRPT_MAP, PREP_DIR, data_file, usage6, sido_grp_of,
                    split_addr, attach_market, nfc)

warnings.filterwarnings("ignore")


def fill_rto(rto, lowst, apsl):
    """감정가 대비 최저가율(%) — API 값이 비면 최저입찰가 ÷ 감정가로 복원"""
    r = pd.to_numeric(rto, errors="coerce")
    alt = pd.to_numeric(lowst, errors="coerce") / pd.to_numeric(apsl, errors="coerce") * 100
    return r.fillna(alt.round(2))


def recent_stat(df, value_col, keys, months=3, min_n=5):
    """예측 시점에 이미 결과가 나온 매각만으로 만든 최근 시장 지표.
    매각월 M 의 행은 (M−H−months+1) ~ (M−H) 월에 끝난 매각의 중앙값을 받음 (같은 필지·자기 자신은 포함 안 됨)."""
    per = df["opbd_date"].dt.to_period("M")
    tab = df.assign(_m=per).groupby(keys + ["_m"])[value_col].agg(["median", "size"])
    out = np.full(len(df), np.nan)
    key_vals = df[keys].astype(str).agg("|".join, axis=1).values
    tab_d = {}
    for idx, row in tab.iterrows():
        k = "|".join(map(str, idx[:-1]))
        tab_d.setdefault(k, {})[idx[-1]] = (row["median"], row["size"])
    for i, (k, m) in enumerate(zip(key_vals, per)):
        vals, ns = [], 0
        for j in range(H, H + months):
            v = tab_d.get(k, {}).get(m - j)
            if v is not None:
                vals.append(v[0]); ns += v[1]
        if vals and ns >= min_n:
            out[i] = float(np.median(vals))
    return out


def build_multiplier_table():
    raw = pd.read_csv(data_file("train_features_20261105_유찰이력추가.csv"), low_memory=False,
                      dtype={"sgg_cd": str, "pnu": str})
    df = raw.copy()
    df["opbd_date"] = pd.to_datetime(df["opbd_date"], errors="coerce")
    df = df.dropna(subset=["opbd_date", "y_ratio", "apslEvlAmt"])
    n0 = len(df)
    df = df.sort_values("opbd_date").drop_duplicates("cltrMngNo", keep="last")        # 재공매 중복 → 마지막 낙찰
    n1 = len(df)
    df = df[~df["onbidCltrNm"].astype(str).str.contains("지분")]                       # 지분 표기 물건
    n2 = len(df)
    df = df[df["y_ratio"].between(*Y_RANGE)].copy()
    n3 = len(df)
    df["usage6"] = [usage6(m, s) for m, s in zip(df["cltrUsgMclsCtgrNm"], df["cltrUsgSclsCtgrNm"])]
    df["prpt"] = df["prptDivNm"].map(PRPT_MAP).fillna("기타")
    df["opbd_qtr"] = df["opbd_date"].dt.quarter
    df["월"] = df["opbd_date"].dt.strftime("%Y-%m")
    df["rto"] = fill_rto(df["apslPrcCtrsLowstBidRto"], df["lowstBidPrcIndctCont"], df["apslEvlAmt"])
    df["mult"] = df["y_ratio"] / (df["rto"] / 100)
    n_rto = df["rto"].isna().sum()
    df = df[df["mult"].between(*MULT_RANGE)].copy()
    df["log_mult"] = np.log(df["mult"])
    df["누적_유찰횟수"] = pd.to_numeric(df["누적_유찰횟수"], errors="coerce")
    df["이전_낙찰횟수"] = pd.to_numeric(df["이전_낙찰횟수"], errors="coerce").fillna(0)
    df["재매각"] = (df["이전_낙찰횟수"] > 0).astype(int)
    df = attach_market(df)
    df["최근_낙찰가율_용도"] = recent_stat(df, "y_ratio", ["usage6"])
    df["최근_배수_용도유형"] = recent_stat(df, "log_mult", ["usage6", "prpt"])
    print(f"[② 배수 학습표] {len(raw)}행 → 결측 정리 {n0} → 재공매 중복 {n1} → 지분 {n2} → 낙찰가율 범위 {n3} "
          f"→ 최저가율 없음 {n_rto}·배수 범위 밖 제외 → {len(df)}행")
    print(pd.crosstab(df["월"], df["prpt"]).T.to_string())
    print("배수(낙찰가÷최저가) 중앙값:", df.groupby("prpt")["mult"].median().round(3).to_dict())
    return df


def build_round_table():
    r = pd.read_csv(data_file("onbid_회차결과_압류_기타일반_지분제외.csv.gz"), dtype=str)
    n0 = len(r)
    r = r[r["pbctStatNm"].isin(["낙찰", "유찰"])].copy()                                  # 취소 등 제외
    r["opbd_date"] = pd.to_datetime(r["cltrOpbdDt"].str[:12], format="%Y%m%d%H%M", errors="coerce")
    r["nsq"] = pd.to_numeric(r["pbctNsq"], errors="coerce").fillna(0)
    r = r.dropna(subset=["opbd_date"]).sort_values(["cltrMngNo", "opbd_date", "nsq"]).reset_index(drop=True)
    r["label"] = (r["pbctStatNm"] == "낙찰").astype(int)
    g = r.groupby("cltrMngNo")
    r["누적_유찰횟수"] = g["label"].transform(lambda s: (1 - s).cumsum().shift(1, fill_value=0))
    r["이전_낙찰횟수"] = g["label"].transform(lambda s: s.cumsum().shift(1, fill_value=0))
    r["재매각"] = (r["이전_낙찰횟수"] > 0).astype(int)
    r["회차_경과일"] = (r["opbd_date"] - g["opbd_date"].transform("min")).dt.days
    r["apslEvlAmt"] = pd.to_numeric(r["apslEvlAmt"], errors="coerce")
    r["rto"] = fill_rto(r["apslPrcCtrsLowstBidRto"], r["lowstBidPrcIndctCont"], r["apslEvlAmt"])
    r = r[(r["apslEvlAmt"] > 0) & r["rto"].between(0.5, 150)].copy()
    r["log_apsl"] = np.log(r["apslEvlAmt"])
    r["usage6"] = [usage6(m, s) for m, s in zip(r["cltrUsgMclsCtgrNm"], r["cltrUsgSclsCtgrNm"])]
    r["prpt"] = r["prptDivNm"].map(PRPT_MAP).fillna("기타")
    r[["sido", "sgg"]] = [split_addr(a) for a in r["onbidCltrNm"]]
    r["sido_grp"] = r["sido"].map(sido_grp_of)
    r["is_daegu_gb"] = r["sido"].isin(["대구광역시", "경상북도"]).astype(int)
    r["개찰월"] = r["opbd_date"].dt.month
    r["월"] = r["opbd_date"].dt.strftime("%Y-%m")
    r = attach_market(r)
    keep = ["cltrMngNo", "pbctCdtnNo", "월", "opbd_date", "label", "prpt", "usage6", "sido", "sgg", "sido_grp",
            "is_daegu_gb", "apslEvlAmt", "log_apsl", "rto", "누적_유찰횟수", "이전_낙찰횟수", "재매각", "회차_경과일",
            "개찰월", "lr_1m", "lr_3m", "lr_6m", "lr_12m"]
    r = r[keep]
    print(f"\n[① 매각확률 회차표] {n0}행 → 낙찰·유찰만, 감정가·최저가율 있는 행 {len(r)}행 | 물건 {r['cltrMngNo'].nunique()}개")
    t = r.groupby("prpt").agg(회차=("label", "size"), 낙찰=("label", "sum"), 물건=("cltrMngNo", "nunique"))
    t["회차당 매각률"] = (t["낙찰"] / t["회차"]).round(4)
    print(t.to_string())
    never = r.groupby("cltrMngNo")["label"].max().eq(0).sum()
    print(f"끝까지 낙찰 기록이 없는 물건 {never}개 (포함 → 매각확률이 높게 나오는 문제 방지)")
    return r


def main():
    print("=" * 70, f"\n1단계. 데이터 정리 (예측 거리 H={H}개월)\n" + "=" * 70)
    m = build_multiplier_table()
    m.to_csv(PREP_DIR / "02_배수_학습표.csv", index=False, encoding="utf-8-sig")
    r = build_round_table()
    r.to_csv(PREP_DIR / "01_매각확률_회차표.csv.gz", index=False, encoding="utf-8", compression="gzip")
    print(f"\n저장: {nfc(PREP_DIR)}/02_배수_학습표.csv, 01_매각확률_회차표.csv.gz")


if __name__ == "__main__":
    main()
