"""2단계. ① 매각확률 모델 — "이번 회차에 팔릴까?" (낙찰 1 / 유찰 0)

입력: output/01_데이터정리/01_매각확률_회차표.csv.gz (공매 회차 1번 = 1행, 끝까지 안 팔린 물건 포함)

모델
  - 기준선: (재산유형 × 최저가율 10%p 구간)별 과거 매각률
  - 로지스틱 회귀
  - LightGBM (격자 후보 중 롤링 CV 평균 log loss 로 선택)
검증: 월 단위 롤링 CV (평가 월 2026-03 ~ 09, 각 fold 는 그 전 달까지 학습) → 설정 선택
최종: 2026-09 까지 전부로 재학습 → 2026-10 공매 회차로 한 번만 채점

모델 두 개를 저장
  - main     : 공매용 (누적 유찰횟수·회차 경과일 포함)
  - transfer : 경매 적용용. 공매는 회차마다 약 10%, 경매는 30% 내려가서 '유찰 횟수'의 의미가 달라
               횟수 대신 가격 수준(최저가율)만으로 학습 → 4단계에서 경매에 적용

지표: AUC, Brier, log loss, 보정 곡선(주) / 정확도·F1(보조)
느린 컴퓨터에서 나눠 돌리기: python step2_매각확률.py --cv-only=2026-03,2026-04 (fold 별 저장) → 마지막에 --final-only
출력: output/02_매각확률/ sell_main.txt, sell_transfer.txt, sell_meta.json, cv_scores.csv, test_report.csv, calibration_test.csv
"""
import sys
import time
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from common import (H, CV_MONTHS, TEST_MONTH, FINAL_TRAIN_END, PREP_DIR, SELL_DIR, LGB_FIXED,
                    make_X, eval_prob, calib_table, show, save_json, nfc, save_booster)

warnings.filterwarnings("ignore")

NUM_MAIN = ["rto", "log_apsl", "누적_유찰횟수", "이전_낙찰횟수", "회차_경과일", "is_daegu_gb", "개찰월", "lr_1m", "lr_6m"]
NUM_TRANSFER = ["rto", "log_apsl", "is_daegu_gb", "lr_1m", "lr_6m"]
CATS = ["usage6", "prpt", "sido_grp"]
CATS_TRANSFER = ["usage6", "prpt"]
GRID = [dict(num_leaves=15, min_child_samples=100, learning_rate=0.05, n_estimators=300),
        dict(num_leaves=31, min_child_samples=50, learning_rate=0.05, n_estimators=600)]


def month_le(df, ym):
    return df["월"] <= ym


def prev_month(ym, k):
    return str(pd.Period(ym, "M") - k)


# ── 기준선 ──
def baseline_fit(tr):
    b = (tr["rto"] // 10).clip(0, 10)
    tab = tr.assign(b=b).groupby(["prpt", "b"])["label"].agg(["sum", "size"])
    glob = tr["label"].mean()
    return {k: (v["sum"] + 20 * glob) / (v["size"] + 20) for k, v in tab.iterrows()}, glob


def baseline_predict(model, te):
    tab, glob = model
    b = (te["rto"] // 10).clip(0, 10)
    return np.array([tab.get((p, bb), glob) for p, bb in zip(te["prpt"], b)])


# ── 로지스틱 ──
def logit_design(df, num, cats, ref=None):
    X = df[num].astype(float)
    med = X.median() if ref is None else ref["med"]
    Xn = X.fillna(med)
    na = X.isna().astype(float).add_suffix("_na")
    D = pd.get_dummies(df[cats].astype(str), dtype=float)
    Z = pd.concat([Xn, na, D], axis=1)
    cols = Z.columns if ref is None else ref["cols"]
    Z = Z.reindex(columns=cols, fill_value=0.0)
    sc = StandardScaler().fit(Z) if ref is None else ref["sc"]
    return sc.transform(Z), {"med": med, "cols": cols, "sc": sc}


def logit_fit_predict(tr, te, num, cats, max_rows=150_000):
    if len(tr) > max_rows:                     # 비교용 기준 모델이라 15만 행 표본으로 학습 (메모리 4GB 컴퓨터에서도 돌도록)
        tr = tr.sample(max_rows, random_state=0)
    Xtr, ref = logit_design(tr, num, cats)
    Xte, _ = logit_design(te, num, cats, ref)
    m = LogisticRegression(max_iter=2000, C=1.0).fit(Xtr, tr["label"])
    return m.predict_proba(Xte)[:, 1]


# ── LightGBM ──
def lgb_fit(tr, num, cats, params):
    X = make_X(tr, num, cats)
    m = lgb.LGBMClassifier(objective="binary", **LGB_FIXED, **params)
    m.fit(X, tr["label"], categorical_feature=cats)
    return m, X.attrs["categories"]


def lgb_predict(m, cat_map, te, num, cats):
    return m.predict_proba(make_X(te, num, cats, cat_map))[:, 1]


def main():
    t0 = time.time()
    print("=" * 70, "\n2단계. ① 매각확률 모델\n" + "=" * 70)
    use = ["월", "label"] + sorted(set(NUM_MAIN + NUM_TRANSFER + CATS))
    df = pd.read_csv(PREP_DIR / "01_매각확률_회차표.csv.gz", usecols=use, low_memory=False)   # 필요한 열만 (메모리 절약)
    for c in NUM_MAIN:
        df[c] = df[c].astype("float32")
    print(f"회차표 {len(df):,}행 | 평가 월 {CV_MONTHS[0]}~{CV_MONTHS[-1]} 롤링 CV, test {TEST_MONTH}")

    # ── 롤링 CV ──  (나눠 돌리기: --cv-only=2026-03,2026-04 → 그 fold 만 저장 / --final-only → 저장된 fold 로 선택만)
    arg = {a.split("=")[0]: (a.split("=") + [""])[1] for a in sys.argv[1:]}
    months = arg["--cv-only"].split(",") if "--cv-only" in arg else ([] if "--final-only" in arg else CV_MONTHS)
    for M in months:
        rows = []
        tr = df[month_le(df, prev_month(M, H))]          # 예측 시점(M−H월 말)까지 결과가 나온 회차만
        va = df[df["월"] == M]
        y = va["label"].values
        rows.append({"평가월": M, **eval_prob("기준선", y, baseline_predict(baseline_fit(tr), va))})
        rows.append({"평가월": M, **eval_prob("로지스틱", y, logit_fit_predict(tr, va, NUM_MAIN, CATS))})
        for i, p in enumerate(GRID):
            m, cm = lgb_fit(tr, NUM_MAIN, CATS, p)
            rows.append({"평가월": M, **eval_prob(f"LightGBM 격자{i}", y, lgb_predict(m, cm, va, NUM_MAIN, CATS))})
        pd.DataFrame(rows).to_csv(SELL_DIR / f"cv_fold_{M}.csv", index=False, encoding="utf-8-sig")
        print(f"  fold {M}: 학습 {len(tr):,} / 평가 {len(va):,} ({time.time() - t0:.0f}초)")
    if "--cv-only" in arg:
        return
    rows = pd.concat([pd.read_csv(SELL_DIR / f"cv_fold_{M}.csv") for M in CV_MONTHS]).to_dict("records")
    cv = pd.DataFrame(rows)
    cv.to_csv(SELL_DIR / "cv_scores.csv", index=False, encoding="utf-8-sig")
    summ = cv.groupby("모델")[["AUC", "Brier", "logloss", "정확도", "F1"]].mean().sort_values("logloss")
    print("\n[롤링 CV 평균 (7개 fold)]"); print(summ.round(4).to_string())
    print("\n[fold별 AUC]"); print(cv.pivot(index="평가월", columns="모델", values="AUC").round(3).to_string())
    lgb_rows = summ[summ.index.str.startswith("LightGBM")]
    best_i = int(lgb_rows.index[0].replace("LightGBM 격자", ""))
    best = GRID[best_i]
    print(f"\n선택: LightGBM 격자{best_i} {best}")

    # ── 최종 학습 (~9월) + test (10월) ──
    full = df[df["월"] <= FINAL_TRAIN_END[:7]]
    te = df[df["월"] == TEST_MONTH]
    y = te["label"].values
    m_main, cm_main = lgb_fit(full, NUM_MAIN, CATS, best)
    m_tr, cm_tr = lgb_fit(full, NUM_TRANSFER, CATS_TRANSFER, best)
    p_main = lgb_predict(m_main, cm_main, te, NUM_MAIN, CATS)
    p_tr = lgb_predict(m_tr, cm_tr, te, NUM_TRANSFER, CATS_TRANSFER)
    rep = pd.DataFrame([eval_prob("기준선", y, baseline_predict(baseline_fit(full), te)),
                        eval_prob("로지스틱", y, logit_fit_predict(full, te, NUM_MAIN, CATS)),
                        eval_prob("LightGBM main", y, p_main),
                        eval_prob("LightGBM transfer (경매용)", y, p_tr)])
    print(f"\n[test {TEST_MONTH} 공매 회차]"); show(rep)
    by = []
    for g, idx in te.groupby("prpt").indices.items():
        by.append(eval_prob(f"main · {g}", y[idx], p_main[idx]))
        by.append(eval_prob(f"transfer · {g}", y[idx], p_tr[idx]))
    print("\n[test] 재산유형별"); show(pd.DataFrame(by))
    ct = calib_table(y, p_main)
    print("\n[test] 보정 곡선 (main)"); show(ct)
    rep.to_csv(SELL_DIR / "test_report.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(by).to_csv(SELL_DIR / "test_by_prpt.csv", index=False, encoding="utf-8-sig")
    ct.to_csv(SELL_DIR / "calibration_test.csv", index=False, encoding="utf-8-sig")
    imp = pd.Series(m_main.booster_.feature_importance("gain"), index=NUM_MAIN + CATS).sort_values(ascending=False)
    print("\n[main] 피처 중요도(gain, 상위)"); print((imp / imp.sum()).round(3).head(10).to_string())

    save_booster(m_main.booster_, SELL_DIR / "sell_main.txt")
    save_booster(m_tr.booster_, SELL_DIR / "sell_transfer.txt")
    save_json({"H": H, "params": best, "train_until": FINAL_TRAIN_END, "train_rows": int(len(full)),
               "main": {"num": NUM_MAIN, "cats": CATS, "categories": cm_main},
               "transfer": {"num": NUM_TRANSFER, "cats": CATS_TRANSFER, "categories": cm_tr},
               "note": "공매 회차 기준 매각확률. 경매 적용은 transfer 모델 + 4단계 절편 보정(9월 법원 매각통계 매각율)"},
              SELL_DIR / "sell_meta.json")
    print(f"\n저장: {nfc(SELL_DIR)} ({time.time() - t0:.0f}초)")


if __name__ == "__main__":
    main()
