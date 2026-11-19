"""3단계. ② 낙찰가 배수 모델 — "팔린다면 현재 최저가의 몇 배에?"  log(낙찰가 ÷ 현재 최저가) 의 P10/P50/P90

입력: output/01_데이터정리/02_배수_학습표.csv (공매 낙찰 1건 = 1행)

기존 1단계(step1_bid_model.py)와 바뀐 점
  - 목표: 낙찰가 ÷ 감정가  →  log(낙찰가 ÷ 현재 최저가)   (공매에서 최저가율과 낙찰가율 순위상관 0.99)
  - 피처: 기존 피처 + 현재 최저가율, 누적 유찰횟수, 재매각 여부, 최근 시장 지표(예측 시점 이전 매각만), 지가변동률 3·12개월
  - 검증: 9월 한 달을 조기종료·설정 선택·CQR 에 세 번 쓰던 구조 → 월 단위 롤링 CV (평가 월 2026-03~09)
          · 설정(트리 수 포함)은 7개 fold 평균 pinball 로 선택 (조기종료 없음)
          · CQR 보정폭은 7개 fold 의 평가 월 예측(학습에 안 쓴 달)을 모아 계산, 공장·창고 포함 용도별
  - 채점: 낙찰가율(낙찰가 ÷ 감정가) 단위로 환산해 기존 모델과 같은 단위로 비교. 재산유형별로 따로 보고
          (2026-06 이전 fold 는 압류재산 낙찰가가 가려져 기타일반재산만 평가됨)

최종: 2026-09 까지 전부로 재학습 → 2026-10 공매로 한 번만 채점
출력: output/03_낙찰가배수/ mult_q10·q50·q90.txt, mult_meta.json, cv_scores.csv, test_report.csv, test_predictions.csv
"""
import time
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb

from common import (H, QS, CV_MONTHS, TEST_MONTH, FINAL_TRAIN_END, PREP_DIR, MULT_DIR, LGB_FIXED, SINBO_USAGE,
                    make_X, pinball, eval_quantile, by_group_q, show, save_json, nfc, save_booster)

warnings.filterwarnings("ignore")

NUM = ["rto", "누적_유찰횟수", "재매각", "log_apsl", "landSqms", "bldSqms", "has_building", "log_land_price",
       "apsl_to_official", "bld_age_c", "bld_grnd_flr", "bld_tot_area", "is_daegu_gb", "opbd_qtr",
       "lr_1m", "lr_3m", "lr_6m", "lr_12m", "최근_낙찰가율_용도", "최근_배수_용도유형"]
CATS = ["usage6", "prpt", "sido_grp", "bld_strct", "bld_purps"]
GRID = [dict(num_leaves=15, min_child_samples=40, learning_rate=0.03, n_estimators=400),
        dict(num_leaves=15, min_child_samples=40, learning_rate=0.03, n_estimators=900),
        dict(num_leaves=7, min_child_samples=60, learning_rate=0.05, n_estimators=500),
        dict(num_leaves=31, min_child_samples=20, learning_rate=0.03, n_estimators=600)]
# 기존 방식 재현용 (감정가 기준 낙찰가율, 최저가 정보 없음) — 기준선으로만 사용
OLD_NUM = ["log_apsl", "landSqms", "bldSqms", "has_building", "log_land_price", "apsl_to_official",
           "bld_age_c", "bld_grnd_flr", "bld_tot_area", "is_daegu_gb", "opbd_qtr", "lr_1m", "lr_6m"]
OLD_CATS = ["usage6", "sido_grp", "bld_strct", "bld_purps", "prpt"]
OLD_PARAMS = dict(num_leaves=15, min_child_samples=40, learning_rate=0.03, n_estimators=600)


def prev_month(ym, k):
    return str(pd.Period(ym, "M") - k)


def fit_q(tr, target, num, cats, params):
    X = make_X(tr, num, cats)
    ms = {}
    for q in QS:
        m = lgb.LGBMRegressor(objective="quantile", alpha=q, **LGB_FIXED, **params)
        m.fit(X, tr[target], categorical_feature=cats)
        ms[q] = m
    return ms, X.attrs["categories"]


def pred_q(ms, cat_map, te, num, cats):
    X = make_X(te, num, cats, cat_map)
    return np.sort(np.column_stack([ms[q].predict(X) for q in QS]), axis=1)


def to_ratio(frame, P_log):
    """log 배수 → 낙찰가율(낙찰가 ÷ 감정가) = 최저가율 × 배수"""
    return (frame["rto"].values[:, None] / 100) * np.exp(P_log)


def group_quantile_baseline(tr, te):
    """최저가 × (재산유형×용도)별 과거 배수 분위수 — 가장 단순한 '최저가 × 고정배수'"""
    tab = tr.groupby(["prpt", "usage6"])["log_mult"].quantile(QS).unstack()
    g = tr["log_mult"].quantile(QS).values
    P = np.array([tab.loc[(p, u)].values if (p, u) in tab.index else g for p, u in zip(te["prpt"], te["usage6"])])
    return P


def cqr_delta(y, P):
    """위·아래를 따로 보정 (P10 아래로 벗어난 정도의 90% 분위수, P90 위로 벗어난 정도의 90% 분위수).
    대칭 보정은 이번 데이터에서 아래쪽만 과하게 넓어져(P10 아래 1.6%) 비대칭으로 바꿈."""
    n = len(y)
    k = min(1.0, np.ceil((n + 1) * 0.9) / n)
    return [float(np.quantile(P[:, 0] - y, k)), float(np.quantile(y - P[:, 2], k))]


def main():
    t0 = time.time()
    print("=" * 70, "\n3단계. ② 낙찰가 배수 모델\n" + "=" * 70)
    df = pd.read_csv(PREP_DIR / "02_배수_학습표.csv", low_memory=False, dtype={"pnu": str, "sgg_cd": str})
    df["opbd_date"] = pd.to_datetime(df["opbd_date"])

    # ── 롤링 CV ──
    rows, oof = [], []
    for M in CV_MONTHS:
        tr = df[df["월"] <= prev_month(M, H)]
        va = df[df["월"] == M]
        if len(va) == 0:
            continue
        y = va["y_ratio"].values
        rows.append({"평가월": M, **eval_quantile("기준선: 최저가×고정배수", y, to_ratio(va, group_quantile_baseline(tr, va)))})
        oms, ocm = fit_q(tr, "y_ratio", OLD_NUM, OLD_CATS, OLD_PARAMS)
        rows.append({"평가월": M, **eval_quantile("기존 방식(감정가 기준)", y, pred_q(oms, ocm, va, OLD_NUM, OLD_CATS))})
        for i, p in enumerate(GRID):
            ms, cm = fit_q(tr, "log_mult", NUM, CATS, p)
            P = pred_q(ms, cm, va, NUM, CATS)
            r = eval_quantile(f"배수 LightGBM 격자{i}", y, to_ratio(va, P))
            r["pinball(log배수)"] = np.mean([pinball(va["log_mult"].values, P[:, j], q) for j, q in enumerate(QS)])
            rows.append({"평가월": M, **r})
            oof.append(pd.DataFrame({"격자": i, "평가월": M, "usage6": va["usage6"].values, "prpt": va["prpt"].values,
                                     "y_log": va["log_mult"].values, "p10": P[:, 0], "p50": P[:, 1], "p90": P[:, 2]}))
        print(f"  fold {M}: 학습 {len(tr):,} / 평가 {len(va):,} (압류 {int((va.prpt == '압류').sum())}) ({time.time() - t0:.0f}초)")
    cv = pd.DataFrame(rows)
    cv.to_csv(MULT_DIR / "cv_scores.csv", index=False, encoding="utf-8-sig")
    cols = ["MAE(P50,%p)", "pinball", "interval_score", "구간폭(%p)", "포함률(P10~P90)", "P50아래"]
    summ = cv.groupby("모델")[cols + ["pinball(log배수)"]].mean().sort_values("pinball")
    print("\n[롤링 CV 평균 — 낙찰가율 단위]"); print(summ.round(4).to_string())
    print("\n[fold별 MAE(%p)]"); print(cv.pivot(index="평가월", columns="모델", values="MAE(P50,%p)").round(2).to_string())
    g = summ[summ.index.str.startswith("배수")]["pinball(log배수)"].sort_values()
    best_i = int(g.index[0].replace("배수 LightGBM 격자", ""))
    best = GRID[best_i]
    print(f"\n선택: 격자{best_i} {best} (log 배수 pinball 기준)")

    # ── CQR: 롤링 CV 평가 월 예측(학습에 안 쓴 달)으로 보정폭 ──
    O = pd.concat(oof)
    O = O[O["격자"] == best_i]
    PO = O[["p10", "p50", "p90"]].values
    CQR_ALL = cqr_delta(O["y_log"].values, PO)
    CQR_USAGE = {u: cqr_delta(O.loc[O.usage6 == u, "y_log"].values, PO[(O.usage6 == u).values])
                 for u, s in O.groupby("usage6") if len(s) >= 30}
    print(f"CQR 보정폭(log 배수, [아래, 위]) 전체 {np.round(CQR_ALL, 3).tolist()} | 용도별",
          {k: np.round(v, 3).tolist() for k, v in CQR_USAGE.items()})

    def apply_cqr(frame, P):
        P = P.copy()
        d = np.array([CQR_USAGE.get(u, CQR_ALL) for u in frame["usage6"]])
        P[:, 0] -= d[:, 0]; P[:, 2] += d[:, 1]
        return np.sort(P, axis=1)

    # ── 최종 학습 (~9월) + test (10월, 한 번만) ──
    full = df[df["월"] <= FINAL_TRAIN_END[:7]]
    te = df[df["월"] == TEST_MONTH].copy()
    y = te["y_ratio"].values
    ms, cm = fit_q(full, "log_mult", NUM, CATS, best)
    P_raw = pred_q(ms, cm, te, NUM, CATS)
    P_cqr = apply_cqr(te, P_raw)
    oms, ocm = fit_q(full, "y_ratio", OLD_NUM, OLD_CATS, OLD_PARAMS)
    P_old = pred_q(oms, ocm, te, OLD_NUM, OLD_CATS)
    P_base = to_ratio(te, group_quantile_baseline(full, te))
    w = te["apslEvlAmt"].values
    rep = pd.DataFrame([eval_quantile("기준선: 최저가×고정배수", y, P_base, w),
                        eval_quantile("기존 방식(감정가 기준)", y, P_old, w),
                        eval_quantile("★ 배수 LightGBM", y, to_ratio(te, P_raw), w),
                        eval_quantile("★ 배수 LightGBM + CQR", y, to_ratio(te, P_cqr), w)])
    print(f"\n[test {TEST_MONTH} 공매 {len(te)}건 — 낙찰가율 단위]"); show(rep)
    P_fin = to_ratio(te, P_cqr)
    print("\n[test] 재산유형별 (★ + CQR)"); show(by_group_q(te, y, P_fin, "prpt"))
    print("\n[test] 재산유형별 (기존 방식)"); show(by_group_q(te, y, P_old, "prpt"))
    print("\n[test] 용도별 (★ + CQR)"); show(by_group_q(te, y, P_fin, "usage6"))
    m_sb = te["usage6"].isin(SINBO_USAGE).values
    print("\n[test] 신보 용도만")
    show(pd.DataFrame([eval_quantile("기존 방식", y[m_sb], P_old[m_sb]), eval_quantile("★ + CQR", y[m_sb], P_fin[m_sb])]))
    imp = pd.Series(ms[0.5].booster_.feature_importance("gain"), index=NUM + CATS).sort_values(ascending=False)
    print("\n[P50] 피처 중요도(gain, 상위)"); print((imp / imp.sum()).round(3).head(12).to_string())

    rep.to_csv(MULT_DIR / "test_report.csv", index=False, encoding="utf-8-sig")
    out = te[["cltrMngNo", "onbidCltrNm", "prpt", "usage6", "sido", "opbd_date", "apslEvlAmt", "rto", "y_ratio", "mult"]].copy()
    out[["p10", "p50", "p90"]] = P_fin
    out[["old_p10", "old_p50", "old_p90"]] = P_old
    out.to_csv(MULT_DIR / "test_predictions.csv", index=False, encoding="utf-8-sig")
    for q in QS:
        save_booster(ms[q].booster_, MULT_DIR / f"mult_q{int(q * 100)}.txt")
    save_json({"H": H, "target": "log(낙찰가 ÷ 현재 최저가)", "num": NUM, "cats": CATS, "categories": cm,
               "params": best, "cqr_all": CQR_ALL, "cqr_by_usage": CQR_USAGE,
               "train_until": FINAL_TRAIN_END, "train_rows": int(len(full)),
               "note": "예측 낙찰가 = 현재 최저가 × exp(P). 경매는 4단계에서 재산유형을 고정하고 평가"},
              MULT_DIR / "mult_meta.json")
    print(f"\n저장: {nfc(MULT_DIR)} ({time.time() - t0:.0f}초)")


if __name__ == "__main__":
    main()
