"""1단계. 공매 낙찰가 모델 (원본: 코드/bid_model_colab.ipynb)

기준선 → 선형 분위수 회귀 → LightGBM 분위수 회귀 (P10/P50/P90) → CQR 구간 보정
train ~2026-08 / val 2026-09 / test 2026-10 → 9월까지(train+val)로 재학습해 저장 (10월 경매 시험셋과 기간이 겹치지 않게)

결과: output/model_out/lgb_q10·q50·q90.txt, model_meta.json, test_predictions.csv, val_test_report.csv
"""
import json
import sys
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb
import statsmodels.api as sm

from common import (QS, NUM, CAT_BASE, MODEL_DIR, load_gongmae, attach_market, make_X,
                    evaluate, by_group, show)

warnings.filterwarnings("ignore")

TRAIN_END, VAL_END, TEST_END = "2026-08-31", "2026-09-30", "2026-10-31"
# 최종 모델 학습 마감일. 경매 219건 시험셋이 2026년 10월 매각이라, 10월 공매까지 넣으면 같은 달 시장 정보가 섞여
# 시험 점수가 실제보다 좋게 나올 수 있음 → 9월 30일까지만 학습 (원래 설계). 10월 이후 실제 업무용으로 다시 학습할 때만 바꿀 것.
FINAL_TRAIN_END = VAL_END
SEED = 42
USE_PRPT = True            # 재산유형(압류/기타일반)을 피처로 (val 비교 후 팀 결정)
CQR_BY_USAGE = True        # 용도별 구간 보정 (30건 미만 용도는 전체 보정폭)
SINBO_USAGE = ["공장·창고", "상가·업무", "대지", "농지·임야"]
GRID = [dict(num_leaves=15, min_child_samples=40, learning_rate=0.03),
        dict(num_leaves=31, min_child_samples=20, learning_rate=0.03),
        dict(num_leaves=7, min_child_samples=60, learning_rate=0.05)]
# 팀이 9월 val로 확정한 설정 (발표 자료와 동일). 격자0·격자1의 val pinball 차이가 0.001 이하라
# 라이브러리 버전에 따라 자동 선택이 뒤집힐 수 있어 고정해 둠. 다시 고르려면: python step1_bid_model.py --search
TEAM_PARAMS = GRID[0]
AUTO_SEARCH = "--search" in sys.argv
LGB_FIXED = dict(subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                 random_state=SEED, verbose=-1, n_jobs=1)   # n_jobs=1: 실행할 때마다 같은 결과


def split_of(d):
    if d <= pd.Timestamp(TRAIN_END): return "train"
    if d <= pd.Timestamp(VAL_END): return "val"
    if d <= pd.Timestamp(TEST_END): return "test"
    return "after"


def baseline_predict(train, target, keys):
    """같은 (유형×)용도의 train 낙찰가율 10/50/90% 분위수를 그대로 (현업 매각통계 방식)"""
    tab = train.groupby(keys)["y_ratio"].quantile(QS).unstack()
    glob_q = train["y_ratio"].quantile(QS).values
    P = []
    for _, r in target[keys].iterrows():
        k = tuple(r) if len(keys) > 1 else r.iloc[0]
        P.append(tab.loc[k].values if k in tab.index else glob_q)
    return np.array(P)


def linear_design(frame, ref=None):
    num = ["log_apsl", "log_land_price", "has_building", "bld_age_c", "lr_1m"]
    X = frame[num].apply(pd.to_numeric, errors="coerce").astype(float)
    med = X.median() if ref is None else ref["med"]
    X = X.fillna(med)
    for c in num:
        X[c + "_na"] = frame[c].isna().astype(float)
    cats = ["usage6"] + (["prpt"] if USE_PRPT else [])
    D = pd.get_dummies(frame[cats].astype(str), drop_first=False, dtype=float)
    X = pd.concat([X, D], axis=1)
    cols = X.columns if ref is None else ref["cols"]
    X = X.reindex(columns=cols, fill_value=0.0)
    X = X.loc[:, X.std() > 0] if ref is None else X
    return X, {"med": med, "cols": X.columns}


def fit_linear(tr, y_tr, targets):
    Xl_tr, ref = linear_design(tr)
    fits = {q: sm.QuantReg(y_tr, sm.add_constant(Xl_tr, has_constant="add")).fit(q=q, max_iter=5000) for q in QS}
    out = []
    for t in targets:
        Xl, _ = linear_design(t, ref)
        out.append(np.column_stack([fits[q].predict(sm.add_constant(Xl, has_constant="add")) for q in QS]))
    return out


def fit_lgb(train, valid, cats, params):
    Xtr = make_X(train, cats)
    cat_map = Xtr.attrs["categories"]
    Xva = make_X(valid, cats, cat_map)
    models, P = {}, []
    for q in QS:
        m = lgb.LGBMRegressor(objective="quantile", alpha=q, n_estimators=3000, **LGB_FIXED, **params)
        m.fit(Xtr, train["y_ratio"], eval_set=[(Xva, valid["y_ratio"])], eval_metric="quantile", categorical_feature=cats,
              callbacks=[lgb.early_stopping(100, verbose=False)])
        models[q] = m
        P.append(m.predict(Xva))
    return models, np.column_stack(P), cat_map


def cqr_delta(y, P):
    """val에서 실제값이 구간 밖으로 벗어난 정도의 80% 분위수만큼 구간을 넓힘(음수면 좁힘)"""
    s_ = np.maximum(P[:, 0] - y, y - P[:, 2]); n_ = len(y)
    return float(np.quantile(s_, min(1.0, np.ceil((n_ + 1) * 0.8) / n_)))


def main():
    print("=" * 70, "\n1단계. 공매 낙찰가 모델\n" + "=" * 70)
    df = load_gongmae()
    df = attach_market(df, verbose=True)
    df["split"] = df["opbd_date"].map(split_of)
    tr, va, te = (df[df.split == s] for s in ["train", "val", "test"])
    y_tr, y_va, y_te = tr["y_ratio"].values, va["y_ratio"].values, te["y_ratio"].values
    print(pd.crosstab(df["split"], df["prpt"]).to_string(), "\n")

    # ── 기준선 + 선형 ──
    base_keys = ["prpt", "usage6"] if USE_PRPT else ["usage6"]
    P_base_va = baseline_predict(tr, va, base_keys)
    P_lin_va, P_lin_te = fit_linear(tr, y_tr, [va, te])
    res = [evaluate("기준선", y_va, P_base_va), evaluate("선형 분위수 회귀", y_va, P_lin_va)]

    # ── LightGBM 격자 × 재산유형 유무 ──
    print("LightGBM 격자 탐색 중 (1~3분)...")
    best = None
    for prpt_flag in [True, False]:
        cats = CAT_BASE + (["prpt"] if prpt_flag else [])
        for i, params in enumerate(GRID):
            models, P, cat_map = fit_lgb(tr, va, cats, params)
            r = evaluate(f"LightGBM 격자{i} 유형{'O' if prpt_flag else 'X'}", y_va, P)
            res.append(r)
            if prpt_flag != USE_PRPT:
                continue
            pick = (r["pinball"] < best["r"]["pinball"]) if (AUTO_SEARCH and best) else (params == TEAM_PARAMS)
            if best is None and AUTO_SEARCH or pick:
                best = {"r": r, "params": params, "models": models, "P": P, "cats": cats, "cat_map": cat_map}
    print("\n[val 9월] 모델 비교"); show(pd.DataFrame(res))
    n_trees = {q: m.best_iteration_ for q, m in best["models"].items()}
    print("\n선택" + (" (자동 선택)" if AUTO_SEARCH else " (팀 확정 설정)") + ":", best["r"]["모델"], best["params"], "| 트리 수", n_trees)

    # ── CQR ──
    P_va = np.sort(best["P"], axis=1)
    CQR_DELTA = cqr_delta(y_va, P_va)
    CQR_USAGE = {u: cqr_delta(y_va[(va["usage6"] == u).values], P_va[(va["usage6"] == u).values])
                 for u, g in va.groupby("usage6") if len(g) >= 30}

    def apply_cqr(frame, P):
        P = np.sort(P, axis=1).copy()
        d = frame["usage6"].map(CQR_USAGE).fillna(CQR_DELTA).values if CQR_BY_USAGE else CQR_DELTA
        P[:, 0] -= d; P[:, 2] += d
        return P

    print(f"\nCQR 전체 보정폭 {CQR_DELTA * 100:+.2f}%p | 용도별", {k: round(v * 100, 2) for k, v in CQR_USAGE.items()})
    P_va_cqr = apply_cqr(va, P_va)
    show(pd.DataFrame([evaluate("LightGBM", y_va, P_va), evaluate("LightGBM + CQR", y_va, P_va_cqr)]))

    # ── test (10월) — 설정은 위에서 다 정했고 여기선 채점만 ──
    Xte = make_X(te, best["cats"], best["cat_map"])
    P_te = apply_cqr(te, np.column_stack([best["models"][q].predict(Xte) for q in QS]))
    P_base_te = baseline_predict(tr, te, base_keys)
    test_tbl = pd.DataFrame([evaluate("기준선", y_te, P_base_te), evaluate("선형", y_te, P_lin_te),
                             evaluate("LightGBM + CQR", y_te, P_te)])
    print("\n[test 10월 공매]"); show(test_tbl)
    m_sb = te["usage6"].isin(SINBO_USAGE).values
    print("\n[test] ★ 신보 용도만")
    show(pd.DataFrame([evaluate("기준선", y_te[m_sb], P_base_te[m_sb]),
                       evaluate("LightGBM + CQR", y_te[m_sb], P_te[m_sb])]))
    print("\n[test] 용도별"); print(by_group(te, y_te, P_te, "usage6").to_string())
    out = te[["cltrMngNo", "onbidCltrNm", "prpt", "usage6", "sido", "opbd_date", "apslEvlAmt", "y_ratio"]].copy()
    out[["p10", "p50", "p90"]] = P_te
    out.to_csv(MODEL_DIR / "test_predictions.csv", index=False, encoding="utf-8-sig")
    pd.concat([pd.DataFrame(res).assign(구분="val"), test_tbl.assign(구분="test")]).to_csv(
        MODEL_DIR / "val_test_report.csv", index=False, encoding="utf-8-sig")

    # ── 최종 모델: train+val(~9월) 재학습 (트리 수는 데이터 증가 비율만큼). 10월(test)은 넣지 않음 ──
    full = df[df["opbd_date"] <= pd.Timestamp(FINAL_TRAIN_END)]
    Xf = make_X(full, best["cats"])
    for q in QS:
        n_iter = max(50, int(n_trees[q] * len(full) / len(tr)))
        m = lgb.LGBMRegressor(objective="quantile", alpha=q, n_estimators=n_iter, **LGB_FIXED, **best["params"])
        m.fit(Xf, full["y_ratio"], categorical_feature=best["cats"])
        m.booster_.save_model(str(MODEL_DIR / f"lgb_q{int(q * 100)}.txt"))
    meta = {"features_num": NUM, "features_cat": best["cats"],
            "categories": Xf.attrs["categories"],
            "cqr_delta": CQR_DELTA, "cqr_by_usage": CQR_USAGE if CQR_BY_USAGE else {},
            "params": best["params"], "use_prpt": USE_PRPT, "n_trees_val": n_trees,
            "train_rows": int(len(full)), "train_until": FINAL_TRAIN_END,
            "note": "경매 예측 시 용도별 재산유형 고정 후 공매→경매 보정 계수 곱함 (calib_out/calibration.json)"}
    json.dump(meta, open(MODEL_DIR / "model_meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n저장: {MODEL_DIR}/lgb_q10·q50·q90.txt, model_meta.json, test_predictions.csv")


if __name__ == "__main__":
    main()
