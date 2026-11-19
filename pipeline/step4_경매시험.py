"""4단계. 경매 시험 — 공매로 학습한 ①·② 를 2026-10 대구·경북 경매 388건에 적용해 채점 (시험만, 학습·설정 선택에 쓰지 않음)

입력
  - data/경매_시험셋_388건_정리.csv : 388건 (매각 220 / 미매각 168), 10월 최저가·유찰 횟수(31073 = 0회 반영)
  - data/auction_features.csv       : 매각 220건의 공시지가·건축물대장 (팀 0단계 결과)
  - data/기존모델_경매예측_20261116.csv : 기존 모델(감정가 기준)의 경매 예측 — 비교 기준선
  - data/매각통계/26.9 *            : 9월 법원 매각통계 (① 절편 보정 참고값, 10월이 아닌 달)
  - data/감정가구성_40건_정리본.csv  : 공장 기계기구 비중 (민감도 분석)

경매에 적용할 때 맞춘 것
  - 현재 최저가율 = 10월 최저매각가 ÷ 감정가
  - 경매는 유찰 1회에 30%, 공매는 약 10% 내려가서 '유찰 횟수' 의미가 다름 → ② 에는 공매 환산 유찰횟수 = (100 − 최저가율)/10,
    ① 에는 횟수를 빼고 가격 수준으로 학습한 transfer 모델 사용
  - 재산유형은 경매에 없는 항목 → PRPT_FOR_AUCTION 로 고정 (기본 '압류', 비교로 '기타일반'도 출력)
  - 최근 시장 지표는 예측 시점(9월 말) 이전 공매 낙찰로 계산

채점
  ① 매각확률 (388건, 1214 제외): AUC·Brier·log loss·보정 곡선 / 주의: 미매각에 변경·취하가 섞여 있을 수 있음
     · 공매 그대로 / 9월 법원 매각통계로 절편만 맞춤 / ★ 경매 재보정(기울기·절편, 사건 단위 2-fold 교차)
  ② 낙찰가 (매각 219건): 낙찰가율 단위 MAE·pinball·interval score·포함률·하회율, 용도별, 기존 모델과 비교
     · 보정 없음 (공매 모델 그대로)
     · ★ 경매 보정: 용도군별 log 배수 이동폭을 '다른 절반의 사건'으로 추정 (사건 단위 2-fold 교차) → 자기 사건은 보지 않음
     · 사건 단위 부트스트랩 신뢰구간

출력: output/04_경매시험/ auction_predictions.csv (배당 엔진 입력 형식), auction_eval.csv, sell_eval.csv, auction_shift.json 등
"""
import json
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb

from common import (H, QS, Y_RANGE, AUCTION_DROP, PREP_DIR, SELL_DIR, MULT_DIR, CALIB_DIR, STAT_DIR, SEED,
                    data_file, sido_grp_of, split_addr, attach_market, make_X, eval_quantile, eval_prob,
                    calib_table, by_group_q, show, save_json, nfc, sub, read_booster)

warnings.filterwarnings("ignore")

PRPT_FOR_AUCTION = "압류"           # 경매 물건에 붙일 재산유형 (신보 회수 대상과 성격이 가까운 압류재산)
SHIFT_GROUPS = {"공장·창고": "공장·창고", "상가·업무": "상가·업무", "대지": "토지", "농지·임야": "토지"}
SHRINK_K = 20                       # 용도군 이동폭을 건수가 적으면 전체 이동폭 쪽으로 축소
MACH_RECOVERY = 0.5                 # 민감도: 기계기구 몫이 부동산 몫 대비 몇 배 비율로 팔리는지 (1.0 = 구분 안 함)
N_BOOT = 1000
K_MAX = 5                           # ① 경매 재보정에 넣는 유찰 횟수 구간 (0,1,2,3,4,5회 이상)
SELL_K_C = 1.0                      # 유찰 횟수 계수 L2 규제 (건수가 적은 구간은 0 쪽으로 줄임)


def auction_usage(u, court_u):
    if u in ("공장", "창고"): return "공장·창고"
    if u == "상가": return "상가·업무"
    if court_u == "공장용지": return "공장·창고"
    if court_u in ("대지", "잡종지"): return "대지"
    return "농지·임야"


def load_cases():
    d = pd.read_csv(data_file("경매_시험셋_388건_정리.csv"))
    d["사건"] = d["법원"] + " " + d["사건번호"] + "-" + d["물건번호"].astype(int).astype(str)
    f = pd.read_csv(data_file("auction_features.csv"))
    d = d.merge(f.drop(columns=["addr_clean", "vw_status", "vw_matched", "지번일치", "price_year"], errors="ignore"),
                on="사건", how="left")
    d["usage6"] = [auction_usage(u, c) for u, c in zip(d["용도(우리)"], d["법원 표기 용도"])]
    d[["sido", "sgg"]] = [split_addr(a) for a in d["주소"]]
    d["sido_grp"] = d["sido"].map(sido_grp_of)
    d["is_daegu_gb"] = 1
    d["apslEvlAmt"] = pd.to_numeric(d["감정가(원)"], errors="coerce")
    d["log_apsl"] = np.log(d["apslEvlAmt"])
    d["landSqms"] = pd.to_numeric(d["토지면적(㎡)"], errors="coerce")
    d["bldSqms"] = pd.to_numeric(d["건물면적(㎡)"], errors="coerce")
    d["has_building"] = d["usage6"].isin(["공장·창고", "상가·업무"]).astype(float)
    d["rto"] = pd.to_numeric(d["최저가율"], errors="coerce") * 100
    d["누적_유찰횟수"] = ((100 - d["rto"]) / 10).round().clip(lower=0)      # 공매 환산 (경매 30% ↔ 공매 10%)
    d["경매_유찰횟수"] = d["유찰 횟수(계산)"]
    d["이전_낙찰횟수"] = 0
    d["재매각"] = 0
    d["회차_경과일"] = np.nan
    d["opbd_date"] = pd.Timestamp("2026-10-15")
    d["opbd_qtr"] = 4
    d["개찰월"] = 10
    d["prpt"] = PRPT_FOR_AUCTION
    d["label"] = (d["10월 결과"] == "매각").astype(int)
    d["매각가"] = pd.to_numeric(d["매각가(원)"], errors="coerce")
    d["y"] = d["매각가"] / d["apslEvlAmt"]
    d = attach_market(d)
    return d


def recent_market(case, prpt):
    """예측 시점(10월 기준 H개월 전까지) 공매 낙찰로 만든 최근 지표 — 2단계 학습표와 같은 정의"""
    g = pd.read_csv(PREP_DIR / "02_배수_학습표.csv", low_memory=False)
    last = pd.Period("2026-10", "M") - H
    months = [str(last - k) for k in range(3)]
    s = g[g["월"].isin(months)]
    r1 = s.groupby("usage6")["y_ratio"].median()
    r2 = s[s["prpt"] == prpt].groupby("usage6")["log_mult"].median()
    case = case.copy()
    case["최근_낙찰가율_용도"] = case["usage6"].map(r1)
    case["최근_배수_용도유형"] = case["usage6"].map(r2)
    return case


def load_booster(path):
    return read_booster(path)


def predict_mult(case, prpt):
    meta = json.load(open(MULT_DIR / "mult_meta.json", encoding="utf-8"))
    c = recent_market(case.assign(prpt=prpt), prpt)
    X = make_X(c, meta["num"], meta["cats"], meta["categories"]).to_numpy(dtype=float)
    P = np.sort(np.column_stack([load_booster(MULT_DIR / f"mult_q{int(q * 100)}.txt").predict(X) for q in QS]), axis=1)
    d = np.array([meta["cqr_by_usage"].get(u, meta["cqr_all"]) for u in c["usage6"]])
    P[:, 0] -= d[:, 0]; P[:, 2] += d[:, 1]
    return np.sort(P, axis=1)                                   # log 배수


def predict_sell(case, prpt):
    meta = json.load(open(SELL_DIR / "sell_meta.json", encoding="utf-8"))["transfer"]
    X = make_X(case.assign(prpt=prpt), meta["num"], meta["cats"], meta["categories"]).to_numpy(dtype=float)
    return load_booster(SELL_DIR / "sell_transfer.txt").predict(X)


def ratio(case, P_log):
    return (case["rto"].values[:, None] / 100) * np.exp(P_log)


# ── 9월 법원 매각통계 매각율 (① 절편 보정 참고값) ──
STAT_KEYS = {"상가·업무": [("상가,오피스텔,근린시설", "소계")], "대지": [("대지,임야,전답", "대지")],
             "농지·임야": [("대지,임야,전답", "임야"), ("대지,임야,전답", "전답")], "공장·창고": [("기타", None)]}


def sept_sale_rate():
    def rd(name):
        p = sub(STAT_DIR, name)
        if not p.exists():
            return None
        t = pd.read_csv(p, encoding="utf-8-sig")
        t.columns = ["대", "소", "경매건수", "매각건수", "감정가", "매각가", "매각율", "매각가율"]
        t["대"] = t["대"].ffill()
        for c in ["경매건수", "매각건수"]:
            t[c] = pd.to_numeric(t[c].astype(str).str.replace(",", ""), errors="coerce")
        return t
    out = {}
    tabs = [t for t in (rd("26.9 대구 용도별 매각통계.csv"), rd("26.9 경북 용도별 매각통계.csv")) if t is not None]
    nat = rd("26.9 전국 용도별 매각통계.csv")
    for u, keys in STAT_KEYS.items():
        n = s = 0
        for t in (tabs if u != "공장·창고" else ([nat] if nat is not None else [])):
            for a, b in keys:
                r = t[(t["대"] == a) & ((t["소"] == b) if b else t["소"].isna())]
                n += r["경매건수"].sum(); s += r["매각건수"].sum()
        out[u] = s / n if n else np.nan
    return out


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _platt_fit(z, y):
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression(C=1e6, max_iter=1000).fit(z.reshape(-1, 1), y)
    return float(m.intercept_[0]), float(m.coef_[0, 0])


def crossfit_platt(case, y, p):
    """① 경매용 재보정: logit(p_경매) = a + b·logit(p_공매 transfer). 사건 단위 2-fold 교차 추정(평가용) + 전체 추정(실사용).
    공매는 낮은 최저가율에서도 잘 안 팔려서 경매에 그대로 쓰면 매각확률이 크게 낮게 나옴 → 기울기까지 맞춤.
    주의: 시험셋 388건은 10월 매각공고 전수(지분 제외)인데 매각 57%로, 9월 법원 매각통계의 회차당 매각율(12~20%)과 차이가 큼.
          원인은 확인 못 함(통계의 '경매건수' 집계 기준 차이, 미매각에 변경·취하 포함 등 가능) → 절대 수준은 조심해서 해석."""
    rng = np.random.default_rng(SEED + 1)
    key = (case["법원"] + " " + case["사건번호"]).values
    uk = np.unique(key)
    half = dict(zip(uk, rng.permutation(len(uk)) % 2))
    fold = np.array([half[k] for k in key])
    z = logit(p)
    out = np.zeros(len(y))
    for f in (0, 1):
        a, b = _platt_fit(z[fold != f], y[fold != f])
        out[fold == f] = 1 / (1 + np.exp(-(a + b * z[fold == f])))
    return out, _platt_fit(z, y)


def auction_k(rto):
    """최저가율(%) → 경매 유찰 횟수 (최저가 = 감정가 × 0.7^k). 화면(JS)에서도 같은 규칙으로 계산"""
    k = np.round(np.log(np.clip(np.asarray(rto, dtype=float), 1e-6, None) / 100) / np.log(AUCTION_DROP))
    return np.clip(k, 0, K_MAX).astype(int)


def _platt_k_design(z, k):
    return np.column_stack([z, np.eye(K_MAX + 1)[k]])


def _platt_k_fit(z, k, y):
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression(C=SELL_K_C, max_iter=2000).fit(_platt_k_design(z, k), y)
    return {"a": float(m.intercept_[0]), "b": float(m.coef_[0, 0]), "k": [float(v) for v in m.coef_[0, 1:]]}


def platt_k_predict(par, p, k):
    z = par["a"] + par["b"] * logit(p) + np.asarray(par["k"])[k]
    return 1 / (1 + np.exp(-z))


def crossfit_platt_k(case, y, p):
    """① 경매용 재보정 + 유찰 횟수: logit(p_경매) = a + b·logit(p_공매) + c_k (k = 경매 유찰 횟수 0~5+).
    Platt 만으로는 공매 모델의 최저가율 기울기가 경매와 달라 신건(유찰 0회, 최저가 100%)을 과대예측함
    (10월 신건 32건 중 매각 0건인데 예측 17%) → 유찰 횟수별 차이를 따로 맞춤. 사건 단위 2-fold 교차(평가) + 전체(실사용)"""
    rng = np.random.default_rng(SEED + 1)
    key = (case["법원"] + " " + case["사건번호"]).values
    uk = np.unique(key)
    half = dict(zip(uk, rng.permutation(len(uk)) % 2))
    fold = np.array([half[k] for k in key])
    z = logit(p)
    kk = auction_k(case["rto"].values)
    out = np.zeros(len(y))
    for f in (0, 1):
        par = _platt_k_fit(z[fold != f], kk[fold != f], y[fold != f])
        out[fold == f] = platt_k_predict(par, p[fold == f], kk[fold == f])
    return out, _platt_k_fit(z, kk, y)


def boot_ci(case, y, P, n=N_BOOT):
    """사건 단위 부트스트랩 (같은 사건의 여러 물건은 함께 뽑음) — MAE 95% 구간"""
    rng = np.random.default_rng(SEED)
    key = case["법원"] + " " + case["사건번호"]
    groups = list(case.groupby(key).indices.values())
    vals = []
    for _ in range(n):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        vals.append(np.mean(np.abs(y[idx] - P[idx, 1])) * 100)
    return np.percentile(vals, [2.5, 97.5])


def _shift_fit(resid, grp, mask):
    s_all = float(np.median(resid[mask]))
    out = {"(전체)": s_all}
    for g in np.unique(grp):
        m = mask & (grp == g)
        n = int(m.sum())
        out[g] = float((n * np.median(resid[m]) + SHRINK_K * s_all) / (n + SHRINK_K)) if n else s_all
    return out


def _widen_fit(ylog, Plog, mask):
    """경매용 구간 보정 (위·아래 따로, 90% 분위수) — 공매보다 경매 낙찰가가 더 흩어져서 필요"""
    n = int(mask.sum())
    k = min(1.0, np.ceil((n + 1) * 0.9) / n)
    return [float(np.quantile(Plog[mask, 0] - ylog[mask], k)), float(np.quantile(ylog[mask] - Plog[mask, 2], k))]


def crossfit_shift(case, y, P_log):
    """용도군별 log 배수 이동폭 + 구간 보정을 사건 단위 2-fold 로 교차 추정 (자기 사건이 들어간 절반은 쓰지 않음)
    반환: 교차 추정한 log 예측(P), 실사용용(219건 전부) 이동폭·구간 보정"""
    rng = np.random.default_rng(SEED)
    key = (case["법원"] + " " + case["사건번호"]).values
    uk = np.unique(key)
    half = dict(zip(uk, rng.permutation(len(uk)) % 2))
    fold = np.array([half[k] for k in key])
    grp = case["usage6"].map(SHIFT_GROUPS).fillna("기타").values
    lr = np.log(case["rto"].values / 100)
    ylog = np.log(y) - lr                                          # 실제 log 배수
    resid = ylog - P_log[:, 1]
    P_cf = P_log.copy()
    for f in (0, 1):
        src, tgt = fold != f, fold == f
        sh = _shift_fit(resid, grp, src)
        s = np.array([sh.get(g, sh["(전체)"]) for g in grp])
        Ps = P_log + s[:, None]
        wd = _widen_fit(ylog, Ps, src)
        P_cf[tgt] = Ps[tgt]
        P_cf[tgt, 0] -= wd[0]; P_cf[tgt, 2] += wd[1]
    sh = _shift_fit(resid, grp, np.ones(len(y), bool))
    Ps = P_log + np.array([sh.get(g, sh["(전체)"]) for g in grp])[:, None]
    wd = _widen_fit(ylog, Ps, np.ones(len(y), bool))
    return np.sort(P_cf, axis=1), sh, wd


def main():
    print("=" * 70, f"\n4단계. 경매 시험 (재산유형 고정: {PRPT_FOR_AUCTION})\n" + "=" * 70)
    case = load_cases()
    case = case[~(case["label"].eq(1) & ~case["y"].between(*Y_RANGE))].copy()      # 1214 (낙찰가율 2.64) 제외
    print(f"경매 {len(case)}건 (매각 {case['label'].sum()} / 미매각 {(1 - case['label']).sum()}) | 1214 제외")

    # ── ① 매각확률 ──
    p_raw = predict_sell(case, PRPT_FOR_AUCTION)
    rate9 = sept_sale_rate()
    shift_u = {u: logit(rate9[u]) - logit(p_raw[(case.usage6 == u).values].mean())
               for u in case["usage6"].unique() if not np.isnan(rate9.get(u, np.nan))}
    p_stat = 1 / (1 + np.exp(-(logit(p_raw) + case["usage6"].map(shift_u).fillna(0).values)))
    y1 = case["label"].values
    base9 = case["usage6"].map(rate9).values
    p_cf, platt = crossfit_platt(case, y1, p_raw)
    p_cfk, platt_k = crossfit_platt_k(case, y1, p_raw)
    sell_rep = pd.DataFrame([eval_prob("기준선: 9월 매각통계 매각율(용도별 상수)", y1, base9),
                             eval_prob("① transfer (공매 학습 그대로)", y1, p_raw),
                             eval_prob("① transfer + 9월 매각통계 절편 보정", y1, p_stat),
                             eval_prob("① transfer + 경매 재보정 (사건 교차)", y1, p_cf),
                             eval_prob("★ ① transfer + 경매 재보정 + 유찰 횟수 (사건 교차)", y1, p_cfk)])
    print("\n[① 매각확률 — 경매 10월]"); show(sell_rep)
    print("9월 법원 매각율(회차당):", {k: round(v, 3) for k, v in rate9.items()},
          f"| 시험셋 실제 매각률 {y1.mean():.1%} (10월 매각공고 전수) → 법원 통계와 집계 기준이 달라 절대 수준 비교는 어려움, AUC(순서)로 판단")
    by = [eval_prob(f"transfer · {u}", y1[idx], p_raw[idx]) for u, idx in case.groupby("usage6").indices.items()]
    print("\n[①] 용도별"); show(pd.DataFrame(by))
    print("\n[①] 경매 유찰 횟수별 실제 매각률 vs 예측 평균")
    print(case.assign(p=p_raw, p2=p_cf, p3=p_cfk, k=auction_k(case["rto"].values)).groupby("k").agg(n=("label", "size"), 실제=("label", "mean"),
          예측_공매그대로=("p", "mean"), 예측_재보정=("p2", "mean"), 예측_재보정_유찰=("p3", "mean")).round(3).to_string())
    print(f"경매 재보정(전체 추정): logit p = {platt[0]:.3f} + {platt[1]:.3f} × logit p_공매")
    print(f"★ 실사용: logit p = {platt_k['a']:.3f} + {platt_k['b']:.3f} × logit p_공매 + 유찰 0~5회 계수 {np.round(platt_k['k'], 3).tolist()}")
    sell_rep.to_csv(CALIB_DIR / "sell_eval.csv", index=False, encoding="utf-8-sig")
    calib_table(y1, p_raw).to_csv(CALIB_DIR / "sell_calibration.csv", index=False, encoding="utf-8-sig")

    # ── ② 낙찰가 (매각 219건) ──
    sold = case[case["label"] == 1].copy()
    y = sold["y"].values; w = sold["apslEvlAmt"].values
    P_log = predict_mult(sold, PRPT_FOR_AUCTION)
    P0 = ratio(sold, P_log)
    P_alt = ratio(sold, predict_mult(sold, "기타일반"))
    P_cf_log, final_shift, final_widen = crossfit_shift(sold, y, P_log)
    P_cf = ratio(sold, P_cf_log)
    old = pd.read_csv(data_file("기존모델_경매예측_20261116.csv"))
    om = sold[["사건"]].merge(old[["사건", "p10", "p50", "p90"]], on="사건", how="left")
    has_old = om["p50"].notna().values
    P_old = om[["p10", "p50", "p90"]].values
    P_min = np.column_stack([sold["rto"].values / 100] * 3)
    rows = [eval_quantile("기준선: 최저가 그대로", y, P_min, w),
            eval_quantile(f"기존 모델 (감정가 기준, n={has_old.sum()})", y[has_old], P_old[has_old], w[has_old]),
            eval_quantile("② 보정 없음 (재산유형 압류)", y, P0, w),
            eval_quantile("② 보정 없음 (재산유형 기타일반)", y, P_alt, w),
            eval_quantile("★ ② + 경매 보정 (사건 교차)", y, P_cf, w)]
    rep = pd.DataFrame(rows)
    ci = {r["모델"]: boot_ci(sold if not r["모델"].startswith("기존") else sold[has_old],
                            y if not r["모델"].startswith("기존") else y[has_old],
                            P if not r["모델"].startswith("기존") else P_old[has_old])
          for r, P in zip(rows, [P_min, P_old, P0, P_alt, P_cf])}
    rep["MAE 95%구간"] = rep["모델"].map(lambda m: f"{ci[m][0]:.1f}~{ci[m][1]:.1f}")
    print(f"\n[② 낙찰가 — 경매 매각 {len(sold)}건, 낙찰가율 단위]"); show(rep)
    print("\n경매 보정 이동폭(log 배수, 219건 전부로 추정 — 실사용값):", {k: round(v, 3) for k, v in final_shift.items()},
          "→ 배수", {k: round(float(np.exp(v)), 3) for k, v in final_shift.items()},
          "| 구간 보정 [아래, 위]", np.round(final_widen, 3).tolist())
    print("\n[★] 용도별"); show(by_group_q(sold, y, P_cf, "usage6"))
    print("\n[기존 모델] 용도별"); show(by_group_q(sold[has_old], y[has_old], P_old[has_old], "usage6"))

    # ── 공장 기계기구 민감도 ──
    comp = pd.read_csv(data_file("감정가구성_40건_정리본.csv"))
    comp["사건"] = comp["법원"] + " " + comp["사건번호"] + "-" + comp["물건번호"].astype(int).astype(str)
    mshare = sold["사건"].map(comp.set_index("사건")["기계기구 비중(%)"]).fillna(0).values / 100
    adj = (1 - mshare) + MACH_RECOVERY * mshare
    m_f = (sold["usage6"] == "공장·창고").values & (mshare > 0)
    if m_f.sum():
        print(f"\n[민감도] 기계기구 비중이 확인된 공장 {m_f.sum()}건 — 기계 몫이 부동산 몫의 {MACH_RECOVERY:.0%} 수준으로 팔린다고 가정하면")
        show(pd.DataFrame([eval_quantile("★ 그대로", y[m_f], P_cf[m_f]),
                           eval_quantile(f"★ 기계 몫 ×{MACH_RECOVERY}", y[m_f], P_cf[m_f] * adj[m_f, None])]))

    # ── 저장: 배당 엔진 입력 형식 (사건, usage6, p10/p50/p90 = 낙찰가 ÷ 감정가) ──
    out = sold[["사건", "용도(우리)", "usage6", "apslEvlAmt", "매각가", "y", "rto", "경매_유찰횟수"]].rename(columns={"용도(우리)": "용도"})
    out[["p10", "p50", "p90"]] = P_cf
    out[["p10_보정없음", "p50_보정없음", "p90_보정없음"]] = P0
    out.to_csv(CALIB_DIR / "auction_predictions.csv", index=False, encoding="utf-8-sig")
    rep.to_csv(CALIB_DIR / "auction_eval.csv", index=False, encoding="utf-8-sig")
    allp = case[["사건", "usage6", "apslEvlAmt", "rto", "경매_유찰횟수", "label", "y"]].copy()
    allp["p_sell_공매그대로"] = p_raw
    allp["p_sell_platt"] = p_cf
    allp["p_sell"] = p_cfk
    allp.to_csv(CALIB_DIR / "auction_sell_predictions.csv", index=False, encoding="utf-8-sig")
    save_json({"prpt_for_auction": PRPT_FOR_AUCTION, "shift_log_by_group": final_shift, "shift_groups": SHIFT_GROUPS,
               "widen_log": final_widen,
               "sell_logit_shift_sept_stat": shift_u, "sell_platt": list(platt), "sell_platt_k": platt_k, "sell_k_max": K_MAX, "auction_drop": AUCTION_DROP,
               "note": "실사용: 낙찰가 = 현재 최저가 × exp(② 예측 + 용도군 이동폭). 이동폭은 2026-10 경매 219건 전체로 추정"},
              CALIB_DIR / "auction_shift.json")
    print(f"\n저장: {nfc(CALIB_DIR)}/auction_predictions.csv (배당 엔진 입력), auction_eval.csv, sell_eval.csv, auction_shift.json")


if __name__ == "__main__":
    main()
