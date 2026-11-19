"""6단계. 시뮬레이션 — ① 매각확률 × ② 낙찰가 배수를 회차별로 몬테카를로로 묶어 최종 낙찰가 분포 만들기

  진행 중 : 현재 최저가(지금까지 유찰 k회)에서 시작
  시작 전 : 최저가 = 감정가(k = 0)에서 시작
  매 회차: p = ①(현재 최저가율) → 팔리면 낙찰가 = 최저가 × ②배수(분위수 사이 보간 추출), 안 팔리면 최저가 × 0.7 로 다음 회차
  → 최종 낙찰가율(낙찰가 ÷ 감정가) P10/P50/P90, 예상 회차, MAX_ROUNDS 안에 안 팔릴 확률

경매 적용값은 4단계 결과(auction_shift.json) 사용:
  ① transfer 모델 + 경매 재보정(기본) 또는 9월 법원 매각통계 절편,  ② 배수 + 용도군 이동폭 + 구간 보정
대상: 배당 50건 (대상 사건 시트) — 두 시나리오 모두 계산하고, 둘 다 배당 엔진에 넣어 신보 회수액 분포까지 계산.
  (배당 엔진은 '팔린다면'의 낙찰가 분포로 계산. MAX_ROUNDS 안에 안 팔릴 확률은 따로 표시)
주의: 실사용 경매 보정값은 388건 전체로 추정했고 배당 50건이 그 안에 있음 → 여기 결과를 실제와 비교한 값은 성능 점수가 아님
      (성능 점수는 사건 교차 추정을 쓴 4·5단계 결과)
출력: output/06_시뮬레이션/ simulation_50.csv, recovery_sim_50.csv, 진행중/, 시작전/ (배당 엔진 결과)
"""
import json
import warnings

import numpy as np
import pandas as pd

import dividend_engine as E
from common import QS, AUCTION_DROP, CALIB_DIR, OUT_ROOT, SEED, show, nfc
from step4_경매시험 import load_cases, predict_mult, predict_sell, logit

warnings.filterwarnings("ignore")

SIM_DIR = OUT_ROOT / "06_시뮬레이션"
SIM_DIR.mkdir(parents=True, exist_ok=True)
MAX_ROUNDS = 8
SELL_CALIB = "platt"            # "platt": 4단계 경매 재보정(10월 매각공고 388건 기준, 매각 57%) / "stat": 9월 법원 매각통계 절편만(회차당 12~20%)
N_SIM = 4000


def sample_from_q(P, n, rng):
    """P10/P50/P90 (log) → 분위수 사이 선형 보간 추출 (꼬리는 반 구간 연장, 배당 엔진 sample_ratio 와 같은 방식)"""
    p10, p50, p90 = P
    lo, hi = p10 - 0.5 * (p50 - p10), p90 + 0.5 * (p90 - p50)
    return np.interp(rng.random(n), [0.0, 0.1, 0.5, 0.9, 1.0], [lo, p10, p50, p90, hi])


def round_table(case_row, start_rto, prpt, shift):
    """회차별 최저가율에서 ① 매각확률과 ② 배수 분위수를 미리 계산"""
    rows = pd.DataFrame([case_row] * MAX_ROUNDS).reset_index(drop=True)
    rows["rto"] = start_rto * AUCTION_DROP ** np.arange(MAX_ROUNDS)
    rows["누적_유찰횟수"] = ((100 - rows["rto"]) / 10).round().clip(lower=0)
    p = predict_sell(rows, prpt)
    u = case_row["usage6"]
    if SELL_CALIB == "platt":                                     # 경매 재보정 (4단계, 기본)
        a, b = shift["sell_platt"]
        p = 1 / (1 + np.exp(-(a + b * logit(p))))
    else:                                                         # 9월 법원 매각통계 절편만
        p = 1 / (1 + np.exp(-(logit(p) + shift["sell_logit_shift_sept_stat"].get(u, 0.0))))
    P = predict_mult(rows, prpt)
    g = shift["shift_groups"].get(u, "기타")
    s = shift["shift_log_by_group"].get(g, shift["shift_log_by_group"]["(전체)"])
    P = P + s
    P[:, 0] -= shift["widen_log"][0]; P[:, 2] += shift["widen_log"][1]
    return rows["rto"].values, p, np.sort(P, axis=1)


def simulate(case_row, start_rto, prpt, shift, rng):
    rto, p, P = round_table(case_row, start_rto, prpt, shift)
    price = np.full(N_SIM, np.nan)
    rnd = np.full(N_SIM, np.nan)
    alive = np.ones(N_SIM, bool)
    for k in range(MAX_ROUNDS):
        sell = alive & (rng.random(N_SIM) < p[k])
        n = sell.sum()
        if n:
            price[sell] = rto[k] / 100 * np.exp(sample_from_q(P[k], n, rng))
            rnd[sell] = k + 1
        alive &= ~sell
    sold = ~np.isnan(price)
    q = np.percentile(price[sold], [10, 50, 90]) if sold.any() else [np.nan] * 3
    return {"p10": q[0], "p50": q[1], "p90": q[2], "예상회차": np.nanmean(rnd),
            f"{MAX_ROUNDS}회 안에 미매각": 1 - sold.mean(), "첫회차 매각확률": p[0]}


def main():
    print("=" * 70, "\n6단계. 시뮬레이션 (① × ② 회차별 몬테카를로)\n" + "=" * 70)
    shift = json.load(open(CALIB_DIR / "auction_shift.json", encoding="utf-8"))
    prpt = shift["prpt_for_auction"]
    case = load_cases()
    keys = set(E.load_inputs()["target"]["key"])                  # 배당 50건 (수집양식 ⓪ 대상 사건)
    c50 = case[case["사건"].str.split(" ", n=1).str[1].isin(keys)].copy()
    print(f"배당 50건 중 시험셋에서 찾은 사건 {len(c50)}건")
    rng = np.random.default_rng(SEED)
    rows = []
    for _, r in c50.iterrows():
        a = simulate(r, r["rto"], prpt, shift, rng)
        b = simulate(r, 100.0, prpt, shift, rng)
        rows.append({"사건": r["사건"], "usage6": r["usage6"], "감정가": r["apslEvlAmt"], "10월 최저가율": r["rto"] / 100,
                     "경매 유찰횟수": r["경매_유찰횟수"], "실제 낙찰가율": r["y"],
                     **{f"진행중_{k}": v for k, v in a.items()}, **{f"시작전_{k}": v for k, v in b.items()}})
    sim = pd.DataFrame(rows)
    sim.to_csv(SIM_DIR / "simulation_50.csv", index=False, encoding="utf-8-sig")
    print("\n[요약] 중앙값 비교 (낙찰가율)")
    print(sim[["진행중_p50", "시작전_p50", "실제 낙찰가율", "진행중_예상회차", "시작전_예상회차",
               f"시작전_{MAX_ROUNDS}회 안에 미매각"]].describe().loc[["mean", "50%"]].round(3).to_string())
    print("\n[예시 5건]"); show(sim.head(5)[["사건", "usage6", "10월 최저가율", "실제 낙찰가율", "진행중_p10", "진행중_p50",
                                            "진행중_p90", "시작전_p10", "시작전_p50", "시작전_p90", "시작전_예상회차"]])

    # 두 시나리오 → 배당 엔진 (신보 회수액 분포)
    res = {}
    for sc in ("진행중", "시작전"):
        pin = SIM_DIR / f"{sc}_입력"
        pin.mkdir(exist_ok=True)
        sim.rename(columns={f"{sc}_p10": "p10", f"{sc}_p50": "p50", f"{sc}_p90": "p90"})[["사건", "usage6", "p10", "p50", "p90"]] \
            .to_csv(pin / "auction_predictions.csv", index=False, encoding="utf-8-sig")
        E.CALIB_DIR, E.DIV_DIR = pin, SIM_DIR / sc
        E.DIV_DIR.mkdir(exist_ok=True)
        E.main()
        res[sc] = pd.read_csv(SIM_DIR / sc / "dividend_results.csv")

    sim["key"] = sim["사건"].str.split(" ", n=1).str[1]
    out = sim[["사건", "key", "usage6", "감정가", "10월 최저가율", "실제 낙찰가율",
               "진행중_p50", f"진행중_{MAX_ROUNDS}회 안에 미매각", "시작전_p50", f"시작전_{MAX_ROUNDS}회 안에 미매각"]]
    for sc, r in res.items():
        r = r[["key", "상태", "신보채권액", "실제가_신보배당", "회수_중앙값", "회수_P10", "회수_P90", "영원확률"]]
        if sc == "시작전":
            r = r.drop(columns=["신보채권액", "실제가_신보배당"])
        out = out.merge(r.rename(columns={c: f"{sc}_{c}" for c in r.columns if c not in ("key", "신보채권액", "실제가_신보배당")}),
                        on="key", how="left")
    out.to_csv(SIM_DIR / "recovery_sim_50.csv", index=False, encoding="utf-8-sig")
    ok = out[out["진행중_상태"] == "계산 완료"]
    print(f"\n[신보 회수액 분포 — 배당 엔진 계산 완료: 진행중 {len(ok)}건 / 시작전 {(out['시작전_상태'] == '계산 완료').sum()}건]")
    summ = pd.DataFrame({
        "시나리오": ["진행 중 (10월 최저가에서)", "시작 전 (감정가에서)"],
        "회수 중앙값 합계(백만원)": [ok["진행중_회수_중앙값"].sum() / 1e6, ok["시작전_회수_중앙값"].sum() / 1e6],
        "0원 확률 평균": [ok["진행중_영원확률"].mean(), ok["시작전_영원확률"].mean()],
        f"{MAX_ROUNDS}회 안에 미매각 평균": [ok[f"진행중_{MAX_ROUNDS}회 안에 미매각"].mean(), ok[f"시작전_{MAX_ROUNDS}회 안에 미매각"].mean()]})
    summ["실제 배당 합계(백만원) — 참고, in-sample"] = ok["실제가_신보배당"].sum() / 1e6
    show(summ)
    print(f"\n저장: {nfc(SIM_DIR)}/simulation_50.csv, recovery_sim_50.csv, 진행중/, 시작전/")


if __name__ == "__main__":
    main()
