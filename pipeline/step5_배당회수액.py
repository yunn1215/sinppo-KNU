"""5단계. 배당 50건 — 신보 회수액으로 채점 (낙찰가 예측 → 배당 엔진 → 회수액)

배당 엔진(dividend_engine.py)은 기존 step3_dividend.py 를 그대로 복사한 것 (계산 규칙 수정 없음).
입력
  - data/배당계산기_데이터_수집양식_v2_50건_최종.xlsx (등기부 구분·토지 감정가 반영본)
  - output/04_경매시험/auction_predictions.csv  (★ 새 모델: 현재 최저가 × 배수 + 경매 보정, 사건 교차 추정)
  - data/기존모델_경매예측_20261116.csv        (기존 모델: 감정가 기준 낙찰가율)

채점 기준값 = 실제 매각가를 같은 엔진에 넣었을 때의 신보 배당 (→ 낙찰가 예측이 틀려서 생기는 회수액 오차만 측정)
지표: 회수액 MAE(원), 'P10~P90 안에 실제 회수액' 비율, 회수 여부(1원이라도 받나) 정확도·F1
출력: output/05_배당회수액/ 신모델/, 기존모델/ (엔진 결과), recovery_eval.csv, recovery_by_case.csv
"""
import shutil
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

import dividend_engine as E
from common import CALIB_DIR, DIV_DIR, data_file, show, nfc

warnings.filterwarnings("ignore")


def run_engine(pred_dir, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    E.CALIB_DIR, E.DIV_DIR = pred_dir, out_dir
    E.main()
    return pd.read_csv(out_dir / "dividend_results.csv")


def score(name, r):
    r = r[r["상태"] == "계산 완료"].copy()
    a, p = r["실제가_신보배당"].values, r["회수_중앙값"].values
    return {"모델": name, "n": len(r),
            "회수액 MAE(백만원)": np.mean(np.abs(p - a)) / 1e6,
            "회수액 중앙 절대오차(백만원)": np.median(np.abs(p - a)) / 1e6,
            "Σ예측/Σ실제": p.sum() / a.sum() if a.sum() else np.nan,
            "P10~P90 안에 실제": np.mean((a >= r["회수_P10"].values - 0.5) & (a <= r["회수_P90"].values + 0.5)),
            "회수 여부 정확도": accuracy_score(a > 0.5, p > 0.5),
            "회수 여부 F1": f1_score(a > 0.5, p > 0.5, zero_division=0)}


def main():
    print("=" * 70, "\n5단계. 배당 50건 — 신보 회수액 채점\n" + "=" * 70)
    new = run_engine(CALIB_DIR, DIV_DIR / "신모델")

    old_in = DIV_DIR / "기존모델_입력"
    old_in.mkdir(parents=True, exist_ok=True)
    shutil.copy(data_file("기존모델_경매예측_20261116.csv"), old_in / "auction_predictions.csv")
    old = run_engine(old_in, DIV_DIR / "기존모델")

    rep = pd.DataFrame([score("기존 모델 (감정가 기준)", old), score("★ 새 모델 (최저가 × 배수)", new)])
    print("\n[신보 회수액 — 실제 매각가 기준 배당과 비교]"); show(rep)
    rep.to_csv(DIV_DIR / "recovery_eval.csv", index=False, encoding="utf-8-sig")

    cols = ["key", "감정가", "매각가", "신보채권액", "실제가_신보배당", "회수_중앙값", "회수_P10", "회수_P90", "p50"]
    m = new[new["상태"] == "계산 완료"][cols].merge(
        old[old["상태"] == "계산 완료"][["key", "회수_중앙값", "p50"]].rename(columns={"회수_중앙값": "기존_회수_중앙값", "p50": "기존_p50"}),
        on="key", how="left")
    m["실제 낙찰가율"] = m["매각가"] / m["감정가"]
    m["새_오차(백만원)"] = (m["회수_중앙값"] - m["실제가_신보배당"]) / 1e6
    m["기존_오차(백만원)"] = (m["기존_회수_중앙값"] - m["실제가_신보배당"]) / 1e6
    m.to_csv(DIV_DIR / "recovery_by_case.csv", index=False, encoding="utf-8-sig")
    big = m.reindex(m["기존_오차(백만원)"].abs().sort_values(ascending=False).index).head(8)
    print("\n[기존 모델 오차가 컸던 사건] (백만원)")
    print(big[["key", "실제 낙찰가율", "기존_p50", "p50", "실제가_신보배당", "기존_오차(백만원)", "새_오차(백만원)"]]
          .assign(실제가_신보배당=lambda d: d["실제가_신보배당"] / 1e6).round(2).to_string(index=False))
    print(f"\n저장: {nfc(DIV_DIR)}/recovery_eval.csv, recovery_by_case.csv, 신모델/, 기존모델/")


if __name__ == "__main__":
    main()
