"""공통 설정·함수 — 최종 낙찰가 모델 (2단계 구조: ① 매각확률 × ② 낙찰가 배수)

기존 pipeline/common.py (공매 낙찰가율 모델) 를 최종 모델에 맞춰 바꾼 파일.

폴더 구조 (이 파일 기준):
    (프로젝트 맨 위)
      data/                ← 입력 데이터 (읽기만 함)
      매각통계/  지역별 지가지수, 지가변동률/
      pipeline/            ← 이 코드
      dashboard/           ← 시뮬레이터 화면 템플릿 (7단계가 사용)
      output/              ← 실행 결과 (단계별 폴더)
      settings.json        ← 배당 계산 가정값
"""
import json
import os
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

# ───────────────────────── 경로 ─────────────────────────
ROOT = Path(__file__).resolve().parent.parent          # 프로젝트 맨 위


def nfc(s):
    return unicodedata.normalize("NFC", str(s))


def sub(parent, *parts):
    """한글 파일명 정규화(NFC/NFD)가 달라도 찾아줌"""
    p = Path(parent)
    for part in parts:
        hit = [q for q in p.iterdir() if nfc(q.name) == nfc(part)] if p.exists() else []
        p = hit[0] if hit else p / part
    return p


DATA_DIR = sub(ROOT, "data")
STAT_DIR = sub(ROOT, "매각통계")
LAND_DIR = sub(ROOT, "지역별 지가지수, 지가변동률")
OUT_ROOT = Path(os.environ.get("SINBO_OUT", ROOT / "output"))
PREP_DIR = OUT_ROOT / "01_데이터정리"
SELL_DIR = OUT_ROOT / "02_매각확률"
MULT_DIR = OUT_ROOT / "03_낙찰가배수"
CALIB_DIR = OUT_ROOT / "04_경매시험"          # 배당 엔진이 여기의 auction_predictions.csv 를 읽음
DIV_DIR = OUT_ROOT / "05_배당회수액"
MODEL_DIR = MULT_DIR                          # (기존 코드 호환용 이름)
for d in (PREP_DIR, SELL_DIR, MULT_DIR, CALIB_DIR, DIV_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ───────────────────────── 분석 설정 (어제 정한 분석 프로세스) ─────────────────────────
H = 1                                   # 예측 거리(개월). 1 = 경매 진행 중(다음 기일 약 1달 뒤). 시작 전을 보려면 6
RONE_PUB_LAG = 1                        # 지가변동률 공표 시차(개월) → 실제 지연 = H + RONE_PUB_LAG
CV_MONTHS = ["2026-03", "2026-04", "2026-05", "2026-06", "2026-07", "2026-08", "2026-09"]   # 롤링 CV 평가 월
TEST_MONTH = "2026-10"                  # 최종 test (한 번만 채점)
FINAL_TRAIN_END = "2026-09-30"          # 최종 모델 학습 마감 (10월 경매 시험셋과 겹치지 않게)
QS = [0.1, 0.5, 0.9]
Y_RANGE = (0.05, 1.5)                   # 낙찰가율 극단값 기준 (공매·경매 동일)
MULT_RANGE = (0.5, 20.0)                # 낙찰가 ÷ 최저가 극단값 기준
AUCTION_DROP = 0.7                      # 대구지방법원 유찰 1회당 최저가 저감 (30%)
SEED = 42
USAGES = ["아파트", "기타주거", "상가·업무", "공장·창고", "대지", "농지·임야"]
LAND_USAGES = ["대지", "농지·임야"]
BLDG_USAGES = ["아파트", "기타주거", "상가·업무", "공장·창고"]
SINBO_USAGE = ["공장·창고", "상가·업무", "대지", "농지·임야"]
PRPT_MAP = {"압류재산": "압류", "기타일반재산": "기타일반"}
LGB_FIXED = dict(subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                 random_state=SEED, verbose=-1, n_jobs=1)          # n_jobs=1: 실행할 때마다 같은 결과


def save_booster(booster, path):
    """LightGBM 모델 저장. Windows 에서 한글 경로(02_매각확률 …)는 LightGBM 이 직접 못 열어서 파이썬으로 씀"""
    Path(path).write_text(booster.model_to_string(), encoding="utf-8")


def read_booster(path):
    """LightGBM 모델 읽기 (한글 경로 대응: 파이썬으로 읽어서 문자열로 넘김)"""
    import lightgbm as lgb
    return lgb.Booster(model_str=Path(path).read_text(encoding="utf-8"))


def data_file(name):
    p = sub(DATA_DIR, name)
    if not p.exists():
        raise FileNotFoundError(f"{p} 가 없어요. data 폴더를 확인하세요.")
    return p


# ───────────────────────── 용도 6묶음 ─────────────────────────
LAND_DAEJI = ("대지", "잡종지", "주차장", "주유소용지", "종교용지", "수도용지", "기타토지",
              "도로", "공원", "구거", "하천", "묘지", "염전")


def usage6(m, s):
    """온비드 (중분류, 소분류) → 용도 6묶음 (기존과 동일)"""
    s = str(s)
    if s == "아파트":
        return "아파트"
    if m == "주거용건물" or s == "주/상용건물":
        return "기타주거"
    if s in ("공장시설", "기타제조공장", "창고시설", "공장용지", "창고용지"):
        return "공장·창고"
    if m == "토지":
        return "대지" if s in LAND_DAEJI else "농지·임야"
    return "상가·업무"


SIDO_GRP = {"광주광역시": "광주·전남", "전라남도": "광주·전남", "전남광주통합특별시": "광주·전남"}


def sido_grp_of(sido):
    return SIDO_GRP.get(sido, sido)


def split_addr(a):
    """주소 → (시도, 시군구). '대구광역시 달서구', '경상북도 포항시 북구' 처리 (가려진 주소 '* ' 도 시도·시군구까지는 읽음)"""
    p = [w for w in str(a).replace("*", " ").split() if w]
    sido = p[0] if p else None
    sgg = None
    if len(p) > 1:
        sgg = p[1]
        if len(p) > 2 and p[1].endswith("시") and p[2].endswith("구"):
            sgg = f"{p[1]} {p[2]}"
    return sido, sgg


# ───────────────────────── R-ONE 지가변동률 (기존 함수, 지연 개월을 인자로) ─────────────────────────
SIDO_SHORT = {"서울특별시": "서울", "부산광역시": "부산", "대구광역시": "대구", "인천광역시": "인천",
              "광주광역시": "광주", "대전광역시": "대전", "울산광역시": "울산", "세종특별자치시": "세종",
              "경기도": "경기", "강원특별자치도": "강원", "강원도": "강원", "충청북도": "충북",
              "충청남도": "충남", "전북특별자치도": "전북", "전라북도": "전북", "전라남도": "전남",
              "전남광주통합특별시": "전남", "경상북도": "경북", "경상남도": "경남", "제주특별자치도": "제주"}


def read_rone(fname, kind):
    t = pd.read_csv(sub(LAND_DIR, fname), encoding="cp949", header=None)
    hdr_m, hdr_k = t.iloc[0], t.iloc[1]
    body = t.iloc[3:] if kind == "rate" else t.iloc[2:]
    rows = []
    for j in range(4, t.shape[1]):
        if kind == "rate" and str(hdr_k[j]) != "변동률":
            continue
        ym = pd.to_datetime(str(hdr_m[j]).replace("년 ", "-").replace("월", ""), format="%Y-%m", errors="coerce")
        if pd.isna(ym):
            continue
        rows.append(pd.DataFrame({"a": body[1].values, "b": body[2].values, "c": body[3].values,
                                  "ym": ym.to_period("M"), "v": pd.to_numeric(body[j], errors="coerce").values}))
    return pd.concat(rows, ignore_index=True)


_RONE = None


def rone_tables():
    global _RONE
    if _RONE is None:
        rate = read_rone("(월) 지역별 지가변동률.csv", "rate")
        idx = read_rone("(월) 지역별 지가지수.csv", "index")
        _RONE = ({(r.a, r.b, r.c, r.ym): r.v for r in rate.itertuples()},
                 {(r.a, r.b, r.c, r.ym): r.v for r in idx.itertuples()})
    return _RONE


def region_keys(sido, sgg):
    a = SIDO_SHORT.get(sido, sido)
    if not isinstance(sgg, str) or not sgg.strip():
        return [(a, a, a)]
    parts = sgg.split()
    if len(parts) >= 2:
        return [(a, parts[0], parts[1]), (a, parts[0], parts[0]), (a, a, a)]
    return [(a, parts[0], parts[0]), (a, a, a)]


def attach_market(frame, date_col="opbd_date", lag=None):
    """예측 시점에 알 수 있던 지가변동률만 붙임: 매각월 − (H + 공표시차) 개월의 월 변동률(lr_1m),
    그 시점까지 3·6·12개월 누적(lr_3m·lr_6m·lr_12m). 기존 코드는 lag=2 고정이었음 (= H 1 + 공표 1)."""
    lag = H + RONE_PUB_LAG if lag is None else lag
    rate_d, idx_d = rone_tables()
    out = {k: [] for k in ["lr_1m", "lr_3m", "lr_6m", "lr_12m"]}
    for sido, sgg, d in zip(frame["sido"], frame["sgg"], frame[date_col]):
        if pd.isna(d):
            for k in out: out[k].append(None)
            continue
        m = pd.Timestamp(d).to_period("M") - lag
        v = {k: None for k in out}
        for key in region_keys(sido, sgg):
            if v["lr_1m"] is None and (key + (m,)) in rate_d:
                v["lr_1m"] = rate_d[key + (m,)]
            for n in (3, 6, 12):
                k = f"lr_{n}m"
                if v[k] is None and (key + (m,)) in idx_d and (key + (m - n,)) in idx_d:
                    v[k] = (idx_d[key + (m,)] / idx_d[key + (m - n,)] - 1) * 100
        for k in out: out[k].append(v[k])
    frame = frame.copy()
    for k, vals in out.items():
        frame[k] = pd.to_numeric(pd.Series(vals, index=frame.index), errors="coerce")
    return frame


# ───────────────────────── 피처 표 (기존 make_X 방식: 범주 → 목록 번호) ─────────────────────────
def _cat_str(v):
    if v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NA or v is pd.NaT:
        return "없음"
    t = str(v)
    return "없음" if t in ("nan", "<NA>", "None", "NaT") else t


def make_X(frame, num, cats, categories=None):
    """숫자 피처 + 범주 피처(학습 때 범주 목록의 번호, 없는 값은 NaN). categories 를 주면 그 목록 사용."""
    X = frame[num].apply(pd.to_numeric, errors="coerce").astype(float)
    used = {}
    for c in cats:
        vals = [_cat_str(v) for v in frame[c].tolist()]
        lst = list(categories[c]) if categories else sorted(set(vals))
        pos = {v: i for i, v in enumerate(lst)}
        X[c] = np.array([pos.get(v, np.nan) for v in vals], dtype=float)
        used[c] = lst
    X.attrs["categories"] = used
    return X


# ───────────────────────── 평가 지표 ─────────────────────────
def pinball(y, p, q):
    e = np.asarray(y) - np.asarray(p)
    return float(np.mean(np.maximum(q * e, (q - 1) * e)))


def interval_score(y, lo, hi, alpha=0.2):
    """구간 폭 + 벗어난 만큼 × 2/α (넓게 잡으면 폭 벌점, 좁게 잡아 놓치면 이탈 벌점). α=0.2 → P10~P90"""
    y, lo, hi = map(np.asarray, (y, lo, hi))
    return float(np.mean((hi - lo) + 2 / alpha * (lo - y) * (y < lo) + 2 / alpha * (y - hi) * (y > hi)))


def eval_quantile(name, y, P, w=None):
    """분위수 예측 채점 (y·P 는 낙찰가율 = 낙찰가 ÷ 감정가 단위로 통일)"""
    P = np.sort(np.asarray(P, dtype=float), axis=1)
    y = np.asarray(y, dtype=float)
    out = {"모델": name, "n": len(y),
           "MAE(P50,%p)": np.mean(np.abs(y - P[:, 1])) * 100,
           "RMSE(P50,%p)": np.sqrt(np.mean((y - P[:, 1]) ** 2)) * 100,
           "R2(P50)": 1 - np.sum((y - P[:, 1]) ** 2) / np.sum((y - y.mean()) ** 2) if len(y) > 1 else np.nan,
           "pinball": np.mean([pinball(y, P[:, i], q) for i, q in enumerate(QS)]),
           "interval_score": interval_score(y, P[:, 0], P[:, 2]),
           "구간폭(%p)": np.mean(P[:, 2] - P[:, 0]) * 100,
           "포함률(P10~P90)": np.mean((y >= P[:, 0]) & (y <= P[:, 2])),
           "P10아래": np.mean(y < P[:, 0]), "P50아래": np.mean(y < P[:, 1]), "P90아래": np.mean(y < P[:, 2]),
           "평균치우침(P50-실제,%p)": np.mean(P[:, 1] - y) * 100}
    if w is not None:
        w = np.asarray(w, dtype=float)
        out["Σ예측/Σ감정가"] = float((P[:, 1] * w).sum() / w.sum())
        out["Σ실제/Σ감정가"] = float((y * w).sum() / w.sum())
    return out


def eval_prob(name, y, p, thr=0.5):
    """매각확률 채점: AUC, Brier, log loss, 정확도·F1 (보조)"""
    from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss, accuracy_score, f1_score
    y = np.asarray(y).astype(int)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return {"모델": name, "n": len(y), "실제 매각률": y.mean(), "평균 예측확률": p.mean(),
            "AUC": roc_auc_score(y, p) if len(set(y)) > 1 else np.nan,
            "Brier": brier_score_loss(y, p), "logloss": log_loss(y, p, labels=[0, 1]),
            "정확도": accuracy_score(y, p >= thr), "F1": f1_score(y, p >= thr, zero_division=0)}


def calib_table(y, p, bins=(0, .02, .05, .1, .2, .3, .5, .7, 1.0)):
    """보정 곡선: 예측확률 구간별 실제 매각률"""
    t = pd.DataFrame({"y": np.asarray(y).astype(int), "p": np.asarray(p, dtype=float)})
    t["구간"] = pd.cut(t["p"], bins, include_lowest=True)
    return t.groupby("구간", observed=True).agg(n=("y", "size"), 평균예측=("p", "mean"), 실제매각률=("y", "mean")).reset_index()


def by_group_q(frame, y, P, col):
    P = np.sort(np.asarray(P, dtype=float), axis=1)
    rows = []
    for g, idx in frame.groupby(col).indices.items():
        r = eval_quantile(g, np.asarray(y)[idx], P[idx])
        rows.append({col: g, "n": r["n"], "MAE(%p)": r["MAE(P50,%p)"], "치우침(%p)": r["평균치우침(P50-실제,%p)"],
                     "포함률": r["포함률(P10~P90)"], "구간폭(%p)": r["구간폭(%p)"]})
    return pd.DataFrame(rows)


def show(df, n=None):
    df = df if n is None else df.head(n)
    print(df.round(4).to_string(index=False) if isinstance(df, pd.DataFrame) else df)


def save_json(obj, path):
    json.dump(obj, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
