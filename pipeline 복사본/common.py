"""공통 설정·함수 (코랩 노트북 4개에서 같이 쓰던 부분을 한 곳에 모음)

폴더 구조 (이 파일 기준):
    신보프로젝트/
      data/                       ← 원본 데이터 (팀이 준 것, 읽기만 함)
      매각통계/                    ← 법원 매각통계 12개
      지역별 지가지수, 지가변동률/   ← R-ONE 2개
      pipeline/                   ← 이 코드
      output/                     ← 실행 결과 (model_out / calib_out / dividend_out)
"""
import os
import json
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

# ───────────────────────── 경로 ─────────────────────────
ROOT = Path(__file__).resolve().parent.parent


def nfc(s):
    return unicodedata.normalize("NFC", str(s))


def sub(parent, *parts):
    """macOS·드라이브에서 한글 파일명 정규화(NFC/NFD)가 달라도 찾아줌"""
    p = Path(parent)
    for part in parts:
        hit = [q for q in p.iterdir() if nfc(q.name) == nfc(part)] if p.exists() else []
        p = hit[0] if hit else p / part
    return p


DATA_DIR = sub(ROOT, "data")
STAT_DIR = sub(ROOT, "매각통계")
LAND_DIR = sub(ROOT, "지역별 지가지수, 지가변동률")
OUT_ROOT = Path(os.environ.get("SINBO_OUT", ROOT / "output"))
MODEL_DIR = OUT_ROOT / "model_out"
CALIB_DIR = OUT_ROOT / "calib_out"
DIV_DIR = OUT_ROOT / "dividend_out"
for d in (MODEL_DIR, CALIB_DIR, DIV_DIR):
    d.mkdir(parents=True, exist_ok=True)

QS = [0.1, 0.5, 0.9]
# 건물 없는 물건(토지)의 건물면적: 학습 데이터(온비드)는 0, 노트북은 예측 때 빈칸(NaN). 기본값은 노트북과 같게 False.
# True로 바꾸면 학습 데이터와 같은 0으로 예측 (경매 219건 MAE 14.09%p → 13.90%p)
LAND_BLD_AS_ZERO = False
Y_RANGE = (0.05, 1.5)          # 낙찰가율 극단값 기준 (공매 학습·경매 평가 동일)
USAGES = ["아파트", "기타주거", "상가·업무", "공장·창고", "대지", "농지·임야"]
LAND_USAGES = ["대지", "농지·임야"]
BLDG_USAGES = ["아파트", "기타주거", "상가·업무", "공장·창고"]


def latest_train_csv():
    fs = sorted(p for p in DATA_DIR.iterdir() if nfc(p.name).startswith("train_features_") and p.suffix == ".csv")
    if not fs:
        raise FileNotFoundError(f"{DATA_DIR} 에 train_features_*.csv 가 없어요.")
    return fs[-1]


# ───────────────────────── 용도 6묶음 ─────────────────────────
LAND_DAEJI = ("대지", "잡종지", "주차장", "주유소용지", "종교용지", "수도용지", "기타토지",
              "도로", "공원", "구거", "하천", "묘지", "염전")


def usage6(m, s):
    """온비드 (중분류, 소분류) → 용도 6묶음"""
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


def load_gongmae(verbose=True):
    """온비드 학습 표 → 정리된 DataFrame (중복·결측·극단값 제거, usage6/prpt/분기 추가)"""
    path = latest_train_csv()
    raw = pd.read_csv(path, low_memory=False, dtype={"sgg_cd": str, "pnu": str})
    df = raw.copy()
    df["opbd_date"] = pd.to_datetime(df["opbd_date"], errors="coerce")
    df = df.dropna(subset=["opbd_date", "y_ratio", "apslEvlAmt"])
    df = df.sort_values("opbd_date").drop_duplicates("cltrMngNo", keep="last")
    n0 = len(df)
    df = df[df["y_ratio"].between(*Y_RANGE)].copy()
    df["usage6"] = [usage6(m, s) for m, s in zip(df["cltrUsgMclsCtgrNm"], df["cltrUsgSclsCtgrNm"])]
    df["prpt"] = df["prptDivNm"].map({"압류재산": "압류", "기타일반재산": "기타일반"}).fillna("기타")
    df["opbd_qtr"] = df["opbd_date"].dt.quarter
    if verbose:
        print(f"입력: {nfc(path.name)}  {len(raw)} → 중복·결측 정리 {n0} → 극단값 정리 {len(df)}건")
    return df


# ───────────────────────── R-ONE 지가변동률 ─────────────────────────
SIDO_SHORT = {"서울특별시": "서울", "부산광역시": "부산", "대구광역시": "대구", "인천광역시": "인천",
              "광주광역시": "광주", "대전광역시": "대전", "울산광역시": "울산", "세종특별자치시": "세종",
              "경기도": "경기", "강원특별자치도": "강원", "강원도": "강원", "충청북도": "충북",
              "충청남도": "충남", "전북특별자치도": "전북", "전라북도": "전북", "전라남도": "전남",
              "경상북도": "경북", "경상남도": "경남", "제주특별자치도": "제주"}


def read_rone(fname, kind):
    """R-ONE 넓은 표 → (시도, 시군구, 구, 월, 값) 긴 표"""
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
        if not sub(LAND_DIR, "(월) 지역별 지가변동률.csv").exists():
            _RONE = ({}, {})
        else:
            rate = read_rone("(월) 지역별 지가변동률.csv", "rate")
            idx = read_rone("(월) 지역별 지가지수.csv", "index")
            _RONE = ({(r.a, r.b, r.c, r.ym): r.v for r in rate.itertuples()},
                     {(r.a, r.b, r.c, r.ym): r.v for r in idx.itertuples()})
    return _RONE


def region_keys(sido, sgg):
    """(시도, 시군구) → R-ONE에서 찾을 후보 키 (구 → 시 → 시도 순)"""
    a = SIDO_SHORT.get(sido, sido)
    if not isinstance(sgg, str) or not sgg.strip():
        return [(a, a, a)]
    parts = sgg.split()
    if len(parts) >= 2:
        return [(a, parts[0], parts[1]), (a, parts[0], parts[0]), (a, a, a)]
    return [(a, parts[0], parts[0]), (a, a, a)]


def attach_market(frame, lag=2, verbose=False):
    """매각 월의 2개월 전 월간 변동률(lr_1m), 그 시점까지 6개월 누적(lr_6m) — 발표 시차 누수 방지"""
    rate_d, idx_d = rone_tables()
    lr1, lr6, hit = [], [], []
    for sido, sgg, d in zip(frame["sido"], frame["sgg"], frame["opbd_date"]):
        m = d.to_period("M") - lag
        v1 = v6 = None
        lvl = None
        for k, key in enumerate(region_keys(sido, sgg)):
            if v1 is None and (key + (m,)) in rate_d:
                v1 = rate_d[key + (m,)]
                lvl = k
            if v6 is None and (key + (m,)) in idx_d and (key + (m - 6,)) in idx_d:
                v6 = (idx_d[key + (m,)] / idx_d[key + (m - 6,)] - 1) * 100
        lr1.append(v1); lr6.append(v6); hit.append(lvl)
    frame = frame.copy()
    frame["lr_1m"] = pd.to_numeric(pd.Series(lr1, index=frame.index), errors="coerce")
    frame["lr_6m"] = pd.to_numeric(pd.Series(lr6, index=frame.index), errors="coerce")
    if verbose:
        h = pd.Series(hit).map({0: "시군구/구", 1: "시", 2: "시도"}).fillna("없음")
        print("지가변동률 연결 수준:", h.value_counts().to_dict())
    return frame


# ───────────────────────── 모델 피처 / 예측 ─────────────────────────
NUM = ["log_apsl", "landSqms", "bldSqms", "has_building", "log_land_price", "apsl_to_official",
       "bld_age_c", "bld_grnd_flr", "bld_tot_area", "is_daegu_gb", "opbd_qtr", "lr_1m", "lr_6m"]
CAT_BASE = ["usage6", "sido_grp", "bld_strct", "bld_purps"]


def _cat_str(v):
    if v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NA or v is pd.NaT:
        return "없음"
    t = str(v)
    return "없음" if t in ("nan", "<NA>", "None", "NaT") else t


def make_X(frame, cats, categories=None):
    """피처 표. 범주형은 pandas Categorical 대신 '범주 목록에서의 번호'(없는 값은 NaN)로 직접 바꿈
    → pandas 버전(2.x/3.x)에 따라 범주 처리가 달라져 예측이 바뀌는 문제를 막음. 브라우저(engine.js)도 같은 방식.
    categories 를 주면 그 목록(학습 때 model_meta.json 에 저장한 것)을 쓰고, 없으면 값을 정렬해 새로 만듦.
    만든 목록은 X.attrs["categories"] 에 담김."""
    X = frame[NUM].apply(pd.to_numeric, errors="coerce").astype(float)
    used = {}
    for c in cats:
        vals = [_cat_str(v) for v in frame[c].tolist()]
        lst = list(categories[c]) if categories else sorted(set(vals))
        pos = {v: i for i, v in enumerate(lst)}
        X[c] = np.array([pos.get(v, np.nan) for v in vals], dtype=float)
        used[c] = lst
    X.attrs["categories"] = used
    return X


class AuctionModel:
    """저장된 공매 모델(P10/P50/P90) + CQR + 공매→경매 보정 계수를 묶어서 경매 낙찰가율을 예측"""

    def __init__(self, model_dir=MODEL_DIR, calib_dir=CALIB_DIR):
        import lightgbm as lgb
        self.meta = json.load(open(Path(model_dir) / "model_meta.json", encoding="utf-8"))
        self.boosters = {q: lgb.Booster(model_file=str(Path(model_dir) / f"lgb_q{int(q * 100)}.txt")) for q in QS}
        cp = Path(calib_dir) / "calibration.json"
        self.calib = json.load(open(cp, encoding="utf-8")) if cp.exists() else None

    def predict_gongmae(self, frame, prpt_value=None):
        """공매 기준 P10/P50/P90 (정렬 + CQR 적용). prpt_value 를 주면 재산유형을 그 값으로 고정"""
        f = frame.copy()
        if prpt_value is not None:
            f["prpt"] = prpt_value
        X = make_X(f, self.meta["features_cat"], self.meta["categories"])
        # 숫자 배열로 넘김: 범주는 이미 meta 목록의 번호. 예전(pandas Categorical) 방식으로 학습한 팀 모델도 같은 번호를 씀
        P = np.sort(np.column_stack([self.boosters[q].predict(X.to_numpy(dtype=float)) for q in QS]), axis=1)
        d = f["usage6"].map(self.meta.get("cqr_by_usage") or {}).fillna(self.meta["cqr_delta"]).values
        P[:, 0] -= d
        P[:, 2] += d
        return P

    def predict_auction(self, frame):
        """경매 기준 P10/P50/P90 = 공매 예측(용도별 재산유형 고정) × 용도별 보정 계수"""
        if self.calib is None:
            raise FileNotFoundError("calibration.json 이 없어요. step2_calibration.py 를 먼저 실행하세요.")
        P = np.zeros((len(frame), 3))
        for u in frame["usage6"].unique():
            m = (frame["usage6"] == u).values
            P[m] = self.predict_gongmae(frame[m], self.calib["fix_prpt_by_usage"][u])
        fac = frame["usage6"].map(self.calib["factors"]).values[:, None]
        return np.clip(P * fac, 0, None)


# ───────────────────────── 평가 지표 ─────────────────────────
def pinball(y, p, q):
    e = y - p
    return np.mean(np.maximum(q * e, (q - 1) * e))


def evaluate(name, y, P):
    P = np.sort(P, axis=1)
    return {"모델": name, "n": len(y),
            "MAE(P50, %p)": np.mean(np.abs(y - P[:, 1])) * 100,
            "포함률(P10~P90)": np.mean((y >= P[:, 0]) & (y <= P[:, 2])),
            "구간폭(%p)": np.mean(P[:, 2] - P[:, 0]) * 100,
            "pinball": np.mean([pinball(y, P[:, i], q) for i, q in enumerate(QS)])}


def by_group(frame, y, P, col):
    P = np.sort(P, axis=1)
    g = pd.DataFrame({"g": frame[col].values, "y": y, "p10": P[:, 0], "p50": P[:, 1], "p90": P[:, 2]})
    return g.groupby("g").apply(lambda s: pd.Series({
        "n": len(s), "MAE(%p)": np.mean(np.abs(s.y - s.p50)) * 100,
        "포함률": np.mean((s.y >= s.p10) & (s.y <= s.p90))}), include_groups=False).round(3)


def show(df):
    print(df.round(3).to_string(index=False) if isinstance(df, pd.DataFrame) else df)
