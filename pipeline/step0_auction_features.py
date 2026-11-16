"""0단계(선택). 경매 사건에 공시지가·건축물대장 붙이기 (원본: 코드/auction_features.ipynb)

data/경매_매각사례_시험셋.xlsx 의 주소 → V-World PNU → 개별공시지가 + 건축HUB 건축물대장 → 피처
- 이미 조회한 주소는 캐시(data/auction_feat_cache.jsonl)에서 읽으므로, 사건이 새로 늘지 않았으면 인증키 없이 돌아감
- 새 주소가 있으면 인증키 필요: 환경변수 VWORLD_KEY, DATA_GO_KR_KEY (없으면 입력창)
- V-World는 해외 IP를 막으므로 국내 PC에서 실행

결과: output/auction_features.csv (2단계가 data/ 보다 이것을 먼저 씀)
"""
import json
import os
import re
import time
import getpass

import numpy as np
import pandas as pd

from common import DATA_DIR, OUT_ROOT, sub

URL_GEOCODE = "https://api.vworld.kr/req/address"
URL_DATA = "https://api.vworld.kr/req/data"
URL_PRICE = "https://api.vworld.kr/ned/data/getIndvdLandPriceAttr"
URL_BLD = "https://apis.data.go.kr/1613000/BldRgstHubService/getBrTitleInfo"
CACHE_IN = DATA_DIR / "auction_feat_cache.jsonl"
CACHE_OUT = OUT_ROOT / "auction_feat_cache.jsonl"
OUT = OUT_ROOT / "auction_features.csv"
JIBUN = re.compile(r"^(.*?(?:동|리|가|로|길)\s*(?:산\s?)?\d+(?:-\d+)?)")
KEYS = {}


def clean_addr(a):
    """'…동 643-7 외 1필지', '…리(平沙里) 838' → 지번까지만"""
    s = re.sub(r"\(.*?\)", "", str(a)).split(",")[0].strip()
    m = JIBUN.match(s)
    return re.sub(r"\s+", " ", m.group(1)).strip() if m else s


def get_json(url, params):
    import requests
    for i in range(3):
        try:
            return requests.get(url, params=params, timeout=20).json()
        except Exception as e:
            if i == 2:
                print("  ! 호출 실패:", url.split("/")[-1], e)
            time.sleep(1 + i)
    return None


def find_rows(js, field):
    if isinstance(js, dict):
        if field in js: return [js]
        for v in js.values():
            r = find_rows(v, field)
            if r: return r
    elif isinstance(js, list):
        hit = [x for x in js if isinstance(x, dict) and field in x]
        if hit: return hit
        for v in js:
            r = find_rows(v, field)
            if r: return r
    return []


def addr_to_pnu(addr):
    g = get_json(URL_GEOCODE, {"key": KEYS["vw"], "format": "json", "service": "address", "request": "getcoord",
                               "version": "2.0", "crs": "epsg:4326", "type": "parcel", "refine": "true",
                               "simple": "false", "address": addr})
    try:
        resp = g["response"]
        if resp["status"] != "OK":
            return None, f"geocode_{resp['status']}", None
        st = resp.get("refined", {}).get("structure", {})
        pnu = st.get("level4LC")
        matched = resp.get("refined", {}).get("text")
        if not (pnu and len(str(pnu)) == 19):
            x, y = resp["result"]["point"]["x"], resp["result"]["point"]["y"]
            d = get_json(URL_DATA, {"key": KEYS["vw"], "format": "json", "service": "data", "request": "GetFeature",
                                    "data": "LP_PA_CBND_BUBUN", "geomFilter": f"POINT({x} {y})", "crs": "EPSG:4326",
                                    "size": 1, "geometry": "false", "attribute": "true"})
            rows = find_rows(d, "pnu")
            pnu = rows[0]["pnu"] if rows else None
        return (str(pnu) if pnu else None), ("ok" if pnu else "parcel_fail"), matched
    except Exception:
        return None, "geocode_fail", None


def land_prices(pnu):
    rows = find_rows(get_json(URL_PRICE, {"key": KEYS["vw"], "format": "json", "pnu": pnu, "numOfRows": 100, "pageNo": 1}),
                     "pblntfPclnd")
    out = {}
    for r in rows:
        try:
            out[int(r["stdrYear"])] = float(str(r["pblntfPclnd"]).replace(",", ""))
        except Exception:
            pass
    return out


def building(pnu):
    plat = {"1": "0", "2": "1"}.get(pnu[10], "0")
    js = get_json(URL_BLD, {"serviceKey": KEYS["go"], "sigunguCd": pnu[0:5], "bjdongCd": pnu[5:10], "platGbCd": plat,
                            "bun": pnu[11:15], "ji": pnu[15:19], "numOfRows": 100, "pageNo": 1, "_type": "json"})
    rows = find_rows(js, "mainPurpsCdNm")
    if not rows:
        return {"bld_found": False}
    main = [r for r in rows if "주" in str(r.get("mainAtchGbCdNm", "주"))] or rows

    def num(v):
        x = pd.to_numeric(str(v).replace(",", ""), errors="coerce")
        return float(x) if pd.notna(x) else None
    best = max(main, key=lambda r: num(r.get("totArea")) or -1)
    return {"bld_found": True, "bld_n_dong": len(main), "bld_use_apr": str(best.get("useAprDay") or "").strip(),
            "bld_strct": best.get("strctCdNm"), "bld_purps": best.get("mainPurpsCdNm"),
            "bld_grnd_flr": num(best.get("grndFlrCnt")), "bld_ugrnd_flr": num(best.get("ugrndFlrCnt")),
            "bld_tot_area": num(best.get("totArea")), "bld_regstr_gb": best.get("regstrGbCdNm")}


def pick_price(prices, d):
    """1~5월 매각 → 전년도 공시지가 (발표 시점 누수 방지)"""
    if not prices or pd.isna(d): return np.nan, np.nan
    limit = d.year if d.month >= 6 else d.year - 1
    ok = [int(y) for y in prices if int(y) <= limit]
    if not ok: return np.nan, np.nan
    return prices[str(max(ok))], max(ok)


def main():
    print("=" * 70, "\n0단계. 경매 사건 공시지가·건축물대장\n" + "=" * 70)
    cases = pd.read_excel(sub(DATA_DIR, "경매_매각사례_시험셋.xlsx"), sheet_name="② 사례 기록", header=4)
    cases = cases[cases["★ 사건번호"].notna()].copy()
    cases["사건"] = (cases["★ 법원"].astype(str) + " " + cases["★ 사건번호"].astype(str) + "-"
                    + cases["★ 물건번호"].astype(int).astype(str))
    cases["addr"] = cases["★ 소재지 (지번까지)"].map(clean_addr)
    cases["sale_date"] = pd.to_datetime(cases["★ 매각기일"], errors="coerce")

    done = {}
    for path in (CACHE_IN, CACHE_OUT):
        if path.exists():
            for line in open(path, encoding="utf-8"):
                if line.strip():
                    c = json.loads(line); done[c["addr"]] = c
    todo = [a for a in cases["addr"].unique() if a not in done]
    print(f"주소 {cases['addr'].nunique()}개 중 캐시에 있음 {cases['addr'].nunique() - len(todo)}개, 새로 조회 {len(todo)}개")
    if todo:
        import sys
        KEYS["vw"] = os.environ.get("VWORLD_KEY", "").strip()
        KEYS["go"] = os.environ.get("DATA_GO_KR_KEY", "").strip()
        if not (KEYS["vw"] and KEYS["go"]):
            if not sys.stdin.isatty():
                print("⚠️ 새로 조회할 주소가 있는데 인증키가 없어요. VWORLD_KEY, DATA_GO_KR_KEY 환경변수를 넣고 다시 실행하세요.\n"
                      "   (아무것도 저장하지 않았어요. 2단계는 팀이 준 data/auction_features.csv 를 그대로 씁니다)")
                return
            KEYS["vw"] = KEYS["vw"] or getpass.getpass("V-World 인증키: ").strip()
            KEYS["go"] = KEYS["go"] or getpass.getpass("공공데이터포털 인증키(Decoding): ").strip()
        with open(CACHE_OUT, "a", encoding="utf-8") as f:
            for i, a in enumerate(todo, 1):
                pnu, st, matched = addr_to_pnu(a)
                rec = {"addr": a, "pnu": pnu, "status": st, "matched": matched, "prices": {}, "bld": {"bld_found": False}}
                if pnu:
                    rec["prices"] = land_prices(pnu); rec["bld"] = building(pnu)
                done[a] = rec
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n"); f.flush()
                if i % 20 == 0 or i == len(todo):
                    print(f"  {i}/{len(todo)}")
                time.sleep(0.1)

    rows = []
    for _, c in cases.iterrows():
        rec = done.get(c["addr"], {})
        prices = {str(k): v for k, v in (rec.get("prices") or {}).items()}
        lp, yr = pick_price(prices, c["sale_date"])
        land = pd.to_numeric(c["토지면적(㎡)"], errors="coerce")
        apsl = pd.to_numeric(c["★ 감정가(원)"], errors="coerce")
        b = rec.get("bld") or {}
        age = np.nan
        ua = str(b.get("bld_use_apr", "") or "")
        if b.get("bld_found") and re.fullmatch(r"\d{8}", ua):
            if ua[4:6] == "00": ua = ua[:4] + "0701"
            elif ua[6:8] == "00": ua = ua[:6] + "15"
            d0 = pd.to_datetime(ua, format="%Y%m%d", errors="coerce")
            if pd.notna(d0):
                age = (c["sale_date"] - d0).days / 365.25
        is_land = pd.isna(pd.to_numeric(c["건물면적(㎡)"], errors="coerce"))
        rows.append({
            "사건": c["사건"], "addr_clean": c["addr"], "pnu": rec.get("pnu"), "vw_status": rec.get("status"),
            "vw_matched": rec.get("matched"),
            "지번일치": (str(rec.get("matched") or "").split()[-1:] == str(c["addr"]).split()[-1:]),
            "land_price": lp, "price_year": yr,
            "log_land_price": np.log(lp) if pd.notna(lp) and lp > 0 else np.nan,
            "apsl_to_official": apsl / (lp * land) if pd.notna(lp) and pd.notna(land) and lp * land > 0 else np.nan,
            "bld_found": bool(b.get("bld_found")) and not is_land,
            "bld_age_c": np.clip(age, 0, 80) if not is_land else np.nan,
            "bld_grnd_flr": b.get("bld_grnd_flr") if not is_land else np.nan,
            "bld_tot_area": b.get("bld_tot_area") if not is_land else np.nan,
            "bld_strct": b.get("bld_strct") if not is_land else np.nan,
            "bld_purps": b.get("bld_purps") if not is_land else np.nan,
        })
    feat = pd.DataFrame(rows)
    feat.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"공시지가 붙음 {feat['land_price'].notna().sum()}/{len(feat)} | "
          f"건축물대장 붙음 {feat['bld_found'].sum()}/{(~cases['건물면적(㎡)'].isna()).sum()}")
    print("저장:", OUT)


if __name__ == "__main__":
    main()
