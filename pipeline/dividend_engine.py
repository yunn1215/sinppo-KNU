"""배당 엔진 — 기존 pipeline/step3_dividend.py 를 그대로 복사 (계산 로직 수정 없음, 7/7 정답지 일치 확인된 버전)
등기부별 배당 테스트 파일 경로도 기존과 같음: dashboard/fixtures_split.json

(원래 설명) 3단계. 배당 계산기 + 몬테카를로 (원본: 코드/dividend_mc_colab.ipynb)

수집양식 엑셀(⑩ 등기부 권리, ⑩-2 임차인)과 2단계의 경매 낙찰가 예측을 읽어서,
사건마다 신보 회수액 분포(중앙값·범위·0원 확률)와 실제 매각가 기준 배당표를 계산.

결과: output/dividend_out/dividend_results.csv, dividend_tables.csv, dashboard_cases.json
"""
import json
import re
import sys

import numpy as np
import pandas as pd

from common import ROOT, DATA_DIR, CALIB_DIR, DIV_DIR, nfc, sub

# ───────── 0. 설정 (가정값) — 프로젝트 맨 위 settings.json 에서 읽음. 대시보드도 같은 파일을 씀 ─────────
SETTINGS_PATH = ROOT / "settings.json"
_S = {k: v for k, v in json.load(open(SETTINGS_PATH, encoding="utf-8")).items() if not k.startswith("_")}
SINBO_MODE = _S["SINBO_MODE"]                  # "general" 맨 뒤 일반채권자 / "attach" 가압류 날짜 지정 / "mortgage" 근저당 날짜 지정
SINBO_DATE = _S["SINBO_DATE"]                  # attach·mortgage일 때 신보 등기 날짜 (예: "2021-03-01")
SINBO_AMOUNT_RATE = _S["SINBO_AMOUNT_RATE"]    # 신보 채권액 = 감정가 × 비율 (팀 확정 전 임시값)
SINBO_AMOUNT_FIXED = _S["SINBO_AMOUNT_FIXED"]  # 금액으로 정하려면 숫자(원). 정하면 위 비율 무시
USE_ACTUAL_SINBO = _S["USE_ACTUAL_SINBO"]      # 등기부에 신보가 실제로 나오면 그 값을 우선
COST_RATE = _S["COST_RATE"]                    # 집행비용 = 감정가 × 1% (가정값)
MORTGAGE_RATIO = _S["MORTGAGE_RATIO"]          # 근저당 실제 잔액 = 채권최고액 ÷ 1.2 (1.0이면 최고액 그대로)
USE_AUCTION_CLAIM = _S["USE_AUCTION_CLAIM"]    # 경매 신청 채권자의 근저당·가압류에 청구금액 사용
TAX_SCENARIOS = _S["TAX_SCENARIOS"]            # 세금 압류는 있는데 금액이 없을 때, 같은 확률
TAX_FIRST = _S["TAX_FIRST"]                    # True면 그 세금을 당해세로 보고 근저당보다 먼저
INCLUDE_AUCTION_CLAIM = _S["INCLUDE_AUCTION_CLAIM"]   # 경매개시결정 청구금액을 일반채권으로 따로 넣을지 (근저당과 중복 가능)
ABSORB = _S["ABSORB"]                          # 가압류 안분 후 흡수배당
PARTIAL_SUBROGATION = _S["PARTIAL_SUBROGATION"]   # 근저당 일부대위: "pro_rata" 은행과 비율 / "bank_first" 은행 먼저
SPLIT_BOOKS = _S.get("SPLIT_BOOKS", True)      # 토지·건물 등기부별 배당 (민법 368조)
N_SIM = _S["N_SIM"]
SEED = _S["SEED"]

DMAX, GMAX = "9999-99-99", "99999"            # 날짜 미상: 담보권은 맨 뒤, 일반채권은 그보다도 뒤


def sinbo_amount(apsl):
    return float(SINBO_AMOUNT_FIXED) if SINBO_AMOUNT_FIXED else float(apsl) * SINBO_AMOUNT_RATE


# ───────── 1. 배당 엔진 ─────────
TAXY = ("세금", "공과금")      # 일반채권보다 앞서는 조세·공과금


def _subordinate(t, s):
    """풀 안에서 t가 s보다 후순위인가 (s가 흡수할 수 있는 상대). 세금·공과금은 앞선 가압류 몫에서도 가져옴"""
    if s.get("kind") in TAXY and not t.get("secured"):
        return True
    ds = s["date"] or DMAX
    dt = t["date"] or (DMAX if t.get("secured") else GMAX)
    return dt > ds


def absorb(share, W, claims):
    """안분 후 흡수: 담보권·세금(날짜순)이 자기보다 후순위인 권리의 안분액에서 모자란 만큼 가져옴 (후순위 몫 비율대로)"""
    share = share.copy()
    order = sorted([j for j, c in enumerate(claims) if c.get("secured")], key=lambda j: claims[j]["date"] or DMAX)
    for j in order:
        subs = [t for t in range(len(claims)) if t != j and _subordinate(claims[t], claims[j])]
        if not subs:
            continue
        need = np.maximum(W[j] - share[j], 0)
        avail = share[subs].sum(axis=0)
        take = np.minimum(need, avail)
        f = np.where(avail > 0, take / np.maximum(avail, 1e-9), 0.0)
        share[subs] = share[subs] * (1 - f)
        share[j] = share[j] + take
    return share


def split_pay(g, M):
    """등기부별 배당 (engine.js splitOnce 와 같은 계산, 낙찰가 표본 n개를 한꺼번에).
    각 부동산 경매대가 = 그 부동산 몫 − 선순위(공동저당은 분담액) − 같은 순위 몫 (대법원 2001다66291 · 2020다258893),
    공동저당 분담액 = 채권 × 그 부동산 경매대가 / 경매대가 합 (민법 368조 1항), 값이 맞을 때까지 반복.
    반환: 근저당(position)별 받은 돈 [(n,)...], 남은 돈 (n,)"""
    B, P = g["books"], g["positions"]
    N = [M * g["shares"][b] for b in B]
    has = [[b in p["books"] for b in B] for p in P]
    joint = [len(p["books"]) > 1 for p in P]
    shs = [sum(g["shares"][b] for b in p["books"]) for p in P]
    z = np.zeros_like(M)
    d = [[(z if not has[i][bi] else (p["want"] * g["shares"][b] / shs[i] + z if joint[i] else p["want"] + z)) for bi, b in enumerate(B)]
         for i, p in enumerate(P)]
    for _ in range(200):
        a = [[z for _ in B] for _ in P]
        for bi, b in enumerate(B):
            rem = N[bi].copy()
            for grp in g["orders"][b]:
                tot = sum(d[i][bi] for i in grp)
                give = np.minimum(tot, np.maximum(rem, 0))
                part = [np.where(tot > 0, d[i][bi] / np.maximum(tot, 1e-12) * give, 0.0) for i in grp]
                allp = sum(part)
                for k, i in enumerate(grp):
                    a[i][bi] = np.maximum(0, rem - (allp - part[k]))
                rem = rem - give
        diff = 0.0
        for i, p in enumerate(P):
            if not joint[i]:
                continue
            S = sum(a[i][bi] for bi in range(len(B)) if has[i][bi])
            for bi in range(len(B)):
                if not has[i][bi]:
                    continue
                v = np.where(S > 0, np.minimum(a[i][bi], p["want"] * a[i][bi] / np.maximum(S, 1e-12)), 0.0)
                diff = max(diff, float(np.max(np.abs(v - d[i][bi]))) if v.size else 0.0)
                d[i][bi] = v
        if diff < 0.01:
            break
    got = [z.copy() for _ in P]
    left = []
    for bi, b in enumerate(B):
        rem = N[bi].copy()
        for grp in g["orders"][b]:
            tot = sum(d[i][bi] for i in grp)
            give = np.minimum(tot, np.maximum(rem, 0))
            for i in grp:
                got[i] = got[i] + np.where(tot > 0, d[i][bi] / np.maximum(tot, 1e-12) * give, 0.0)
            rem = rem - give
        left.append(rem)
    # 남은 돈이 있는 등기부: 그 등기부의 근저당이 아직 못 받은 만큼 순위대로 (일반채권보다 먼저)
    for bi, b in enumerate(B):
        rem = left[bi]
        for grp in g["orders"][b]:
            sf = [np.maximum(0, P[i]["want"] - got[i]) for i in grp]
            tot = sum(sf)
            give = np.minimum(tot, np.maximum(rem, 0))
            for k, i in enumerate(grp):
                got[i] = got[i] + np.where(tot > 0, sf[k] / np.maximum(tot, 1e-12) * give, 0.0)
            rem = rem - give
        left[bi] = rem
    return got, sum(left)


def waterfall(prices, groups):
    """prices: (n,) 낙찰가 배열. groups: 순위 그룹 리스트 [{"name", "claims":[{"id","name","amount"}], "caps"(선택), "pool"(선택), "split"(선택)}]
    순위마다 채우고 남은 돈만 다음 순위로. 같은 그룹 안은 금액 비율로 안분.
    pool 그룹: 먼저 안분하고, 담보권·세금(secured)은 후순위 권리 몫을 흡수.
    split 그룹: 토지·건물 등기부별 배당 (split_pay).
    반환: {claim_id: (n,) 배당액}, 남은 돈(n,)"""
    prices = np.asarray(prices, dtype=float)
    remain = prices.copy()
    paid = {}
    for g in groups:
        if not g["claims"]:
            continue
        if g.get("split"):
            M = np.maximum(remain, 0)
            got, left = split_pay(g, M)
            byid = {c["id"]: c for c in g["claims"]}
            for i, p in enumerate(g["positions"]):
                mem = [byid[x] for x in p["members"]]
                T = got[i].copy()
                def pay(lst, T):
                    w = sum(c["amount"] for c in lst)
                    give = np.minimum(T, w)
                    for c in lst:
                        paid[c["id"]] = paid.get(c["id"], 0) + (c["amount"] / w * give if w > 0 else 0 * give)
                    return T - give
                if g["rule"] == "bank_first":
                    T = pay([c for c in mem if not c.get("subrog")], T); pay([c for c in mem if c.get("subrog")], T)
                else:
                    pay(mem, T)
            remain = remain - M + left
            continue
        W = np.vstack([np.full_like(prices, float(c["amount"])) for c in g["claims"]])
        for cap_rate, idx in g.get("caps") or []:          # 소액임차인: 배당할 돈(매각가 − 집행비용) × 한도비율까지
            tot = W[idx].sum(axis=0)
            lim = np.maximum(remain, 0) * cap_rate
            W[idx] = W[idx] * np.where(tot > lim, lim / np.maximum(tot, 1e-9), 1.0)
        total = W.sum(axis=0)
        give = np.minimum(total, np.maximum(remain, 0))
        share = np.where(total > 0, W / np.maximum(total, 1e-9), 0) * give
        if g.get("pool"):
            share = absorb(share, W, g["claims"])
        for c, s_ in zip(g["claims"], share):
            paid[c["id"]] = paid.get(c["id"], 0) + s_
        remain = remain - give
    return paid, remain


def sample_ratio(p10, p50, p90, n, rng):
    """P10/P50/P90 사이 선형 보간 + 꼬리 연장 (P10−0.5×(P50−P10) ~ P90+0.5×(P90−P50)), 하한 1%"""
    p10, p50, p90 = np.sort([p10, p50, p90])
    lo = p10 - 0.5 * (p50 - p10); hi = p90 + 0.5 * (p90 - p50)
    r = np.interp(rng.random(n), [0.0, 0.1, 0.5, 0.9, 1.0], [lo, p10, p50, p90, hi])
    return np.maximum(r, 0.01)


def G(name, claims, caps=None, pool=False):
    return {"name": name, "claims": claims, **({"caps": caps} if caps else {}), **({"pool": True} if pool else {})}


def C(cid, amount, date=None, secured=None):
    c = {"id": cid, "name": cid, "amount": amount}
    if secured is not None:
        c.update(date=date, secured=secured)
    return c


# ───────── 2. 손계산 테스트 ─────────
def hand_tests():
    def check(label, got, want, tol=1):
        ok = abs(float(got) - want) <= tol
        print(("  ✅" if ok else "  ❌"), f"{label}: 코드 {float(got):,.0f} / 정답 {want:,.0f}")
        assert ok, label
    groups = [G("집행비용", [C("비용", 10_000_000)]), G("근저당", [C("은행", 540_000_000)]),
              G("일반", [C("신보", 300_000_000), C("캐피탈", 200_000_000)])]
    p, _ = waterfall([600_000_000], groups); check("T1 낙찰가 6억 → 신보", p["신보"][0], 30_000_000)
    p, _ = waterfall([500_000_000], groups); check("T2 낙찰가 5억 → 신보", p["신보"][0], 0)
    p, _ = waterfall([300_000_000], [G("같은날", [C("A", 200_000_000), C("B", 400_000_000)])])
    check("T3 같은 날 근저당 A", p["A"][0], 100_000_000); check("T3 같은 날 근저당 B", p["B"][0], 200_000_000)
    p, _ = waterfall([100_000_000], [G("최우선", [C("임1", 28_000_000), C("임2", 28_000_000)], caps=[(0.5, [0, 1])])])
    check("T4 소액임차 1/2 한도", p["임1"][0], 25_000_000)
    p, _ = waterfall([400_000_000], [G("2020 그룹", [C("신보", 300_000_000), C("은행", 300_000_000)])])
    check("T5 가압류 안분 예외", p["신보"][0], 200_000_000)
    # T6 안분 후 흡수: 가압류A(2019) 2억, 근저당B(2020) 3억, 가압류C(2021) 1억, 배당할 돈 3억
    #    1차 A 1억 · B 1.5억 · C 0.5억 → B가 C의 0.5억 흡수 → A 1억, B 2억, C 0
    pool = G("풀", [C("A", 200_000_000, "2019-01-01", False), C("B", 300_000_000, "2020-01-01", True),
                    C("C", 100_000_000, "2021-01-01", False)], pool=True)
    p, _ = waterfall([300_000_000], [pool])
    check("T6 안분 후 흡수 A", p["A"][0], 100_000_000); check("T6 안분 후 흡수 B", p["B"][0], 200_000_000)
    check("T6 안분 후 흡수 C", p["C"][0], 0)
    # T7 소액 한도: 매각가 1억, 집행비용 1천만 → 한도 (1억−1천만)/2 = 4,500만 → 임차인 2명(2,800만씩)이 2,250만씩
    p, _ = waterfall([100_000_000], [G("집행비용", [C("E", 10_000_000)]),
                                      G("최우선", [C("임1", 28_000_000), C("임2", 28_000_000)], caps=[(0.5, [0, 1])])])
    check("T7 소액 한도 = 배당할 돈의 1/2", p["임1"][0], 22_500_000)
    # T8 (발표 자료 예시) 낙찰가 3.5억, 집행비용 1,000만, 신보 근저당 3억, 세금 1억 → 세금이 늦으면 3억 / 빠르면 2.4억
    late = [G("비용", [C("E", 10_000_000)]), G("신보근저당", [C("신보", 300_000_000)]), G("세금", [C("T", 100_000_000)])]
    early = [late[0], late[2], late[1]]
    check("T8 세금이 늦을 때 신보", waterfall([350_000_000], late)[0]["신보"][0], 300_000_000)
    check("T8 세금이 빠를 때 신보", waterfall([350_000_000], early)[0]["신보"][0], 240_000_000)
    split_tests(check)
    print("  손계산 테스트 모두 통과")


def split_tests(check):
    """T10 토지·건물 등기부별 배당 (민법 368조 · 대법원 2001다66291): dashboard/fixtures_split.json — JS 테스트와 같은 사례"""
    fx = json.load(open(ROOT / "dashboard" / "fixtures_split.json", encoding="utf-8"))
    global PARTIAL_SUBROGATION
    keep = PARTIAL_SUBROGATION
    for f in fx.values():
        reg = pd.DataFrame([{"key": "FX", "등기목적": r["type"], "권리자 유형": r["who"], "권리자": r["name"], "접수일": pd.Timestamp(r["date"]),
                             "금액": r["amount"] if r["amount"] is not None else np.nan, "메모": r["memo"],
                             "갑구/을구": "을구" if r.get("rank") else "갑구",
                             "순위번호": int(r["rank"][1:]) if r.get("rank") else np.nan, BOOK_COL: r["book"], "유효": True} for r in f["rights"]])
        D = {"reg": reg, "ten": pd.DataFrame(columns=["key"]), "small_tbl": pd.DataFrame(), "region_tbl": pd.DataFrame(columns=["시·군·구", "기준 지역"])}
        row = pd.Series({"key": "FX", "감정가(원)": f["apsl"], "최선순위 설정일": None, "소재지": "", LAND_COL: f["land"]})
        for rule, want in f["want"].items():
            PARTIAL_SUBROGATION = rule
            mg, info = build_case(row, D)
            paid, _ = waterfall([f["price"]], mg(0))
            got = sum(float(paid.get(i, [0])[0]) for i in info["sinbo_ids"])
            check(f"T10 {f['name']} {f['price'] / 1e8:g}억 {'비율' if rule == 'pro_rata' else '은행 먼저'}", round(got / 1e6), round(want / 1e6))
    PARTIAL_SUBROGATION = keep


# ───────── 2-1. 등기부 줄 정리 · 청구금액 · 일부이전 (engine.js 와 같은 규칙) ─────────
CLAIM_RE = re.compile(r"청구금액[:：]?\s*([\d,]{4,})")
ALIAS_RES = [re.compile(r"양도(?:인|전)([^,()—·]+)"), re.compile(r"([^,()—·]+?)근저당권?양수"),
             re.compile(r"변경전(?:상호)?([^,()—·]+)")]
SUBROG_RE = re.compile(r"일부이전|대위변제|대위분")
PARTIAL_RE = re.compile(r"일부이전")
RANK_RE = re.compile(r"을\s*구?\s*(\d+)")
MORT_TYPES = ("근저당권설정", "전세권설정", "임차권등기")
BOOKS = ("토지", "건물")
RMAX = "zzzzzz"
INF = float("inf")


def isnum(v):
    return v is not None and not (isinstance(v, float) and np.isnan(v))


def norm_name(s):
    s = re.sub(r"\(주\)|㈜|주식회사", "", nfc(s or ""))
    return re.sub(r"\s", "", re.sub(r"\(.*?\)", "", s))


def claim_of(r):
    """경매개시결정 줄의 청구금액: 금액 칸 → 없으면 메모의 '청구금액 N'"""
    if isnum(r.get("amount")):
        return r["amount"]
    m = CLAIM_RE.search(re.sub(r"(?<=\d)\s+(?=[\d,])", "", nfc(r.get("memo") or "")))
    return float(m.group(1).replace(",", "")) if m else None


def memo_max(memo):
    """메모에 적힌 원래 채권최고액 (팀이 은행 몫에서 신보 이전분을 빼고 입력한 경우)"""
    t = nfc(memo or "")
    m = re.search(r"채권최고액\s*([\d,]{5,})", t)
    if m:
        return float(m.group(1).replace(",", ""))
    m = re.search(r"원래\s*([\d.]+)\s*억", t)
    return float(m.group(1)) * 1e8 if m else None


def rank_no_of(r):
    k = str(r.get("rank") or "")
    if not k.startswith("을"):
        return None
    m = re.search(r"\d+", k)
    return m.group(0) if m else None


def book_pos(r):
    """등기부별 위치 {토지: {date, rank}, 건물: {...}} — 모르면 None"""
    b = re.sub(r"\s", "", nfc(r.get("book") or ""))
    here = {"date": r.get("date"), "rank": rank_no_of(r)}
    if b in BOOKS:
        return {b: here}
    if re.search(r"토지.*건물|건물.*토지|공통|공동", b):
        return {"토지": dict(here), "건물": dict(here)}
    return None


def merge_books(rows):
    """토지·건물 등기부에 한 줄씩 적힌 같은 권리(종류·권리자·금액)를 한 줄로. 등기부별 접수일·순위번호는 _bp"""
    out, idx = [], {}
    for r in rows:
        bp = book_pos(r)
        one = next(iter(bp)) if bp and len(bp) == 1 else None
        if not one:
            out.append({**r, "_bp": bp}); continue
        amt = r["amount"] if isnum(r.get("amount")) else claim_of(r)
        key = (r["type"], "신보" if r["who"] == "신보" else norm_name(r["name"]), amt)
        lst = idx.setdefault(key, [])
        mate = next((o for o in lst if one not in o["_bp"]), None)
        if mate:
            mate["_bp"][one] = bp[one]
            if (r["date"] or DMAX) < (mate["date"] or DMAX):
                mate["date"] = r["date"]
            continue
        o = {**r, "_bp": dict(bp)}; out.append(o); lst.append(o)
    return out


def lead_rank(r):
    if not r.get("_bp"):
        return rank_no_of(r)
    e = sorted(r["_bp"].values(), key=lambda p: p["date"] or DMAX)
    return e[0]["rank"] if e else None


def rank_num(x):
    return INF if x is None else float(x)


def row_key(rows, j):
    return (rows[j]["date"] or DMAX, rank_num(lead_rank(rows[j])), j)


def link_subrog(rows):
    """신보 근저당 중 '일부이전·대위변제분' → 원래 근저당 줄 번호 {j: o}"""
    links = {}
    base = [o for o, r in enumerate(rows) if r["type"] == "근저당권설정" and r["who"] != "신보" and isnum(r["amount"])]
    for j, r in enumerate(rows):
        if r["type"] != "근저당권설정" or r["who"] != "신보":
            continue
        memo = nfc(r.get("memo") or "") + " " + nfc(r.get("kind") or "")
        if not SUBROG_RE.search(memo):
            continue
        flat = re.sub(r"\s", "", memo)
        named = [o for o in base if norm_name(rows[o]["name"]) and norm_name(rows[o]["name"]) in flat]
        same = [o for o in base if r["date"] and rows[o]["date"] == r["date"]]
        cand = same or (named if PARTIAL_RE.search(memo) else [])
        if not cand:
            continue
        nums = set(RANK_RE.findall(flat)); my = lead_rank(r)
        cand = sorted(cand, key=lambda o: (0 if lead_rank(rows[o]) is not None and lead_rank(rows[o]) == my else 1,
                                           0 if str(lead_rank(rows[o])) in nums else 1, 0 if o in named else 1, o))
        links[j] = cand[0]
    return links


def cap_of(r):
    return memo_max(r.get("memo")) or r["amount"]


def resolve_auction_claims(rows, ratio, links):
    """경매 청구금액 → 신청 채권자 근저당(순위대로 각 최고액 − 신보 몫까지) · 강제경매면 그 채권자의 가압류.
    양도 전·양도인·X 근저당 양수·변경 전 상호 X 도 같은 채권자. 이름으로 못 찾고 근저당권자가 한 곳뿐이면 연결(확인 필요)"""
    over, notes, filled = {}, [], {}
    subs_of = lambda o: sum(rows[j]["amount"] for j, oo in links.items() if oo == o)
    morts = [j for j, r in enumerate(rows) if r["type"] == "근저당권설정" and r["who"] != "신보" and isnum(r["amount"])]
    for a in rows:
        if a["type"] != "경매개시결정":
            continue
        N = claim_of(a)
        if N is None:
            continue
        memo = nfc(a.get("memo") or "")
        forced = "강제" in memo or "강제" in nfc(a.get("kind") or "")
        sinbo = a["who"] == "신보"
        names = {norm_name(a["name"])} | {norm_name(m.group(1)) for rx in ALIAS_RES for m in rx.finditer(re.sub(r"\s", "", memo))}
        names.discard("")
        same = (lambda r: r["who"] == "신보") if sinbo else (lambda r: r["who"] != "신보" and norm_name(r["name"]) in names)
        label = "신보" if sinbo else (a["name"] or "신청 채권자")
        if forced:
            hits = [j for j, r in enumerate(rows) if r["type"] == "가압류" and isnum(r["amount"]) and same(r)]
            if len(hits) == 1:
                over[hits[0]] = N
                notes.append(f"{label} 가압류 → 강제경매 청구금액 {N:,.0f}원 사용")
            elif len(hits) > 1:
                notes.append(f"{label} 가압류가 여러 건이라 강제경매 청구금액은 쓰지 않음")
            continue
        if sinbo:
            continue
        hits = [j for j in morts if same(rows[j])]
        if not hits:
            holders = {norm_name(rows[j]["name"]) for j in morts}
            if len(holders) == 1 and morts:
                hits = list(morts)
                notes.append(f"[확인 필요] {label} 청구금액 {N:,.0f}원을 {next(iter(holders))} 근저당에 연결 (양도인 문구 없음 · 근저당권자가 한 곳뿐)")
        if not hits:
            continue
        hits.sort(key=lambda j: row_key(rows, j))
        left = N
        for j in hits:
            room = max(0.0, cap_of(rows[j]) - subs_of(j) - filled.get(j, 0.0))
            give = min(left, room)
            filled[j] = filled.get(j, 0.0) + give; left -= give; over[j] = filled[j]
        notes.append(f"{label} 근저당 → {over[hits[0]]:,.0f}원 (경매 청구금액 {N:,.0f}원, 최고액 − 신보 몫 이내)" if len(hits) == 1
                     else f"{label} 경매 청구금액 {N:,.0f}원을 근저당 {len(hits)}건에 순위대로 나눔 (각 최고액 − 신보 몫까지)")
        if left > 0.5:
            notes.append(f"{label} 청구금액 중 {left:,.0f}원은 최고액을 넘어 담보 밖")
    return over, notes


# ───────── 3. 데이터 읽기 ─────────
def todate(v):
    if pd.isna(v) or str(v).strip() == "":
        return pd.NaT
    return pd.to_datetime(str(v).strip().replace(".", "-"), errors="coerce")


def tonum(v):
    if pd.isna(v):
        return np.nan
    s = re.sub(r"[^\d.]", "", str(v))
    return float(s) if s else np.nan


def key_of(caseno, item):
    return f"{str(caseno).strip()}-{int(float(item)) if pd.notna(item) else 1}"


def load_inputs():
    fs = [p for p in DATA_DIR.iterdir() if "수집양식" in nfc(p.name) and p.suffix.lower() == ".xlsx" and not p.name.startswith("~$")]
    if not fs:
        raise FileNotFoundError("data 폴더에 '수집양식' 엑셀이 없어요.")
    xlsx = max(fs, key=lambda p: p.stat().st_mtime)
    rd = lambda sheet, header: pd.read_excel(xlsx, sheet_name=sheet, header=header)
    target = rd("⓪ 대상 사건", 4)
    target = target[target["사건번호"].notna() & pd.to_numeric(target["순서"], errors="coerce").notna()].copy()
    reg = rd("⑩ 등기부 권리", 4).dropna(subset=["사건번호"]).copy()
    ten = rd("⑩-2 임차인", 4).dropna(subset=["사건번호"]).copy()
    small_tbl = rd("⑨ 소액임차 기준표", 4).dropna(subset=["시작일"])
    small_tbl = small_tbl[small_tbl["구분"].isin(["주택", "상가"])]
    region_tbl = rd("⑨ 지역 구분", 3).dropna(subset=["시·군·구"])

    pred_path = CALIB_DIR / "auction_predictions.csv"
    if not pred_path.exists():
        raise FileNotFoundError(f"{pred_path} 가 없어요. step2_calibration.py 를 먼저 실행하세요.")
    pred = pd.read_csv(pred_path)
    pred["key"] = pred["사건"].str.split(" ", n=1).str[1]

    for d in (reg, ten):
        d["key"] = [key_of(a, b) for a, b in zip(d["사건번호"], d["물건번호"])]
    target["key"] = [key_of(a, b) for a, b in zip(target["사건번호"], target["물건번호"])]
    reg["접수일"] = reg["접수일"].map(todate)
    reg["금액"] = reg["금액(원)"].map(tonum)
    reg["유효"] = reg["경매 당시 유효"].astype(str).str.strip().str.upper().ne("N")
    for c in ["전입일 (상가는 사업자등록일)", "확정일자", "배당요구일"]:
        ten[c] = ten[c].map(todate)
    ten["보증금"] = ten["보증금(원)"].map(tonum).fillna(0)
    ten["월세"] = ten["월세(원)"].map(tonum).fillna(0)
    print(f"수집양식: {nfc(xlsx.name)} | 대상 {len(target)}건 중 등기부 입력 {target.key.isin(set(reg.key)).sum()}건, 임차인 행 {len(ten)}개")
    return dict(xlsx=xlsx, target=target, reg=reg, ten=ten, small_tbl=small_tbl, region_tbl=region_tbl, pred=pred)


# ───────── 4. 사건별 권리 → 순위 그룹 ─────────
def region_of(addr, region_tbl):
    m = re.match(r"(대구광역시|경상북도)\s+(\S+?(시|군|구))", nfc(addr))
    if not m:
        return "그 밖의 지역"
    name = ("대구 " if m.group(1).startswith("대구") else "경북 ") + m.group(2)
    hit = region_tbl[region_tbl["시·군·구"].map(nfc) == name]
    return hit["기준 지역"].iloc[0] if len(hit) else "그 밖의 지역"


def small_rule(small_tbl, kind, region, base_date):
    t = small_tbl[(small_tbl["구분"] == kind) & (small_tbl["지역"] == region)]
    d = base_date if pd.notna(base_date) else pd.Timestamp.today()
    ts = lambda v: pd.Timestamp.max if str(v).startswith("9999") else pd.Timestamp(v)
    t = t[t["시작일"].map(ts).le(d) & t["종료일"].map(ts).ge(d)]
    return None if t.empty else t.iloc[0]


def iso(d):
    return None if d is None or pd.isna(d) else str(pd.Timestamp(d).date())


BOOK_COL = "등기부(토지/건물)"       # ⑩ 등기부 권리 (선택 열): 토지 / 건물 / 토지·건물. 비우면 합쳐서 계산
LAND_COL = "토지 감정가(원)"         # ⓪ 대상 사건 (선택 열): 등기부별 배당에 쓰는 토지 몫


def rights_of(R):
    """⑩ 행들 → engine.js 와 같은 권리 목록 (날짜는 'YYYY-MM-DD' 문자열)"""
    out = []
    for _, r in R.iterrows():
        gu = str(r["갑구/을구"]) if pd.notna(r["갑구/을구"]) else ""
        out.append({"type": r["등기목적"], "who": r["권리자 유형"], "name": r["권리자"] if pd.notna(r["권리자"]) else "",
                    "date": iso(r["접수일"]), "amount": None if pd.isna(r["금액"]) else float(r["금액"]),
                    "memo": nfc(r["메모"]) if pd.notna(r["메모"]) else "",
                    "rank": f"을{int(r['순위번호'])}" if pd.notna(r["순위번호"]) and gu.startswith("을") else None,
                    "book": nfc(r[BOOK_COL]).strip() if BOOK_COL in R.columns and pd.notna(r[BOOK_COL]) else ""})
    return out


def build_case(row, D):
    """⓪ 한 행 → (make_groups(tax_amount, include_sinbo, tax_first) 함수, info). 계산 규칙은 engine.js buildCase 와 같음"""
    reg, ten = D["reg"], D["ten"]
    k = row["key"]; apsl = float(row["감정가(원)"])
    base_date = todate(row["최선순위 설정일"])
    region = region_of(row["소재지"], D["region_tbl"])
    Rdf = reg[(reg.key == k) & reg.유효].copy()
    T = ten[ten.key == k].copy()
    land = tonum(row.get(LAND_COL)) if LAND_COL in row.index else np.nan
    shares = {"토지": land, "건물": apsl - land} if isnum(land) and 0 < land < apsl else None
    R = merge_books(rights_of(Rdf))
    notes = []
    label = lambda r: f"{r['who'] or ''} {r['type']}".strip()
    auction_date = min([r["date"] for r in R if r["type"] == "경매개시결정" and r["date"]], default=None)
    secured, general, prio, tax_dates, hidden = [], [], [], [], []
    tax_no_amt = 0
    links = link_subrog(R)
    over, cnotes = resolve_auction_claims(R, MORTGAGE_RATIO, links) if USE_AUCTION_CLAIM else ({}, [])
    notes += cnotes
    subs_of = lambda o: sum(R[j]["amount"] for j, oo in links.items() if oo == o)
    jclaim = {}
    for i, r in enumerate(R):
        cid, typ, who = f"r{i}", r["type"], r["who"]
        amt = None if typ == "경매개시결정" else (r["amount"] if isnum(r["amount"]) else None)
        base = {"date": r["date"], "rank": lead_rank(r), "bp": r.get("_bp")}
        if who == "신보" and not USE_ACTUAL_SINBO:
            continue
        if who == "신보" and typ not in ("근저당권설정", "전세권설정", "가압류", "경매개시결정"):
            notes.append(f"신보 '{typ}' 행은 배당에 안 들어감")
        if who == "근로복지공단":
            if amt is not None: prio.append({"id": cid, "name": "근로복지공단(임금)", "amount": amt, "kind": "임금"})
            else: notes.append("근로복지공단 금액 없음 → 0 처리")
            continue
        if typ in MORT_TYPES:
            if amt is None: notes.append(f"{label(r)} 금액 없음 → 제외"); continue
            if i in over:
                a, basis = over[i], "청구금액"
            elif typ == "근저당권설정" and who != "신보":
                a, basis = max(0.0, cap_of(r) / MORTGAGE_RATIO - subs_of(i)), ("추정" if MORTGAGE_RATIO != 1 else "최고액")
            else:
                a, basis = amt, "등기"
            c = {"id": cid, "name": label(r), "amount": a, "max": amt, "kind": "담보", "basis": basis, "who": who, **base}
            if i in links:                       # 일부이전: 새 근저당이 아니라 원래 근저당 순위 안의 신보 몫
                o = R[links[i]]; c["subrog"] = True; c["parentId"] = f"r{links[i]}"
                c["rank"] = lead_rank(o); c["bp"] = o.get("_bp") or c["bp"]
                if o["date"] != r["date"]:
                    c["date"] = o["date"]
                    notes.append(f"신보 일부이전 → 원래 근저당({o['name']} {o['date']}) 순위로")
            jclaim[i] = c
            secured.append(c)
        elif typ == "압류":
            if who in ("세무서", "지자체"):
                tax_dates.append(r["date"])
                if amt is not None: secured.append({"id": cid, "name": label(r), "amount": amt, "kind": "세금", **base})
                else: tax_no_amt += 1; hidden.append(r["name"] or who)
            elif amt is not None:
                secured.append({"id": cid, "name": label(r), "amount": amt, "kind": "공과금", **base})
            else:
                notes.append(f"{label(r)} 금액 없음 → 0 처리 (납부기한이 근저당보다 빠르면 숨은 선순위일 수 있음)"); hidden.append(r["name"] or who)
        elif typ == "가압류":
            if amt is None: notes.append(f"{label(r)} 금액 없음 → 제외"); continue
            general.append({"id": cid, "name": label(r), "amount": over.get(i, amt), "kind": "가압류", "who": who, **base,
                            **({"basis": "청구금액"} if i in over else {})})
        elif typ == "경매개시결정" and INCLUDE_AUCTION_CLAIM and claim_of(r) is not None:
            general.append({"id": cid, "name": label(r), "amount": claim_of(r), "kind": "가압류", "who": who, **base})

    # 같은 근저당 안의 은행 몫 + 신보 몫 ≤ 채권최고액 (메모에 원래 최고액이 있으면 그 값)
    for o in sorted(set(links.values())):
        if o not in jclaim:
            continue
        mem = [jclaim[j] for j, oo in links.items() if oo == o and j in jclaim]
        M = cap_of(R[o]); bank = jclaim[o]
        tot = bank["amount"] + sum(c["amount"] for c in mem)
        if tot > M + 0.5:
            if PARTIAL_SUBROGATION == "bank_first":
                cut = tot - M
                for c in mem:
                    d_ = min(cut, c["amount"]); c["amount"] -= d_; cut -= d_
                bank["amount"] -= cut
            else:
                for c in [bank] + mem:
                    c["amount"] *= M / tot
            notes.append(f"{R[o]['name']} 근저당: 은행 몫 + 신보 몫 {tot:,.0f}원이 채권최고액 {M:,.0f}원을 넘어 최고액에 맞춤")

    for i, t in T.iterrows():
        if str(t["배당요구 여부"]).strip().upper() != "Y":
            notes.append(f"임차인{t['임차인 번호']} 배당요구 없음 → 제외"); continue
        kind = "상가" if str(t["구분"]).strip() == "상가" else "주택"
        dep, rent = t["보증금"], t["월세"]
        judge = dep + rent * 100 if kind == "상가" else dep
        moved = t["전입일 (상가는 사업자등록일)"]
        rule = small_rule(D["small_tbl"], kind, region, base_date)
        first = 0.0
        ok_time = auction_date is None or (pd.notna(moved) and iso(moved) < auction_date)
        if rule is not None and judge <= rule["소액 보증금 상한"] and ok_time:
            first = min(dep, float(rule["최우선변제액"]))
            prio.append({"id": f"t{i}a", "name": f"임차인{t['임차인 번호']} 최우선", "amount": first,
                         "cap": float(rule["한도 비율"]), "kind": "소액임차"})
        rest = dep - first
        conf = t["확정일자"]
        if rest > 0 and pd.notna(conf) and pd.notna(moved):
            secured.append({"id": f"t{i}b", "name": f"임차인{t['임차인 번호']} 확정일자", "amount": rest,
                            "date": iso(max(pd.Timestamp(moved) + pd.Timedelta(days=1), pd.Timestamp(conf))), "rank": None, "bp": None,
                            "kind": "임차인"})

    # 신보 채권: 근저당(일부이전·자기 근저당) + 가압류 중 담보를 넘는 부분만 일반채권 한 줄 (같은 구상채권으로 보고 중복 제거)
    B = sinbo_amount(apsl)
    sinbo_claim = None
    sb_sec = [c for c in secured if c.get("who") == "신보"]
    sb_att = [c for c in general if c.get("who") == "신보"]
    actual = bool(USE_ACTUAL_SINBO and (sb_sec or sb_att))
    if actual:
        own = [c for c in sb_sec if not c.get("subrog")]
        sb_auction = list(dict.fromkeys(v for v in (claim_of(r) for r in R if r["type"] == "경매개시결정" and r["who"] == "신보"
                                                    and "강제" not in nfc(r.get("memo") or "") + nfc(r.get("kind") or "")) if v is not None))
        if own and sb_auction:                   # 신보 자기 근저당: 신보 임의경매 청구금액을 순위대로 최고액까지
            left = sum(sb_auction) - sum(c["amount"] for c in sb_sec if c.get("subrog"))
            for c in own:
                v = max(0.0, min(left, c["max"])); left -= v; c["amount"] = v; c["basis"] = "청구금액"
        sec_amt = sum(c["amount"] for c in sb_sec)
        sub = sum(c["amount"] for c in sb_sec if c.get("subrog"))
        if sb_auction and not own and sub > 0 and abs(sum(sb_auction) - sub) > 1:
            notes.append(f"[확인] 신보 임의경매 청구 합 {sum(sb_auction):,.0f}원 ≠ 일부이전 합 {sub:,.0f}원")
        if sb_sec and sb_att:
            att_amt = sum(c["amount"] for c in sb_att); rem = max(0.0, att_amt - sec_amt)
            general = [c for c in general if c not in sb_att]
            if rem > 0.5:
                d0 = min([c["date"] for c in sb_att if c["date"]], default=None)
                general.append({"id": "SBGEN", "name": "신용보증기금 가압류 (담보 초과분)", "amount": rem, "kind": "가압류", "who": "신보",
                                "date": d0, "rank": None, "bp": None, "basis": "담보 초과분"})
            notes.append(f"신보 가압류 {att_amt:,.0f}원 중 근저당으로 받는 {sec_amt:,.0f}원을 빼고 나머지 {rem:,.0f}원만 일반채권 (같은 구상채권으로 가정)")
        ids = [c for c in secured if c.get("who") == "신보"] + [c for c in general if c.get("who") == "신보"]
        sinbo_ids = [c["id"] for c in ids]
        B = float(sum(c["amount"] for c in ids))
        notes.append(f"신보 채권액 = {B:,.0f}원 (근저당 {sec_amt:,.0f}원{' + 가압류' if any(c['kind'] == '가압류' for c in ids) else ''})")
        notes.append("등기부의 실제 신보 채권 사용")
    else:
        kind = {"mortgage": "담보", "attach": "가압류"}.get(SINBO_MODE, "가압류")
        sinbo_claim = {"id": "SINBO", "name": f"신보(가정·{ {'mortgage': '근저당', 'attach': '가압류'}.get(SINBO_MODE, '일반채권') })",
                       "amount": B, "date": iso(todate(SINBO_DATE)) if SINBO_MODE != "general" else None, "rank": None, "bp": None, "kind": kind}
        sinbo_ids = ["SINBO"]

    tax_needed = tax_no_amt > 0
    tax_date = min([d for d in tax_dates if d], default=None)
    if tax_needed or any(c["kind"] in TAXY for c in secured):
        notes.append("세금·공과금 순서: 법정기일을 몰라 압류일로 대신 정함 (일반채권보다는 앞)")
    cost = apsl * COST_RATE

    # 토지·건물 등기부별 배당이 필요한지
    mort_r = [r for r in R if r["type"] in MORT_TYPES and isnum(r["amount"])]
    books = [b for b in BOOKS if any(r.get("_bp") and b in r["_bp"] for r in R)]
    sh = {"토지": shares["토지"] / apsl, "건물": shares["건물"] / apsl} if shares else None
    linked = {id(R[j]) for j in links}
    why = []
    if len(books) == 2 and all(r.get("_bp") for r in mort_r):
        lone = [r for r in mort_r if id(r) not in linked and len(r["_bp"]) < 2]
        if lone: why.append(f"한쪽 등기부에만 있는 근저당 {len(lone)}건")
        joint = [r for r in mort_r if id(r) not in linked and len(r["_bp"]) == 2]
        ordb = lambda b: [k_ for _, _, k_ in sorted((r["_bp"][b]["date"] or DMAX, rank_num(r["_bp"][b]["rank"]), k_) for k_, r in enumerate(joint))]
        if ordb("토지") != ordb("건물"): why.append("토지·건물 등기부의 근저당 순서가 다름")
    split = bool(why) and sh is not None and SPLIT_BOOKS
    if why and not sh:
        notes.append(f"{' · '.join(why)} → 등기부별로 나눠 배당해야 하지만 토지 감정가가 없어 합쳐서 계산했어요")
    if split:
        notes.append(f"{' · '.join(why)} → 매각대금을 토지·건물 감정가 비율로 나누고, 공동저당은 각 부동산 경매대가 비율로 분담해 등기부별로 배당 (민법 368조 · 대법원 2001다66291)")
    if not books and mort_r:
        n1 = sum(1 for r in R if re.search(r"토지만|건물만", nfc(r.get("memo") or "")))
        if n1:
            notes.append(f"등기부(토지/건물) 구분이 없어 합쳐서 계산했어요 — 메모상 한쪽 등기부에만 있는 권리 {n1}건, 순위가 다를 수 있음")

    lag = PARTIAL_SUBROGATION == "bank_first"
    key = lambda c: (c["date"] or DMAX) + "|" + (str(c["rank"]).rjust(6, "0") if c.get("rank") is not None else RMAX) + ("~" if lag and c.get("subrog") else "")

    def split_group(sec):
        pos, by = [], {}
        for c in sec:
            if c.get("subrog") and c.get("parentId"): continue
            p = {"members": [c], "bp": c.get("bp")}; pos.append(p); by[c["id"]] = p
        for c in sec:
            if not (c.get("subrog") and c.get("parentId")): continue
            p = by.get(c["parentId"])
            if p: p["members"].append(c)
            else:
                q = {"members": [c], "bp": c.get("bp")}; pos.append(q); by[c["id"]] = q
        for p in pos:
            lead = p["members"][0]
            p["bp"] = p["bp"] or {b: {"date": lead["date"], "rank": None} for b in books}
            p["books"] = [b for b in books if b in p["bp"]]
            p["want"] = sum(c["amount"] for c in p["members"])
        orders = {}
        for b in books:
            ix = [i for i, p in enumerate(pos) if b in p["bp"]]
            kb = lambda i: (pos[i]["bp"][b]["date"] or DMAX) + "|" + (str(pos[i]["bp"][b]["rank"]).rjust(6, "0") if pos[i]["bp"][b]["rank"] is not None else RMAX)
            ix.sort(key=lambda i: (kb(i), i))
            grps, last = [], None
            for i in ix:
                if kb(i) != last: grps.append([]); last = kb(i)
                grps[-1].append(i)
            orders[b] = grps
        claims = [{"id": c["id"], "name": c["name"], "amount": c["amount"], "kind": c["kind"], **({"subrog": True} if c.get("subrog") else {})}
                  for p in pos for c in p["members"]]
        return {"name": "토지·건물 등기부별 배당 (민법 368조)", "split": True, "books": books, "shares": sh, "rule": "bank_first" if lag else "pro_rata",
                "positions": [{"members": [c["id"] for c in p["members"]], "books": p["books"], "want": p["want"]} for p in pos],
                "orders": orders, "claims": claims}

    def make_groups(tax_amount=0.0, include_sinbo=True, tax_first=TAX_FIRST):
        sec = list(secured)
        gen = list(general)
        if include_sinbo and sinbo_claim is not None:
            (sec if sinbo_claim["kind"] == "담보" else gen).append(sinbo_claim)
        g = [G("집행비용", [C("COST", cost)])]
        if prio:
            caps = {}
            for j, c in enumerate(prio):
                if "cap" in c: caps.setdefault(c["cap"], []).append(j)
            g.append(G("최우선", [{k2: v for k2, v in c.items() if k2 != "cap"} for c in prio], caps=list(caps.items())))
        if tax_amount > 0:
            tc = {"id": "TAX", "name": "체납세금(추정)", "amount": tax_amount, "date": tax_date, "rank": None, "bp": None, "kind": "세금"}
            if tax_first: g.append(G("체납세금 (당해세 가정)", [tc]))
            else: sec.append(tc)
        mort = sorted(c["date"] for c in sec if c["kind"] == "담보" and c["date"])
        early = [c for c in gen if c["date"] and any(c["date"] < m for m in mort)]
        pool = []
        if early and ABSORB:
            # 안분 후 흡수: 가장 이른 '근저당보다 먼저인 가압류' 날짜부터 뒤의 권리는 모두 한 풀에서 안분 → 담보권·세금이 후순위 몫 흡수
            d0 = min(c["date"] for c in early)
            sec = sorted(sec, key=key)
            pool = [{**c, "date": key(c) if c["date"] else None, "secured": True} for c in sec if key(c) >= d0]
            pool += [{**c, "secured": False, **({"kind": "가압류(안분)"} if c in early else {})} for c in gen]
            pool = sorted(pool, key=lambda c: c["date"] or (DMAX if c["secured"] else GMAX))
            sec = [c for c in sec if key(c) < d0]
            gen = []
        elif early:
            still = []
            for c in gen:
                later = [m for m in mort if c["date"] and c["date"] < m]
                if later: sec.append({**c, "date": later[0], "rank": None, "kind": "가압류(안분)"})
                else: still.append(c)
            gen = still
        sec = sorted(sec, key=key)
        if split and sec:
            g.append(split_group(sec))
        else:
            cur = None
            for c in sec:
                kk = key(c)
                if cur is None or cur[0] != kk:
                    rk = f" · 을{c['rank']}" if c.get("rank") is not None else ""
                    lab = "날짜 미상" if kk.startswith(DMAX) else f"날짜 {kk[:10]}{rk}" + (" (일부대위 · 원래 권리자 다음)" if kk.endswith("~") else "")
                    cur = (kk, G(lab, [])); g.append(cur[1])
                cur[1]["claims"].append({"id": c["id"], "name": c["name"], "amount": c["amount"], "kind": c["kind"]})
        if pool:
            g.append(G(f"안분·흡수 (가압류 {d0}부터)", [{"id": c["id"], "name": c["name"], "amount": c["amount"], "kind": c["kind"],
                                                       "date": c["date"], "secured": c["secured"]} for c in pool], pool=True))
        if gen:
            g.append(G("일반채권", [{"id": c["id"], "name": c["name"], "amount": c["amount"], "kind": c["kind"]} for c in gen]))
        return g

    info = dict(sinbo_ids=sinbo_ids, key=k, 감정가=apsl, 지역=region, 최선순위=base_date, 신보채권액=B,
                세금시나리오=tax_needed, 세금기준일=tax_date, 권리수=len(Rdf), 임차인수=len(T),
                메모=" / ".join(dict.fromkeys(notes)), secured=secured, general=general, prio=prio,
                actual_sinbo=actual, split=split)
    return make_groups, info


# ───────── 5. 몬테카를로 ─────────
def simulate(make_groups, info, p10, p50, p90, rng, n=N_SIM):
    prices = info["감정가"] * sample_ratio(p10, p50, p90, n, rng)
    taxes = rng.choice(TAX_SCENARIOS, n) if info["세금시나리오"] else np.zeros(n)
    S = np.zeros(n)
    for tx in np.unique(taxes):
        m = taxes == tx
        paid, _ = waterfall(prices[m], make_groups(tx))
        S[m] = sum(paid.get(sid, 0) for sid in info["sinbo_ids"])
    return S, prices, taxes


def main():
    print("=" * 70, "\n3단계. 배당 계산기 + 몬테카를로\n" + "=" * 70)
    print("손계산 테스트"); hand_tests()
    if "--test-only" in sys.argv:
        return
    D = load_inputs()
    target, pred = D["target"], D["pred"]
    have = set(D["reg"]["key"])
    rng = np.random.default_rng(SEED)
    rows, tables, dash = [], [], []
    for _, row in target.iterrows():
        k = row["key"]
        if k not in have:
            rows.append(dict(key=k, 상태="대기 (등기부 미입력)")); continue
        pr = pred[pred.key == k]
        if pr.empty:
            rows.append(dict(key=k, 상태="예측값 없음")); continue
        pr = pr.iloc[0]
        make_groups, info = build_case(row, D)
        S, _, _ = simulate(make_groups, info, pr.p10, pr.p50, pr.p90, rng)
        B = info["신보채권액"]

        actual = float(row["매각가(원)"])
        tx_mid = float(np.median(TAX_SCENARIOS)) if info["세금시나리오"] else 0.0
        paid_a, rem_a = waterfall([actual], make_groups(tx_mid))
        for g in make_groups(tx_mid):
            for c in g["claims"]:
                tables.append(dict(key=k, 순위그룹=g["name"], 권리=c["name"], 청구액=round(c["amount"]),
                                   배당액=round(float(paid_a.get(c["id"], [0])[0]))))
        tables.append(dict(key=k, 순위그룹="잉여", 권리="소유자 반환", 청구액=None, 배당액=round(float(rem_a[0]))))

        rows.append(dict(**{kk: vv for kk, vv in info.items() if kk not in ("sinbo_ids", "secured", "general", "prio", "actual_sinbo")},
                         상태="계산 완료", 매각가=actual, p10=pr.p10, p50=pr.p50, p90=pr.p90,
                         회수_중앙값=np.median(S), 회수_평균=S.mean(), 회수_P10=np.percentile(S, 10), 회수_P90=np.percentile(S, 90),
                         영원확률=(S <= 0.5).mean(), 회수율_중앙=np.median(S) / B if B else np.nan,
                         실제가_신보배당=float(sum(paid_a.get(sid, [0])[0] for sid in info["sinbo_ids"]))))

        # 대시보드용: 신보 줄을 뺀 원재료 권리 (날짜·종류 포함 → HTML에서 신보 위치를 바꿔 다시 그룹핑)
        def claim_out(c, sec):
            o = {"name": c["name"], "amount": round(float(c["amount"])), "kind": c.get("kind"), "date": iso(c.get("date"))}
            if c.get("max") is not None: o["max"] = round(float(c["max"]))
            if c.get("basis"): o["basis"] = c["basis"]
            if c.get("subrog"): o["subrog"] = True
            if "cap" in c: o["cap"] = c["cap"]
            if c["id"] in info["sinbo_ids"]: o["sinbo"] = True
            return o
        dash.append(dict(key=k, 법원=row["법원"], 사건번호=row["사건번호"], 물건번호=int(row["물건번호"]), 용도=row["용도"],
                         usage6=pr.usage6, 소재지=row["소재지"], 감정가=info["감정가"], 매각가=actual,
                         p10=float(pr.p10), p50=float(pr.p50), p90=float(pr.p90), 지역=info["지역"],
                         최선순위=iso(info["최선순위"]), 세금시나리오=bool(info["세금시나리오"]), 세금기준일=iso(info["세금기준일"]),
                         실제신보=info["actual_sinbo"], 메모=info["메모"],
                         prio=[claim_out(c, False) for c in info["prio"]],
                         secured=[claim_out(c, True) for c in info["secured"]],
                         general=[claim_out(c, False) for c in info["general"]],
                         # 노트북과 같은 형식의 순위 그룹 (신보 제외, 세금 0)
                         groups=[{"name": g["name"], **({"pool": True} if g.get("pool") else {}),
                                  "claims": [{"name": c["name"], "amount": round(c["amount"]),
                                              **({"date": c["date"], "secured": c["secured"]} if g.get("pool") else {})}
                                             for c in g["claims"] if c["id"] != "SINBO"]}
                                 for g in make_groups(0, include_sinbo=False) if g["name"] != "집행비용"]))

    res = pd.DataFrame(rows)
    done = res[res["상태"] == "계산 완료"]
    print(f"\n계산 완료 {len(done)}건 / 대기 {int((res['상태'] != '계산 완료').sum())}건")
    if len(done):
        show = done[["key", "감정가", "신보채권액", "회수_중앙값", "회수_P10", "회수_P90", "영원확률", "실제가_신보배당"]].copy()
        for c in ["감정가", "신보채권액", "회수_중앙값", "회수_P10", "회수_P90", "실제가_신보배당"]:
            show[c] = (show[c] / 1e6).round(1)
        show["영원확률"] = (show["영원확률"] * 100).round(1).astype(str) + "%"
        print("(금액 단위: 백만 원)"); print(show.to_string(index=False))
        print(f"\n요약: 0원 확률 평균 {done['영원확률'].mean():.0%} | 실제 매각가 기준 신보 0원 사건 {(done['실제가_신보배당'] <= 0.5).mean():.0%}")

    res.to_csv(DIV_DIR / "dividend_results.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(tables).to_csv(DIV_DIR / "dividend_tables.csv", index=False, encoding="utf-8-sig")
    settings = dict(_S, 수집양식=nfc(D["xlsx"].name))
    json.dump({"settings": settings, "cases": dash}, open(DIV_DIR / "dashboard_cases.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=str)
    print(f"\n저장: {DIV_DIR}/dividend_results.csv, dividend_tables.csv, dashboard_cases.json")


if __name__ == "__main__":
    main()
