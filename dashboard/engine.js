/* 회수 시뮬레이터 계산 엔진 — 파이썬 코드와 같은 규칙
 *  [최종 모델] predictAuction2 : ① 회차별 매각확률(transfer + 경매 재보정) × ② 낙찰가 배수(P10/P50/P90 + CQR + 경매 이동폭)
 *                                 → 회차별 몬테카를로 (pipeline/ step4·step6 과 같은 계산)
 *  (아래 predictAuction 은 기존 감정가 기준 모델 — 비교용으로 남겨 둠)
 *  ① predictAuction : LightGBM 나무(P10/P50/P90) → CQR → 공매→경매 보정   (common.AuctionModel, predict_new.build_features)
 *  ② buildCase      : 등기부·임차인 → 순위 그룹                          (step3_dividend.build_case)
 *  ③ waterfall      : 순위대로 채우고 남은 돈만 다음 순위로, 같은 순위는 안분 (step3_dividend.waterfall)
 *  ④ simulate       : 낙찰가율·체납세금 1만 번 뽑기                        (step3_dividend.simulate)
 */
(function (root) {
  "use strict";
  const QS = ["0.1", "0.5", "0.9"];

  /* ───────── ① 낙찰가 모델 ───────── */
  function treeValue(t, x) {
    if (t.f.length === 0) return t.v[0];
    let n = 0;
    while (n >= 0) {
      const fv = x[t.f[n]];
      if (t.c[n]) {                                  // 범주형 분기 (LightGBM CategoricalDecision)
        let iv;
        if (Number.isNaN(fv)) { if (t.mt[n] === 2) { n = t.r[n]; continue; } iv = 0; }
        else { iv = Math.trunc(fv); if (iv < 0) { n = t.r[n]; continue; } }
        n = t.c[n].indexOf(iv) >= 0 ? t.l[n] : t.r[n];
      } else {                                       // 숫자 분기 (NumericalDecision)
        let v = fv;
        if (Number.isNaN(v) && t.mt[n] !== 2) v = 0;
        if ((t.mt[n] === 1 && Math.abs(v) <= 1e-35) || (t.mt[n] === 2 && Number.isNaN(v))) n = t.dl[n] ? t.l[n] : t.r[n];
        else n = v <= t.th[n] ? t.l[n] : t.r[n];
      }
    }
    return t.v[~n];
  }

  const SIDO_SHORT = { "서울특별시": "서울", "부산광역시": "부산", "대구광역시": "대구", "인천광역시": "인천", "광주광역시": "광주",
    "대전광역시": "대전", "울산광역시": "울산", "세종특별자치시": "세종", "경기도": "경기", "강원특별자치도": "강원", "강원도": "강원",
    "충청북도": "충북", "충청남도": "충남", "전북특별자치도": "전북", "전라북도": "전북", "전라남도": "전남", "경상북도": "경북",
    "경상남도": "경남", "제주특별자치도": "제주" };
  const SIDO_GRP = { "광주광역시": "광주·전남", "전라남도": "광주·전남", "전남광주통합특별시": "광주·전남",
    "강원도": "강원특별자치도", "전라북도": "전북특별자치도" };

  function regionKeys(sido, sgg) {
    const a = SIDO_SHORT[sido] || sido;
    if (!sgg || !String(sgg).trim()) return [[a, a, a]];
    const p = String(sgg).trim().split(/\s+/);
    if (p.length >= 2) return [[a, p[0], p[1]], [a, p[0], p[0]], [a, a, a]];
    return [[a, p[0], p[0]], [a, a, a]];
  }
  function addMonths(ym, k) {
    let [y, m] = ym.split("-").map(Number);
    m += k; while (m <= 0) { m += 12; y--; } while (m > 12) { m -= 12; y++; }
    return `${y}-${String(m).padStart(2, "0")}`;
  }
  /** 매각월 2개월 전 지가변동률(lr1) · 그 시점 6개월 누적(lr6). 표보다 미래면 가장 최근 월 */
  function marketRates(rone, sido, sgg, month) {
    let m = addMonths(month, -2);
    if (m > rone.last) m = rone.last;
    let v1 = null, v6 = null;
    for (const k of regionKeys(sido, sgg)) {
      const key = k.join("|");
      if (v1 === null && rone.rate[`${key}|${m}`] !== undefined) v1 = rone.rate[`${key}|${m}`];
      const a = rone.idx[`${key}|${m}`], b = rone.idx[`${key}|${addMonths(m, -6)}`];
      if (v6 === null && a !== undefined && b !== undefined) v6 = (a / b - 1) * 100;
    }
    return { lr1: v1 === null ? NaN : v1, lr6: v6 === null ? NaN : v6, month: m };
  }

  const num = (v) => (v === null || v === undefined || v === "" || Number.isNaN(Number(v)) ? NaN : Number(v));

  /** 입력 → 모델 피처 (predict_new.build_features 와 같은 규칙) */
  function buildFeatures(p, rone, model) {
    const isLand = !(num(p.bld) > 0);
    const apsl = num(p.apsl), land = num(p.land), lp = num(p.lp);
    const mr = marketRates(rone, p.sido, p.sgg, p.month);
    const qtr = Math.floor((Number(p.month.slice(5, 7)) - 1) / 3) + 1;
    const f = {
      log_apsl: Math.log(apsl), landSqms: land, bldSqms: isLand ? NaN : num(p.bld), has_building: isLand ? 0 : 1,
      log_land_price: lp > 0 ? Math.log(lp) : NaN,
      apsl_to_official: lp > 0 && land > 0 ? apsl / (lp * land) : NaN,
      bld_age_c: isLand || Number.isNaN(num(p.age)) ? NaN : Math.min(80, Math.max(0, num(p.age))),
      bld_grnd_flr: isLand ? NaN : num(p.flr),
      bld_tot_area: isLand ? NaN : num(p.tot),
      is_daegu_gb: p.sido === "대구광역시" || p.sido === "경상북도" ? 1 : 0,
      opbd_qtr: qtr, lr_1m: mr.lr1, lr_6m: mr.lr6,
      usage6: p.usage6, sido_grp: SIDO_GRP[p.sido] || p.sido,
      bld_strct: isLand ? null : (p.strct || null), bld_purps: isLand ? null : (p.purps || null),
    };
    if (isLand) f.bldSqms = model.land_bld_zero ? 0 : NaN;   // 학습 데이터는 건물 없는 물건의 건물면적이 0
    return { f, market: mr };
  }

  function vectorize(model, f, prpt) {
    const x = new Array(model.features.length);
    model.num.forEach((k, i) => { const v = f[k]; x[i] = typeof v === "number" ? v : NaN; });
    model.cat.forEach((k, j) => {
      let v = k === "prpt" ? prpt : f[k];
      if (v === null || v === undefined || v === "") v = "없음";
      const idx = model.categories[k].indexOf(String(v));
      x[model.num.length + j] = idx >= 0 ? idx : NaN;
    });
    return x;
  }

  /** 경매 기준 낙찰가율 P10/P50/P90 */
  function predictAuction(model, rone, p) {
    const { f, market } = buildFeatures(p, rone, model);
    const x = vectorize(model, f, model.fix_prpt[p.usage6]);
    const raw = QS.map((q) => model.trees[q].reduce((s, t) => s + treeValue(t, x), 0)).sort((a, b) => a - b);
    const d = model.cqr_by_usage[p.usage6] !== undefined ? model.cqr_by_usage[p.usage6] : model.cqr_delta;
    const gm = [raw[0] - d, raw[1], raw[2] + d];
    const fac = model.factors[p.usage6];
    return { p: gm.map((v) => Math.max(0, v * fac)), gongmae: gm, factor: fac, cqr: d, market, features: f };
  }

  /* ───────── ② 배당: 사건 → 순위 그룹 ───────── */
  const DAY = 86400000;
  const addDay = (d) => new Date(Date.parse(d + "T00:00:00Z") + DAY).toISOString().slice(0, 10);
  const SORTMAX = "9999-99-99";

  function regionOf(sido, sgg, regionTbl) {
    if (sido !== "대구광역시" && sido !== "경상북도") return "그 밖의 지역";
    const tok = String(sgg || "").trim().split(/\s+/)[0] || "";
    const m = tok.match(/^(\S+?(시|군|구))/);
    if (!m) return "그 밖의 지역";
    return regionTbl[(sido === "대구광역시" ? "대구 " : "경북 ") + m[1]] || "그 밖의 지역";
  }

  function smallRule(small, kind, region, baseDate) {
    const d = baseDate || new Date().toISOString().slice(0, 10);
    return small.find((r) => r.kind === kind && r.region === region && r.start <= d && r.end >= d) || null;
  }

  const isNum = (v) => typeof v === "number" && !Number.isNaN(v);
  const DMAX = "9999-99-99", GMAX = "99999", RMAX = "zzzzzz";
  const MORT_TYPES = ["근저당권설정", "전세권설정", "임차권등기"];
  const TAXY = (k) => k === "세금" || k === "공과금";     // 일반채권보다 앞서는 조세·공과금
  const BOOKS = ["토지", "건물"];

  /* 경매 신청 채권자의 청구금액 · 일부이전 연결 (step3_dividend 와 같은 규칙) */
  const CLAIM_RE = /청구금액[:：]?\s*([\d,]{4,})/;
  const ALIAS_RES = [/양도(?:인|전)([^,()—·]+)/g, /([^,()—·]+?)근저당권?양수/g, /변경전(?:상호)?([^,()—·]+)/g];
  const SUBROG_RE = /일부이전|대위변제|대위분/, PARTIAL_RE = /일부이전/, RANK_RE = /을\s*구?\s*(\d+)/g;
  const nfc = (s) => String(s ?? "").normalize("NFC");
  function normName(s) {
    return nfc(s).replace(/\(주\)|㈜|주식회사/g, "").replace(/\(.*?\)/g, "").replace(/\s/g, "");
  }
  function claimOf(r) {
    if (isNum(r.amount)) return r.amount;
    if (isNum(r.claimAmount)) return r.claimAmount;
    const m = nfc(r.memo).replace(/(?<=\d)\s+(?=[\d,])/g, "").match(CLAIM_RE);
    return m ? Number(m[1].replace(/,/g, "")) : null;
  }
  /** 메모에 적힌 원래 채권최고액 (은행 몫에서 신보 이전분을 빼고 입력한 경우) */
  function memoMax(memo) {
    const t = nfc(memo);
    let m = t.match(/채권최고액\s*([\d,]{5,})/); if (m) return Number(m[1].replace(/,/g, ""));
    m = t.match(/원래\s*([\d.]+)\s*억/); if (m) return Number(m[1]) * 1e8;
    return null;
  }
  /** 을구 순위번호 (숫자 문자열). 갑구·모름은 null */
  function rankNoOf(r) {
    if (r.rank_no !== undefined && r.rank_no !== null && r.rank_no !== "") return String(r.rank_no);
    const k = String(r.rank || "");
    if (!/^을/.test(k) && !/^을/.test(String(r.gu || ""))) return null;
    const m = k.match(/\d+/); return m ? m[0] : null;
  }
  /** 등기부별 위치 {토지: {date, rank}, 건물: {...}} — 모르면 null */
  function bookPos(r) {
    if (r.books && Object.keys(r.books).length) {
      const o = {}; for (const [b, p] of Object.entries(r.books)) o[b] = { date: p.date || r.date || null, rank: p.rank !== undefined && p.rank !== null ? String(p.rank) : null };
      return o;
    }
    const b = nfc(r.book).replace(/\s/g, ""), rk = rankNoOf(r), here = { date: r.date || null, rank: rk };
    if (b === "토지" || b === "건물") return { [b]: here };
    if (/토지.*건물|건물.*토지|공통|공동/.test(b)) return { 토지: { ...here }, 건물: { ...here } };
    return null;
  }
  /** 토지·건물 등기부에 한 줄씩 적힌 같은 권리(종류·권리자·금액)를 한 줄로 합침. 등기부별 접수일·순위번호는 _bp 에 */
  function mergeBooks(R) {
    const out = [], idx = new Map();
    R.forEach((r, k) => {
      const bp = bookPos(r);
      const one = bp && Object.keys(bp).length === 1 ? Object.keys(bp)[0] : null;
      if (!one) { out.push({ ...r, _bp: bp, _src: [k] }); return; }
      const key = [r.type, r.who === "신보" ? "신보" : normName(r.name), isNum(r.amount) ? r.amount : claimOf(r)].join("|");
      const list = idx.get(key) || [];
      const mate = list.find((o) => !(one in o._bp));
      if (mate) { mate._bp[one] = bp[one]; mate._src.push(k); if ((r.date || DMAX) < (mate.date || DMAX)) mate.date = r.date; return; }
      const o = { ...r, _bp: { ...bp }, _src: [k] }; out.push(o); list.push(o); idx.set(key, list);
    });
    return out;
  }
  /** 합쳐서 계산할 때의 순위번호: 접수일이 가장 이른 등기부의 번호 */
  function leadRank(r) {
    if (!r._bp) return rankNoOf(r);
    const e = Object.values(r._bp).sort((a, b) => ((a.date || DMAX) < (b.date || DMAX) ? -1 : 1))[0];
    return e ? e.rank : null;
  }
  const rankNum = (r) => (r === null || r === undefined ? Infinity : Number(r));
  const rowOrder = (rows) => (a, b) => {
    const da = rows[a].date || DMAX, db = rows[b].date || DMAX;
    return da < db ? -1 : da > db ? 1 : rankNum(leadRank(rows[a])) - rankNum(leadRank(rows[b])) || a - b;
  };

  /** 신보 근저당 중 '일부이전·대위변제분' → 원래 근저당 줄 번호 */
  function linkSubrog(rows) {
    const links = new Map();
    const base = rows.map((r, o) => o).filter((o) => rows[o].type === "근저당권설정" && rows[o].who !== "신보" && isNum(rows[o].amount));
    rows.forEach((r, j) => {
      if (r.type !== "근저당권설정" || r.who !== "신보") return;
      const memo = nfc(r.memo) + " " + nfc(r.kind);
      if (!SUBROG_RE.test(memo)) return;
      const flat = memo.replace(/\s/g, "");
      const named = base.filter((o) => normName(rows[o].name) && flat.includes(normName(rows[o].name)));
      const same = base.filter((o) => r.date && rows[o].date === r.date);
      let cand = same.length ? same : PARTIAL_RE.test(memo) ? named : [];
      if (!cand.length) return;
      const nums = new Set([...flat.matchAll(RANK_RE)].map((m) => m[1]));
      const myRank = leadRank(r);
      const score = (o) => [leadRank(rows[o]) !== null && leadRank(rows[o]) === myRank ? 0 : 1, nums.has(String(leadRank(rows[o]))) ? 0 : 1, named.includes(o) ? 0 : 1, o];
      cand = cand.slice().sort((a, b) => { const x = score(a), y = score(b); return x[0] - y[0] || x[1] - y[1] || x[2] - y[2] || x[3] - y[3]; });
      links.set(j, cand[0]);
    });
    return links;
  }
  const capOf = (r) => memoMax(r.memo) || r.amount;
  const fmt = (v) => Math.round(v).toLocaleString("ko-KR");

  /** 경매 청구금액 → 신청 채권자 근저당(순위대로 각 최고액 − 신보 몫까지) · 강제경매면 가압류 */
  function resolveAuctionClaims(rows, ratio, links) {
    const over = new Map(), notes = [], filled = new Map();
    const subsOf = (o) => [...links.entries()].filter(([, oo]) => oo === o).reduce((s, [j]) => s + rows[j].amount, 0);
    const morts = rows.map((r, j) => j).filter((j) => rows[j].type === "근저당권설정" && rows[j].who !== "신보" && isNum(rows[j].amount));
    rows.forEach((a) => {
      if (a.type !== "경매개시결정") return;
      const N = claimOf(a);
      if (N === null) return;
      const memo = nfc(a.memo), forced = /강제/.test(memo) || /강제/.test(nfc(a.kind));
      const sinbo = a.who === "신보";
      const names = new Set([normName(a.name)]);
      const flat = memo.replace(/\s/g, "");
      ALIAS_RES.forEach((rx) => { for (const m of flat.matchAll(rx)) names.add(normName(m[1])); });
      names.delete("");
      const same = sinbo ? (r) => r.who === "신보" : (r) => r.who !== "신보" && names.has(normName(r.name));
      const label = sinbo ? "신보" : (a.name || "신청 채권자");
      if (forced) {
        const hits = rows.map((r, j) => j).filter((j) => rows[j].type === "가압류" && isNum(rows[j].amount) && same(rows[j]));
        if (hits.length === 1) { over.set(hits[0], N); notes.push(`${label} 가압류 → 강제경매 청구금액 ${fmt(N)}원 사용`); }
        else if (hits.length > 1) notes.push(`${label} 가압류가 여러 건이라 강제경매 청구금액은 쓰지 않음`);
        return;
      }
      if (sinbo) return;
      let hits = morts.filter((j) => same(rows[j]));
      if (!hits.length) {
        const holders = new Set(morts.map((j) => normName(rows[j].name)));
        if (holders.size === 1 && morts.length) {
          hits = morts.slice();
          notes.push(`[확인 필요] ${label} 청구금액 ${fmt(N)}원을 ${[...holders][0]} 근저당에 연결 (양도인 문구 없음 · 근저당권자가 한 곳뿐)`);
        }
      }
      if (!hits.length) return;
      hits.sort(rowOrder(rows));
      let left = N;
      hits.forEach((j) => {
        const room = Math.max(0, capOf(rows[j]) - subsOf(j) - (filled.get(j) || 0)), give = Math.min(left, room);
        filled.set(j, (filled.get(j) || 0) + give); left -= give; over.set(j, filled.get(j));
      });
      notes.push(hits.length === 1
        ? `${label} 근저당 → ${fmt(over.get(hits[0]))}원 (경매 청구금액 ${fmt(N)}원, 최고액 − 신보 몫 이내)`
        : `${label} 경매 청구금액 ${fmt(N)}원을 근저당 ${hits.length}건에 순위대로 나눔 (각 최고액 − 신보 몫까지)`);
      if (left > 0.5) notes.push(`${label} 청구금액 중 ${fmt(left)}원은 최고액을 넘어 담보 밖`);
    });
    return { over, notes };
  }

  /**
   * cs: {apsl, baseDate, region, shares?:{토지, 건물}, rights:[{type, who, name, date, amount, valid, memo, rank, book|books}], tenants:[...]}
   * S : {sinboMode, sinboDate, sinboAmount, useActualSinbo, costRate, costFixed, mortgageRatio, includeAuctionClaim,
   *      useAuctionClaim, absorb, partialSubrogation, taxFirst, splitBooks}
   */
  function buildCase(cs, S, small) {
    const notes = [];
    const label = (r) => `${r.name && r.name !== "개인" ? r.name : (r.who || "")} ${r.type}`.trim();
    const R = mergeBooks(cs.rights.filter((r) => r.valid !== false));
    const auctionDates = R.filter((r) => r.type === "경매개시결정" && r.date).map((r) => r.date).sort();
    const auctionDate = auctionDates[0] || null;
    let secured = [], general = [];
    const prio = [], taxDates = [], hidden = [];
    let taxNoAmt = 0;
    const links = linkSubrog(R);
    const { over, notes: cnotes } = S.useAuctionClaim !== false ? resolveAuctionClaims(R, S.mortgageRatio, links) : { over: new Map(), notes: [] };
    notes.push(...cnotes);
    const subsOf = (o) => [...links.entries()].filter(([, oo]) => oo === o).reduce((s, [j]) => s + R[j].amount, 0);
    const jclaim = new Map();
    R.forEach((r, i) => {
      const id = "r" + i, typ = r.type, who = r.who;
      const amt = typ === "경매개시결정" ? null : isNum(r.amount) ? r.amount : null;
      const base = { date: r.date || null, rank: leadRank(r), bp: r._bp || null, src: i, srcs: r._src };
      if (who === "신보" && !S.useActualSinbo) return;
      if (who === "신보" && !["근저당권설정", "전세권설정", "가압류", "경매개시결정"].includes(typ)) notes.push(`신보 '${typ}' 행은 배당에 안 들어감`);
      if (who === "근로복지공단") {
        if (amt !== null) prio.push({ id, name: "근로복지공단(임금)", amount: amt, kind: "임금", src: i });
        else notes.push("근로복지공단 금액 없음 → 0 처리");
        return;
      }
      if (MORT_TYPES.includes(typ)) {
        if (amt === null) { notes.push(`${label(r)} 금액 없음 → 제외`); return; }
        let a, basis;
        if (over.has(i)) { a = over.get(i); basis = "청구금액"; }
        else if (typ === "근저당권설정" && who !== "신보") { a = Math.max(0, capOf(r) / S.mortgageRatio - subsOf(i)); basis = S.mortgageRatio !== 1 ? "추정" : "최고액"; }
        else { a = amt; basis = "등기"; }
        const c = { id, name: label(r), amount: a, max: amt, kind: "담보", basis, who, ...base };
        if (links.has(i)) {                     // 일부이전: 새 근저당이 아니라 원래 근저당 순위 안의 신보 몫
          const o = R[links.get(i)]; c.subrog = true; c.parentId = "r" + links.get(i);
          c.rank = leadRank(o); c.bp = o._bp || c.bp;
          if (o.date !== r.date) { c.date = o.date; notes.push(`신보 일부이전 → 원래 근저당(${o.name} ${o.date}) 순위로`); }
        }
        jclaim.set(i, c);
        secured.push(c);
      } else if (typ === "압류") {
        if (who === "세무서" || who === "지자체") {
          taxDates.push(r.date || null);
          if (amt !== null) secured.push({ id, name: label(r), amount: amt, kind: "세금", ...base });
          else { taxNoAmt++; hidden.push(r.name || who); }
        } else if (amt !== null) secured.push({ id, name: label(r), amount: amt, kind: "공과금", ...base });
        else { notes.push(`${label(r)} 금액 없음 → 0 처리 (납부기한이 근저당보다 빠르면 숨은 선순위일 수 있음)`); hidden.push(r.name || who); }
      } else if (typ === "가압류") {
        if (amt === null) { notes.push(`${label(r)} 금액 없음 → 제외`); return; }
        general.push({ id, name: label(r), amount: over.has(i) ? over.get(i) : amt, kind: "가압류", who, ...base, ...(over.has(i) ? { basis: "청구금액" } : {}) });
      } else if (typ === "경매개시결정" && S.includeAuctionClaim && claimOf(r) !== null) {
        general.push({ id, name: label(r), amount: claimOf(r), kind: "가압류", who, ...base });
      }
    });

    // 같은 근저당 안의 은행 몫 + 신보 몫 ≤ 채권최고액 (메모에 원래 최고액이 있으면 그 값)
    [...new Set(links.values())].sort((a, b) => a - b).forEach((o) => {
      if (!jclaim.has(o)) return;
      const mem = [...links.entries()].filter(([j, oo]) => oo === o && jclaim.has(j)).map(([j]) => jclaim.get(j));
      const M = capOf(R[o]), bank = jclaim.get(o);
      const tot = bank.amount + mem.reduce((s, c) => s + c.amount, 0);
      if (tot > M + 0.5) {
        if (S.partialSubrogation === "bank_first") {
          let cut = tot - M;
          mem.forEach((c) => { const d = Math.min(cut, c.amount); c.amount -= d; cut -= d; });
          bank.amount -= cut;
        } else [bank, ...mem].forEach((c) => { c.amount *= M / tot; });
        notes.push(`${R[o].name} 근저당: 은행 몫 + 신보 몫 ${fmt(tot)}원이 채권최고액 ${fmt(M)}원을 넘어 최고액에 맞춤`);
      }
    });

    (cs.tenants || []).forEach((t, i) => {
      const tl = `임차인${t.no ?? i + 1}`;
      if (String(t.claim || "").trim().toUpperCase() !== "Y") { notes.push(`${tl} 배당요구 없음 → 제외`); return; }
      const kind = String(t.kind || "").trim() === "상가" ? "상가" : "주택";
      const dep = isNum(t.deposit) ? t.deposit : 0, rent = isNum(t.rent) ? t.rent : 0;
      const judge = kind === "상가" ? dep + rent * 100 : dep;
      const rule = smallRule(small, kind, cs.region, cs.baseDate);
      let first = 0;
      const okTime = !auctionDate || (t.moved && t.moved < auctionDate);
      if (rule && judge <= rule.limit && okTime) {
        first = Math.min(dep, rule.amount);
        prio.push({ id: `t${i}a`, name: `${tl} 최우선`, amount: first, cap: rule.ratio, kind: "소액임차", tsrc: i });
      }
      const rest = dep - first;
      if (rest > 0 && t.confirmed && t.moved) {
        const a = addDay(t.moved), d = a > t.confirmed ? a : t.confirmed;
        secured.push({ id: `t${i}b`, name: `${tl} 확정일자`, amount: rest, date: d, rank: null, bp: null, kind: "임차인", tsrc: i });
      }
    });

    // 신보 채권: 근저당(일부이전·자기 근저당) + 가압류 중 담보를 넘는 부분만 일반채권 한 줄 (같은 구상채권으로 보고 중복 제거)
    let B = S.sinboAmount, sinboIds, sinboClaim = null, actualSinbo = false;
    const sbSec = secured.filter((c) => c.who === "신보"), sbAtt = general.filter((c) => c.who === "신보");
    if (S.useActualSinbo && (sbSec.length || sbAtt.length)) {
      actualSinbo = true;
      const own = sbSec.filter((c) => !c.subrog);
      const sbAuction = [...new Set(R.filter((r) => r.type === "경매개시결정" && r.who === "신보" && !/강제/.test(nfc(r.memo) + nfc(r.kind))).map(claimOf).filter((v) => v !== null))];
      if (own.length && sbAuction.length) {        // 신보 자기 근저당: 신보 임의경매 청구금액을 순위대로 최고액까지
        let left = sbAuction.reduce((a, v) => a + v, 0) - sbSec.filter((c) => c.subrog).reduce((a, c) => a + c.amount, 0);
        own.forEach((c) => { const v = Math.max(0, Math.min(left, c.max)); left -= v; c.amount = v; c.basis = "청구금액"; });
      }
      const secAmt = sbSec.reduce((a, c) => a + c.amount, 0);
      const sub = sbSec.filter((c) => c.subrog).reduce((a, c) => a + c.amount, 0);
      if (sbAuction.length && !own.length && sub > 0 && Math.abs(sbAuction.reduce((a, v) => a + v, 0) - sub) > 1)
        notes.push(`[확인] 신보 임의경매 청구 합 ${fmt(sbAuction.reduce((a, v) => a + v, 0))}원 ≠ 일부이전 합 ${fmt(sub)}원`);
      if (sbSec.length && sbAtt.length) {
        const attAmt = sbAtt.reduce((a, c) => a + c.amount, 0), rem = Math.max(0, attAmt - secAmt);
        general = general.filter((c) => !sbAtt.includes(c));
        if (rem > 0.5) {
          const d0 = sbAtt.map((c) => c.date).filter(Boolean).sort()[0] || null;
          general.push({ id: "SBGEN", name: "신용보증기금 가압류 (담보 초과분)", amount: rem, kind: "가압류", who: "신보", date: d0, rank: null, bp: null, basis: "담보 초과분" });
        }
        notes.push(`신보 가압류 ${fmt(attAmt)}원 중 근저당으로 받는 ${fmt(secAmt)}원을 빼고 나머지 ${fmt(rem)}원만 일반채권 (같은 구상채권으로 가정)`);
      }
      const ids = secured.filter((c) => c.who === "신보").concat(general.filter((c) => c.who === "신보"));
      sinboIds = ids.map((c) => c.id);
      B = ids.reduce((a, c) => a + c.amount, 0);
      notes.push(`신보 채권액 = ${fmt(B)}원 (근저당 ${fmt(secAmt)}원${ids.some((c) => c.kind === "가압류") ? " + 가압류" : ""})`);
      notes.push("등기부의 실제 신보 채권 사용");
    } else {
      const mode = S.sinboMode;
      sinboClaim = { id: "SINBO", name: `신보 (${{ mortgage: "근저당", attach: "가압류" }[mode] || "일반채권"})`,
        amount: B, date: mode !== "general" ? (S.sinboDate || null) : null, rank: null, bp: null, kind: mode === "mortgage" ? "담보" : "가압류", sinbo: true };
      sinboIds = ["SINBO"];
    }
    const taxNeeded = taxNoAmt > 0;
    const taxDate = taxDates.filter(Boolean).sort()[0] || null;
    if (taxNeeded || secured.some((c) => TAXY(c.kind))) notes.push("세금·공과금 순서: 법정기일을 몰라 압류일로 대신 정함 (일반채권보다는 앞)");
    // 집행비용: 금액을 알면 그 금액, 모르면 감정가 × 비율 (가정값)
    const costAssumed = !(S.costFixed > 0);
    const cost = costAssumed ? cs.apsl * S.costRate : S.costFixed;

    // 토지·건물 등기부별 배당이 필요한지: 한쪽에만 있는 근저당이 있거나, 두 등기부의 근저당 순서가 다르면
    const mortR = R.filter((r) => MORT_TYPES.includes(r.type) && isNum(r.amount));
    const books = BOOKS.filter((b) => R.some((r) => r._bp && r._bp[b]));
    const sh = cs.shares && cs.shares.토지 > 0 && cs.shares.건물 > 0 ? { 토지: cs.shares.토지 / (cs.shares.토지 + cs.shares.건물), 건물: cs.shares.건물 / (cs.shares.토지 + cs.shares.건물) } : null;
    const why = [];
    if (books.length === 2 && mortR.every((r) => r._bp)) {
      const lone = mortR.filter((r, i) => !links.has(R.indexOf(r)) && Object.keys(r._bp).length < 2);
      if (lone.length) why.push(`한쪽 등기부에만 있는 근저당 ${lone.length}건`);
      const joint = mortR.filter((r) => !links.has(R.indexOf(r)) && Object.keys(r._bp).length === 2);
      const ord = (b) => joint.map((r, k) => [r._bp[b].date || DMAX, rankNum(r._bp[b].rank), k]).sort((x, y) => (x[0] < y[0] ? -1 : x[0] > y[0] ? 1 : x[1] - y[1] || x[2] - y[2])).map((x) => x[2]).join(",");
      if (ord("토지") !== ord("건물")) why.push("토지·건물 등기부의 근저당 순서가 다름");
    }
    const split = why.length > 0 && !!sh && S.splitBooks !== false;
    if (why.length && !sh) notes.push(`${why.join(" · ")} → 등기부별로 나눠 배당해야 하지만 토지 감정가가 없어 합쳐서 계산했어요`);
    if (split) notes.push(`${why.join(" · ")} → 매각대금을 토지·건물 감정가 비율로 나누고, 공동저당은 각 부동산 경매대가 비율로 분담해 등기부별로 배당 (민법 368조 · 대법원 2001다66291)`);
    if (!books.length && mortR.length) {
      const n1 = R.filter((r) => /토지만|건물만/.test(nfc(r.memo))).length;
      if (n1) notes.push(`등기부(토지/건물) 구분이 없어 합쳐서 계산했어요 — 메모상 한쪽 등기부에만 있는 권리 ${n1}건, 순위가 다를 수 있음`);
    }

    const lag = S.partialSubrogation === "bank_first";
    const key = (c) => (c.date || DMAX) + "|" + (c.rank !== null && c.rank !== undefined ? String(c.rank).padStart(6, "0") : RMAX) + (lag && c.subrog ? "~" : "");
    const stableSort = (arr, k) => arr.map((c, i) => [c, i]).sort((a, b) => { const x = k(a[0]), y = k(b[0]); return x < y ? -1 : x > y ? 1 : a[1] - b[1]; }).map((p) => p[0]);

    function makeGroups(tax = 0, includeSinbo = true, taxFirst = !!S.taxFirst) {
      let sec = secured.slice(), gen = general.slice();
      if (includeSinbo && sinboClaim) (sinboClaim.kind === "담보" ? sec : gen).push(sinboClaim);
      const g = [{ name: "집행비용", claims: [{ id: "COST", name: costAssumed ? `집행비용 (가정: 감정가 × ${+(S.costRate * 100).toFixed(2)}%)` : "집행비용", amount: cost, kind: "비용" }] }];
      if (prio.length) {
        const caps = new Map();
        prio.forEach((c, j) => { if (c.cap !== undefined) { if (!caps.has(c.cap)) caps.set(c.cap, []); caps.get(c.cap).push(j); } });
        g.push({ name: "최우선", claims: prio, caps: [...caps.entries()] });
      }
      if (tax > 0) {
        const tc = { id: "TAX", name: taxFirst ? "체납세금 (추정 · 당해세 가정)" : "체납세금 (추정)", amount: tax, date: taxDate, rank: null, bp: null, kind: "세금" };
        if (taxFirst) g.push({ name: "체납세금 (당해세 가정)", claims: [tc] }); else sec.push(tc);
      }
      const mort = sec.filter((c) => c.kind === "담보" && c.date).map((c) => c.date).sort();
      const early = gen.filter((c) => c.date && mort.some((m) => c.date < m));
      let pool = [], d0 = null;
      if (early.length && S.absorb !== false) {
        // 안분 후 흡수: 근저당보다 먼저인 가압류 중 가장 이른 날짜부터 뒤의 권리는 한 풀에서 안분 → 담보권·세금이 후순위 몫 흡수
        d0 = early.map((c) => c.date).sort()[0];
        sec = stableSort(sec, key);
        pool = sec.filter((c) => key(c) >= d0).map((c) => ({ ...c, date: c.date ? key(c) : null, secured: true }))
          .concat(gen.map((c) => ({ ...c, secured: false, ...(early.includes(c) ? { kind: "가압류(안분)", origDate: c.date } : {}) })));
        pool = stableSort(pool, (c) => c.date || (c.secured ? DMAX : GMAX));
        sec = sec.filter((c) => key(c) < d0);
        gen = [];
      } else if (early.length) {
        const still = [];
        gen.forEach((c) => {
          const later = c.date ? mort.filter((m) => c.date < m) : [];
          if (later.length) sec.push({ ...c, date: later[0], rank: null, kind: "가압류(안분)", origDate: c.date });
          else still.push(c);
        });
        gen = still;
      }
      sec = stableSort(sec, key);
      if (split && sec.length) g.push(splitGroup(sec));
      else {
        let cur = null;
        sec.forEach((c) => {
          const k = key(c);
          if (!cur || cur.key !== k) {
            const d = k.slice(0, 10), rk = c.rank !== null && c.rank !== undefined ? ` · 을${c.rank}` : "";
            cur = { name: k.startsWith(DMAX) ? "날짜 미상" : `날짜 ${d}${rk}${k.endsWith("~") ? " (일부대위 · 원래 권리자 다음)" : ""}`, key: k, date: d, claims: [] };
            g.push(cur);
          }
          cur.claims.push(c);
        });
      }
      if (pool.length) g.push({ name: `안분·흡수 (가압류 ${d0}부터)`, pool: true, claims: pool });
      if (gen.length) g.push({ name: "일반채권", claims: gen });
      return g;
    }

    /** 등기부별 배당 그룹: 근저당(신보 일부이전은 원래 근저당에 합침)마다 등기부별 순위 */
    function splitGroup(sec) {
      const pos = [], byId = new Map();
      sec.forEach((c) => { if (c.subrog && c.parentId) return; const p = { members: [c], bp: c.bp }; pos.push(p); byId.set(c.id, p); });
      sec.forEach((c) => { if (!(c.subrog && c.parentId)) return; const p = byId.get(c.parentId); if (p) p.members.push(c); else { const q = { members: [c], bp: c.bp }; pos.push(q); byId.set(c.id, q); } });
      pos.forEach((p) => {
        const lead = p.members[0];
        p.bp = p.bp || Object.fromEntries(books.map((b) => [b, { date: lead.date, rank: null }]));
        p.books = books.filter((b) => p.bp[b]);
        p.want = p.members.reduce((a, c) => a + c.amount, 0);
      });
      const orders = {};
      books.forEach((b) => {
        const ix = pos.map((p, i) => i).filter((i) => pos[i].bp[b]);
        const kb = (i) => (pos[i].bp[b].date || DMAX) + "|" + (pos[i].bp[b].rank !== null && pos[i].bp[b].rank !== undefined ? String(pos[i].bp[b].rank).padStart(6, "0") : RMAX);
        ix.sort((x, y) => (kb(x) < kb(y) ? -1 : kb(x) > kb(y) ? 1 : x - y));
        const grps = []; let last = null;
        ix.forEach((i) => { const k = kb(i); if (k !== last) { grps.push([]); last = k; } grps[grps.length - 1].push(i); });
        orders[b] = grps;
      });
      return { name: "토지·건물 등기부별 배당 (민법 368조)", split: true, books, shares: sh, positions: pos.map((p) => ({ members: p.members.map((c) => c.id), books: p.books, want: p.want })),
        orders, rule: lag ? "bank_first" : "pro_rata", claims: pos.flatMap((p) => p.members) };
    }

    return { makeGroups, sinboIds, B, taxNeeded, taxDate, notes: [...new Set(notes)], actualSinbo, auctionDate,
      cost, costAssumed, secured, general, prio, hidden, split, shares: sh, books };
  }

  /* ───────── ③ 워터폴 ───────── */
  /** 풀 안에서 t가 s보다 후순위인가 (s가 흡수할 수 있는 상대). 세금·공과금은 앞선 가압류 몫에서도 가져옴 */
  const subordinate = (t, s) => (TAXY(s.kind) && !t.secured) || (t.date || (t.secured ? DMAX : GMAX)) > (s.date || DMAX);

  /** 등기부별 배당 한 번 (낙찰가 1개): 각 부동산 경매대가 = 그 부동산 몫 − 선순위(공동저당은 분담액) − 같은 순위 몫, 맞을 때까지 반복 */
  function splitOnce(g, M) {
    const B = g.books, P = g.positions, nb = B.length;
    const N = B.map((b) => M * g.shares[b]);
    const has = P.map((p) => B.map((b) => p.books.includes(b)));
    const joint = P.map((p) => p.books.length > 1);
    const shSum = P.map((p) => p.books.reduce((a, b) => a + g.shares[b], 0));
    const d = P.map((p, i) => B.map((b, bi) => (!has[i][bi] ? 0 : joint[i] ? p.want * g.shares[b] / shSum[i] : p.want)));
    for (let it = 0; it < 200; it++) {
      const a = P.map(() => new Array(nb).fill(0));
      B.forEach((b, bi) => {
        let rem = N[bi];
        for (const grp of g.orders[b]) {
          let tot = 0; for (const i of grp) tot += d[i][bi];
          const give = Math.min(tot, Math.max(rem, 0));
          const part = grp.map((i) => (tot > 0 ? d[i][bi] / tot * give : 0));
          let all = 0; for (const v of part) all += v;
          grp.forEach((i, k) => { a[i][bi] = Math.max(0, rem - (all - part[k])); });
          rem -= give;
        }
      });
      let diff = 0;
      P.forEach((p, i) => {
        if (!joint[i]) return;
        let S = 0; B.forEach((b, bi) => { if (has[i][bi]) S += a[i][bi]; });
        B.forEach((b, bi) => { if (!has[i][bi]) return; const v = S > 0 ? Math.min(a[i][bi], p.want * a[i][bi] / S) : 0; diff = Math.max(diff, Math.abs(v - d[i][bi])); d[i][bi] = v; });
      });
      if (diff < 0.01) break;
    }
    const got = new Array(P.length).fill(0), left = new Array(nb).fill(0);
    B.forEach((b, bi) => {
      let rem = N[bi];
      for (const grp of g.orders[b]) {
        let tot = 0; for (const i of grp) tot += d[i][bi];
        const give = Math.min(tot, Math.max(rem, 0));
        grp.forEach((i) => { got[i] += tot > 0 ? d[i][bi] / tot * give : 0; });
        rem -= give;
      }
      left[bi] = rem;
    });
    // 남은 돈이 있는 등기부: 그 등기부의 근저당이 아직 못 받은 만큼 순위대로 (일반채권보다 먼저)
    B.forEach((b, bi) => {
      let rem = left[bi];
      for (const grp of g.orders[b]) {
        if (rem <= 0) break;
        const sf = grp.map((i) => Math.max(0, P[i].want - got[i]));
        let tot = 0; for (const v of sf) tot += v;
        const give = Math.min(tot, rem);
        grp.forEach((i, k) => { got[i] += tot > 0 ? sf[k] / tot * give : 0; });
        rem -= give;
      }
      left[bi] = rem;
    });
    return { got, left: left.reduce((x, y) => x + y, 0) };
  }

  /** prices: 배열(낙찰가). 반환 {paid: {id: Float64Array}, remain} */
  function waterfall(prices, groups) {
    const n = prices.length;
    const remain = Float64Array.from(prices);
    const paid = {};
    for (const g of groups) {
      const k = g.claims.length;
      if (!k) continue;
      for (const c of g.claims) if (!paid[c.id]) paid[c.id] = new Float64Array(n);
      if (g.split) {
        const byId = new Map(g.claims.map((c) => [c.id, c]));
        for (let s = 0; s < n; s++) {
          const M = Math.max(remain[s], 0), r = splitOnce(g, M);
          g.positions.forEach((p, i) => {
            const mem = p.members.map((id) => byId.get(id));
            let T = r.got[i];
            const pay = (list) => { let w = 0; for (const c of list) w += c.amount; const give = Math.min(T, w); list.forEach((c) => { paid[c.id][s] += w > 0 ? c.amount / w * give : 0; }); T -= give; };
            if (g.rule === "bank_first") { pay(mem.filter((c) => !c.subrog)); pay(mem.filter((c) => c.subrog)); }
            else pay(mem);
          });
          remain[s] = remain[s] - M + r.left;
        }
        continue;
      }
      const W = g.claims.map((c) => new Float64Array(n).fill(Number(c.amount)));
      for (const [rate, idx] of g.caps || []) {        // 소액임차인: 배당할 돈(매각가 − 집행비용) × 한도비율까지
        for (let s = 0; s < n; s++) {
          let tot = 0; for (const j of idx) tot += W[j][s];
          const lim = Math.max(remain[s], 0) * rate;
          const sc = tot > lim ? lim / Math.max(tot, 1e-9) : 1.0;
          for (const j of idx) W[j][s] *= sc;
        }
      }
      // 흡수 순서: 담보권·세금을 날짜순으로, 각자 자기보다 후순위인 권리 목록
      const absorbers = g.pool ? g.claims.map((c, j) => j).filter((j) => g.claims[j].secured)
        .map((j, o) => [j, o]).sort((a, b) => { const x = g.claims[a[0]].date || DMAX, y = g.claims[b[0]].date || DMAX; return x < y ? -1 : x > y ? 1 : a[1] - b[1]; })
        .map(([j]) => [j, g.claims.map((c, t) => t).filter((t) => t !== j && subordinate(g.claims[t], g.claims[j]))]).filter(([, subs]) => subs.length) : [];
      const share = new Float64Array(k);
      for (let s = 0; s < n; s++) {
        let total = 0; for (let j = 0; j < k; j++) total += W[j][s];
        const give = Math.min(total, Math.max(remain[s], 0));
        const den = Math.max(total, 1e-9);
        for (let j = 0; j < k; j++) share[j] = total > 0 ? (W[j][s] / den) * give : 0;
        for (const [j, subs] of absorbers) {
          const need = Math.max(W[j][s] - share[j], 0);
          let avail = 0; for (const t of subs) avail += share[t];
          const take = Math.min(need, avail), f = avail > 0 ? take / Math.max(avail, 1e-9) : 0;
          for (const t of subs) share[t] *= 1 - f;
          share[j] += take;
        }
        for (let j = 0; j < k; j++) paid[g.claims[j].id][s] += share[j];
        remain[s] -= give;
      }
    }
    return { paid, remain };
  }

  /* ───────── ④ 몬테카를로 ───────── */
  function mulberry32(a) {
    return function () {
      a |= 0; a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function interp(u, xs, ys) {
    for (let i = 1; i < xs.length; i++) if (u <= xs[i]) return ys[i - 1] + (ys[i] - ys[i - 1]) * (u - xs[i - 1]) / (xs[i] - xs[i - 1]);
    return ys[ys.length - 1];
  }
  /** P10/P50/P90 사이 선형 보간 + 꼬리 (P10−0.5×(P50−P10) ~ P90+0.5×(P90−P50)), 하한 1% */
  function sampleRatios(p, n, rng) {
    const [a, b, c] = p.slice().sort((x, y) => x - y);
    const xs = [0, 0.1, 0.5, 0.9, 1], ys = [a - 0.5 * (b - a), a, b, c, c + 0.5 * (c - b)];
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) out[i] = Math.max(interp(rng(), xs, ys), 0.01);
    return out;
  }
  function percentile(sorted, q) {               // numpy 기본(linear)과 같은 방식
    const pos = (sorted.length - 1) * q, lo = Math.floor(pos), hi = Math.ceil(pos);
    return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
  }
  function sinboOf(paid, ids, n) {
    const S = new Float64Array(n);
    ids.forEach((id) => { if (paid[id]) for (let i = 0; i < n; i++) S[i] += paid[id][i]; });
    return S;
  }
  /** 같은 낙찰가 표본으로 신보 회수액 1만 개 */
  function simulate(built, apsl, p, taxScenarios, n = 10000, seed = 2026, taxFirst) {
    const rng = mulberry32(seed);
    const ratios = sampleRatios(p, n, rng);
    const prices = ratios.map((r) => r * apsl);
    const taxes = new Float64Array(n);
    if (built.taxNeeded) for (let i = 0; i < n; i++) taxes[i] = taxScenarios[Math.floor(rng() * taxScenarios.length)];
    const S = new Float64Array(n);
    const byTax = {};
    for (const tx of new Set(taxes)) {
      const idx = []; for (let i = 0; i < n; i++) if (taxes[i] === tx) idx.push(i);
      const { paid } = waterfall(idx.map((i) => prices[i]), built.makeGroups(tx, true, taxFirst));
      const s = sinboOf(paid, built.sinboIds, idx.length);
      idx.forEach((i, j) => { S[i] = s[j]; });
      byTax[tx] = summarize(s, built.B);
    }
    return { S, prices, taxes, summary: summarize(S, built.B), byTax };
  }
  function summarize(S, B) {
    const s = Float64Array.from(S).sort();
    let zero = 0, sum = 0; for (const v of s) { if (v <= 0.5) zero++; sum += v; }
    const med = percentile(s, 0.5);
    return { median: med, p10: percentile(s, 0.1), p90: percentile(s, 0.9), mean: sum / s.length,
      zero: zero / s.length, rate: B ? med / B : NaN, n: s.length };
  }
  /** 이 값 이하로 팔리면 신보 0원 (세금 0, 이분법) */
  function breakeven(built, apsl) {
    const g = built.makeGroups(0);
    const at = (P) => sinboOf(waterfall([P], g).paid, built.sinboIds, 1)[0];
    let lo = 0, hi = apsl * 5;
    if (at(hi) <= 0.5) return null;
    for (let i = 0; i < 60; i++) { const mid = (lo + hi) / 2; if (at(mid) > 0.5) hi = mid; else lo = mid; }
    return lo;
  }
  /** 한 가격에서의 배당표 */
  function table(built, price, tax, taxFirst) {
    const groups = built.makeGroups(tax, true, taxFirst);
    const { paid, remain } = waterfall([price], groups);
    const rows = [];
    groups.forEach((g, gi) => g.claims.forEach((c) => rows.push({ group: g.name, gi, name: c.name, kind: c.kind, claim: c.amount,
      paid: paid[c.id] ? paid[c.id][0] : 0, sinbo: built.sinboIds.includes(c.id), max: c.max, origDate: c.origDate,
      basis: c.basis, pool: !!g.pool, subrog: !!c.subrog, date: c.date || null, rank: typeof c.rank === "string" ? c.rank : null, bp: c.bp || null })));
    return { rows, remain: remain[0] };
  }


  /* ───────── [최종 모델] ① 매각확률 × ② 낙찰가 배수, 회차별 몬테카를로 ───────── */
  const SIDO_SHORT2 = Object.assign({}, SIDO_SHORT, { "전남광주통합특별시": "전남" });
  const SHIFT_BLDG = ["아파트", "기타주거", "상가·업무", "공장·창고"];        // has_building = 1 (4단계 load_cases 와 같은 규칙)
  function regionKeys2(sido, sgg) {
    const a = SIDO_SHORT2[sido] || sido;
    if (!sgg || !String(sgg).trim()) return [[a, a, a]];
    const p = String(sgg).trim().split(/\s+/);
    if (p.length >= 2) return [[a, p[0], p[1]], [a, p[0], p[0]], [a, a, a]];
    return [[a, p[0], p[0]], [a, a, a]];
  }
  /** 매각월 − lag 개월의 월 변동률(lr_1m)과 3·6·12개월 누적 (common.attach_market 과 같음). 표보다 미래면 가장 최근 월 */
  function marketRates2(rone, sido, sgg, month, lag) {
    let m = addMonths(month, -lag);
    const capped = m > rone.last;
    if (capped) m = rone.last;
    const v = { lr_1m: null, lr_3m: null, lr_6m: null, lr_12m: null };
    for (const k of regionKeys2(sido, sgg)) {
      const key = k.join("|");
      if (v.lr_1m === null && rone.rate[`${key}|${m}`] !== undefined) v.lr_1m = rone.rate[`${key}|${m}`];
      for (const n of [3, 6, 12]) {
        const a = rone.idx[`${key}|${m}`], b = rone.idx[`${key}|${addMonths(m, -n)}`];
        if (v[`lr_${n}m`] === null && a !== undefined && b !== undefined) v[`lr_${n}m`] = (a / b - 1) * 100;
      }
    }
    for (const k in v) if (v[k] === null) v[k] = NaN;
    return { ...v, month: m, capped };
  }
  /** 예측 시점 최근 공매 지표 (4단계 recent_market: 매각월 직전 3개월 낙찰 중앙값). 표보다 미래면 가장 최근 값 */
  function recentStats(M2, usage6, month) {
    const R = M2.recent, ks = Object.keys(R.byMonth).sort();
    let k = month;
    if (k > ks[ks.length - 1]) k = ks[ks.length - 1];
    if (k < ks[0]) k = ks[0];
    const t = R.byMonth[k] || {};
    const g = (o) => (o && o[usage6] !== undefined && o[usage6] !== null ? o[usage6] : NaN);
    return { r1: g(t.r1), r2: g(t.r2), month: k, months: t.months || [] };
  }
  const roundHalfEven = (x) => { const r = Math.round(x); return Math.abs(x % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r; };  // numpy round 와 같게
  const logit = (p) => { const q = Math.min(Math.max(p, 1e-6), 1 - 1e-6); return Math.log(q / (1 - q)); };
  const sigmoid = (z) => 1 / (1 + Math.exp(-z));

  /** 입력 → 피처 (최저가율·유찰횟수는 회차마다 따로 넣음) */
  function buildFeatures2(p, rone, M2) {
    const apsl = num(p.apsl), land = num(p.land), lp = num(p.lp), bld = num(p.bld);
    const mr = marketRates2(rone, p.sido, p.sgg, p.month, M2.lag);
    const rs = recentStats(M2, p.usage6, p.month);
    const hasB = SHIFT_BLDG.includes(p.usage6);
    const f = {
      재매각: 0, log_apsl: Math.log(apsl), landSqms: land, bldSqms: bld, has_building: hasB ? 1 : 0,
      log_land_price: lp > 0 ? Math.log(lp) : NaN, apsl_to_official: lp > 0 && land > 0 ? apsl / (lp * land) : NaN,
      bld_age_c: num(p.age) >= 0 ? Math.min(80, Math.max(0, num(p.age))) : NaN, bld_grnd_flr: num(p.flr), bld_tot_area: num(p.tot),
      is_daegu_gb: p.sido === "대구광역시" || p.sido === "경상북도" ? 1 : 0,
      opbd_qtr: Math.floor((Number(String(p.month).slice(5, 7)) - 1) / 3) + 1,
      lr_1m: mr.lr_1m, lr_3m: mr.lr_3m, lr_6m: mr.lr_6m, lr_12m: mr.lr_12m,
      최근_낙찰가율_용도: rs.r1, 최근_배수_용도유형: rs.r2,
      usage6: p.usage6, sido_grp: SIDO_GRP[p.sido] || p.sido,
      bld_strct: p.strct || null, bld_purps: p.purps || null,
    };
    return { f, market: mr, recent: rs };
  }
  const treeSum = (trees, x) => trees.reduce((s, t) => s + treeValue(t, x), 0);

  /** 회차표: 회차마다 최저가율(×0.7씩), ① 경매 매각확률, ② log 배수 P10/P50/P90 (경매 보정 포함) — step6.round_table 과 같음 */
  function roundTable(M2, f, startRto) {
    const S = M2.sell, Q = M2.mult, A = M2.shift, prpt = M2.prpt, out = [];
    const g = A.groups[f.usage6] || "기타";
    const s = A.logByGroup[g] !== undefined ? A.logByGroup[g] : A.logByGroup["(전체)"];
    const d = Q.cqr_by_usage[f.usage6] !== undefined ? Q.cqr_by_usage[f.usage6] : Q.cqr_all;
    for (let k = 0; k < M2.maxRounds; k++) {
      const rto = startRto * Math.pow(M2.drop, k);
      const fk = { ...f, rto, 누적_유찰횟수: Math.max(0, roundHalfEven((100 - rto) / 10)) };
      const pG = sigmoid(treeSum(S.trees, vectorize(S, fk, prpt)));            // 공매 transfer 모델
      const kA = Math.min(M2.kMax, Math.max(0, roundHalfEven(Math.log(Math.max(rto, 1e-6) / 100) / Math.log(M2.drop))));  // 경매 유찰 횟수
      const pA = A.plattK ? sigmoid(A.plattK.a + A.plattK.b * logit(pG) + A.plattK.k[kA])   // 경매 재보정 + 유찰 횟수
                          : sigmoid(A.platt[0] + A.platt[1] * logit(pG));                   // 경매 재보정만
      const xq = vectorize(Q, fk, prpt);
      let P = QS.map((q) => treeSum(Q.trees[q], xq)).sort((a, b) => a - b);
      P = [P[0] - d[0], P[1], P[2] + d[1]].sort((a, b) => a - b);               // CQR (롤링 예측으로 정한 폭)
      P = P.map((v) => v + s);                                                  // 공매→경매 용도군 이동폭
      P = [P[0] - A.widen[0], P[1], P[2] + A.widen[1]].sort((a, b) => a - b);   // 경매 구간 보정
      out.push({ round: k + 1, rto, auctionFails: kA, pGongmae: pG, p: pA, P });
    }
    return out;
  }
  /** 회차별 몬테카를로 — 팔리면 낙찰가 = 최저가 × exp(배수 추출), 안 팔리면 최저가 × 0.7 로 다음 회차 */
  function simulateRounds(rounds, n, seed) {
    const rng = mulberry32(seed), sold = [], rnd = [];
    for (let i = 0; i < n; i++) {
      for (const r of rounds) {
        if (rng() < r.p) {
          const [a, b, c] = r.P, u = rng();
          const v = interp(u, [0, 0.1, 0.5, 0.9, 1], [a - 0.5 * (b - a), a, b, c, c + 0.5 * (c - b)]);
          sold.push(r.rto / 100 * Math.exp(v)); rnd.push(r.round);
          break;
        }
      }
    }
    const s = Float64Array.from(sold).sort();
    const byRound = rounds.map((r) => rnd.filter((x) => x === r.round).length / n);
    return { p: s.length ? [percentile(s, 0.1), percentile(s, 0.5), percentile(s, 0.9)] : [NaN, NaN, NaN],
      expRound: rnd.length ? rnd.reduce((a, b) => a + b, 0) / rnd.length : NaN, unsold: 1 - s.length / n, byRound, n };
  }
  /** 화면용: 입력 → 최종 낙찰가율 P10/P50/P90 (팔린다면) + 회차표 */
  function predictAuction2(M2, rone, p, n, seed) {
    const { f, market, recent } = buildFeatures2(p, rone, M2);
    const apsl = num(p.apsl), mp = num(p.minPrice), k = Math.max(0, Math.trunc(num(p.fails) || 0));
    const startRto = mp > 0 && apsl > 0 ? (mp / apsl) * 100 : 100 * Math.pow(M2.drop, k);
    const rounds = roundTable(M2, f, startRto);
    const mc = simulateRounds(rounds, n || M2.nSim, seed || 2026);
    return { p: mc.p.map((v) => Math.max(0, v)), rounds, expRound: mc.expRound, unsold: mc.unsold, byRound: mc.byRound,
      startRto, market, recent, features: f, minPriceGiven: mp > 0 };
  }

  const api = { predictAuction, predictAuction2, buildFeatures2, roundTable, simulateRounds, marketRates2, recentStats, roundHalfEven, buildFeatures, marketRates, regionOf, buildCase, resolveAuctionClaims, normName, mergeBooks, splitOnce, waterfall, simulate, summarize,
    breakeven, table, sinboOf, mulberry32, sampleRatios, percentile, SIDO_GRP };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.Engine = api;
})(typeof window !== "undefined" ? window : globalThis);
