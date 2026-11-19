/* 경매 문서(PDF·붙여넣은 글) → 시뮬레이터 입력값
 * ① 옥션원 '법원경매정보' 인쇄 PDF: 표 위치까지 써서 정밀하게 읽음 (parseAuction1)
 * ② 그 밖의 양식: 내용으로 알아봄 — "날짜 + 등기목적 + 금액" 줄은 권리, "감정가·매각가·소재지…" 옆 값은 물건 정보 (parseGeneric)
 * 여러 문서를 올리면 mergeResults 로 합침. 계산 규칙은 노트북(dividend_mc_colab) 그대로라 채권계산서 신고액은 메모로만 남김.
 * 입력: pdf.js 텍스트 조각 [{p 쪽, x, y, w 너비, s 글자}]
 * 출력: { fields: {...}, rights: [...], tenantsNone, warnings: [...], found: [...] }
 * 옥션원 인쇄 PDF는 글자 정보가 들어 있어서 이미지 OCR 없이 정확히 읽을 수 있다.
 */
(function (root) {
  "use strict";

  /** 같은 높이(y)의 조각을 한 줄로. 조각 사이 간격이 있으면 띄어쓰기 */
  function toLines(items) {
    const sorted = items.map((t) => ({ ...t, s: String(t.s || "").replace(/[\u0000-\u001f\u00a0\u200b]/g, " ") }))
      .filter((t) => t.s.trim()).sort((a, b) => a.p - b.p || b.y - a.y || a.x - b.x);
    const lines = [];
    for (const t of sorted) {
      const L = lines[lines.length - 1];
      if (L && L.p === t.p && Math.abs(L.y - t.y) < 3) L.items.push(t);
      else lines.push({ p: t.p, y: t.y, items: [t] });
    }
    for (const L of lines) {
      L.items.sort((a, b) => a.x - b.x);
      let s = "", prevEnd = null;
      for (const t of L.items) {
        if (prevEnd !== null && t.x - prevEnd > 2.2) s += " ";
        s += t.s; prevEnd = t.x + (t.w || 0);
      }
      L.text = s.replace(/\s+/g, " ").trim();
      L.flat = L.text.replace(/\s+/g, "");
    }
    return lines;
  }

  const num = (s) => (s === undefined || s === null ? null : Number(String(s).replace(/,/g, "")));
  const ymd = (y, m, d) => `${y}-${String(m).padStart(2, "0")}-${String(d).padStart(2, "0")}`;

  function usageOf(kind, landOnly) {
    const k = kind || "";
    if (/아파트/.test(k)) return "아파트";
    if (/공장|창고/.test(k)) return "공장·창고";
    if (/다세대|연립|단독|다가구|주택|빌라|도시형/.test(k)) return "기타주거";
    if (/근린|상가|오피스텔|사무|업무|숙박|판매|의료|교육|목욕|주유소|자동차/.test(k)) return "상가·업무";
    if (/전$|답$|^전|^답|임야|과수원|목장|농지/.test(k)) return "농지·임야";
    if (/대지|잡종지|공장용지|창고용지|주차장|도로|학교용지|종교용지/.test(k)) return landOnly ? "대지" : "상가·업무";
    return "";
  }

  function whoOf(name) {
    const n = name || "";
    if (/신용보증기금/.test(n)) return "신보";
    if (/세무서|국세청|^국\(|국\(/.test(n)) return "세무서";
    if (/근로복지공단/.test(n)) return "근로복지공단";
    if (/건강보험|국민연금|고용노동|보험공단/.test(n)) return "건강보험공단";
    if (/은행|뱅크|금고|농협|수협|신협|축협|캐피탈|저축|보험|증권|카드|유동화|금융|자산관리|파이낸스|에이엠씨|AMC|대부|보증재단|진흥공단/.test(n)) return "은행·금융기관";
    if (/(시|군|구|도)청|시장|군수|구청장|도지사|특별시|광역시|자치|읍장|면장/.test(n)) return "지자체";
    // '성주군', '경주시', '수성구', '경상북도' 처럼 행정구역 이름만 있는 권리자 = 지자체 (지방세 압류)
    if (/^[가-힣]{1,10}(시|군|구|도)(\(.*\))?$/.test(n.replace(/\s/g, ""))) return "지자체";
    return "개인·기타 법인";
  }

  function typeOf(kind) {
    const k = kind || "";
    if (/근저당|저당/.test(k)) return "근저당권설정";
    if (/전세권/.test(k)) return "전세권설정";
    if (/임차권/.test(k)) return "임차권등기";
    if (/가압류/.test(k)) return "가압류";
    if (/압류/.test(k)) return "압류";
    if (/경매/.test(k)) return "경매개시결정";
    if (/소유권/.test(k)) return "소유권이전";
    return "기타";
  }

  const ROW = /^(\d+)\((갑|을)(\d+)\)\s*(\d{4})\.(\d{2})\.(\d{2})/;

  /** 등기부 표: 머리글 위치로 열을 나누고, 행 번호 줄 기준 위아래 줄을 묶음 */
  function parseRegisters(lines) {
    const rows = [];
    let cols = null;
    const headers = lines.filter((L) => /^No접수권리종류권리자채권금액/.test(L.flat));
    if (!headers.length) return rows;
    const colsOf = (L) => {
      const want = ["No", "접수", "권리종류", "권리자", "채권금액", "비고", "소멸여부"];
      const c = want.map((w) => { const it = L.items.find((t) => t.s.replace(/\s/g, "") === w); return it ? it.x + (it.w || 0) / 2 : null; });
      return c.every((v) => v !== null) ? c : null;
    };
    const anchors = [], stops = [];
    let inReg = false, section = 0, curBook = "";
    const secBook = {};                        // 표 번호 → "토지" / "건물" (집합건물은 건물)
    const STOP = /^(No접수권리종류|주의사항|자산관리|건축물정보|매각사례|임차인현황|☞|(건물|토지|집합건물)등기부|https?:|\d{2}\.\d{1,2}\.\d{1,2}\.오[전후])/;
    lines.forEach((L, i) => {
      const bm = L.flat.match(/^(건물|토지|집합건물)등기부/);
      if (bm) curBook = bm[1] === "토지" ? "토지" : "건물";
      if (/^(건물|토지|집합건물)?등기부/.test(L.flat) || /^(건물|토지)등기부/.test(L.flat)) inReg = true;
      if (/^No접수권리종류권리자채권금액/.test(L.flat)) { cols = colsOf(L) || cols; inReg = true; section++; secBook[section] = curBook; return; }
      if (/^(주의사항|자산관리|건축물정보|매각사례|임차인현황)/.test(L.flat)) inReg = false;
      if (inReg && cols && ROW.test(L.text.replace(/\s+/g, ""))) anchors.push({ i, L, cols, section });
    });
    lines.forEach((L) => { if (STOP.test(L.flat)) stops.push(L); });
    const colOf = (t, c) => {
      const cx = t.x + (t.w || 0) / 2;
      let best = 0;
      for (let k = 1; k < c.length; k++) if (cx >= (c[k - 1] + c[k]) / 2) best = k;
      return best;
    };
    anchors.forEach((a, ai) => {
      const prev = anchors[ai - 1], next = anchors[ai + 1];
      const up = prev && prev.L.p === a.L.p ? (prev.L.y - a.L.y) / 2 : 22;
      const down = next && next.L.p === a.L.p ? (a.L.y - next.L.y) / 2 : 22;
      // 표 밖(머리글·다음 섹션) 줄은 넣지 않음
      const above = stops.filter((L) => L.p === a.L.p && L.y > a.L.y).map((L) => L.y).sort((x, y) => x - y)[0] ?? Infinity;
      const below = stops.filter((L) => L.p === a.L.p && L.y < a.L.y).map((L) => L.y).sort((x, y) => y - x)[0] ?? -Infinity;
      const band = lines.filter((L) => L.p === a.L.p && L.y <= a.L.y + Math.min(up, 22) && L.y >= a.L.y - Math.min(down, 22) && L.y < above && L.y > below);
      const cell = ["", "", "", "", "", "", ""];
      band.sort((x, y) => y.y - x.y).forEach((L) => L.items.forEach((t) => { const k = colOf(t, a.cols); cell[k] += (cell[k] ? " " : "") + t.s; }));
      const clean = (s) => s.replace(/\s+/g, " ").trim();
      const m = a.L.text.replace(/\s+/g, "").match(ROW);
      const kind = clean(cell[2]).replace(/\s/g, "");
      const holder = clean(cell[3]).replace(/\s*\(\s*/g, "(").replace(/\s*\)\s*/g, ")").replace(/\)(?=[^\s(])/g, ")").trim();
      const amtM = clean(cell[4]).replace(/\s/g, "").match(/([\d,]{4,})원/);
      const note = clean(cell[5]);
      const type = typeOf(kind);
      // 경매개시결정 줄의 금액 = 청구금액 → 금액 칸에 넣음 (계산 엔진이 신청 채권자의 근저당·가압류에 씀)
      const amount = amtM ? num(amtM[1]) : null;
      rows.push({ no: +m[1], gu: m[2] + "구", rank: m[2] + m[3], section: a.section, book: secBook[a.section] || "", date: ymd(m[4], m[5], m[6]), kind, type, name: holder, who: whoOf(holder),
        amount, claimAmount: type === "경매개시결정" ? amount : null,
        memo: [kind !== type ? kind : "", type === "경매개시결정" && amount ? `청구금액 ${amount.toLocaleString("ko-KR")}` : "", note.slice(0, 60)].filter(Boolean).join(" · ") });
    });
    // 근저당 '일부이전·이전'은 새 권리가 아니라 원래 근저당의 순위를 따름 → 원래 접수일로 바꿈
    // 원래 근저당 찾기: 같은 등기부의 같은 을구 번호 → 없으면 등기목적에 적힌 원래 권리자 이름 (예: '경남은행근저당권일부이전')
    const nn = (s) => String(s || "").replace(/\(주\)|㈜|주식회사/g, "").replace(/\(.*?\)/g, "").replace(/\s/g, "");
    rows.forEach((r) => {
      if (!/이전/.test(r.kind) || r.type !== "근저당권설정") return;
      const base = rows.filter((o) => o !== r && o.type === "근저당권설정" && !/이전/.test(o.kind));
      const k = nn(r.kind);
      // 비고·등기목적의 '을구 N번'/'(을N)' → 그 순위의 근저당
      const ref = [...(r.kind + " " + r.memo).replace(/\s/g, "").matchAll(/을구?(\d+)번?/g)].map((m) => "을" + m[1]);
      const orig = base.find((o) => o.section === r.section && o.rank === r.rank && (!nn(o.name) || k.includes(nn(o.name))))
        || base.find((o) => ref.includes(o.rank) && (!nn(o.name) || k.includes(nn(o.name)) || !base.some((b) => nn(b.name) && k.includes(nn(b.name)))))
        || base.find((o) => nn(o.name) && k.includes(nn(o.name))) || base.find((o) => o.rank === r.rank);
      if (orig) { r.memo = `${r.memo} · 원래 근저당(${orig.name} ${orig.date}) 순위를 따름`.replace(/^ · /, ""); r.transferOf = orig.date; r.date = orig.date; }
    });
    // 건물·토지 등기부에 같은 권리가 한 번씩 나옴 → 권리자·권리종류·금액이 같으면 한 건 (등기부마다 i번째끼리 짝지음,
    // 한 등기부 안에서 같은 권리자의 같은 권리가 여러 건이면 그대로 둠). 공동담보는 등기부별 접수일·순위번호가 다를 수 있어
    // 날짜는 키에 안 넣고, 등기부별 위치를 books 에 남김 → 계산 엔진이 등기부별 배당(민법 368조)에 씀
    const keyOf = (r) => [r.type, r.who === "건강보험공단" || r.who === "근로복지공단" ? r.who : nn(r.name), r.amount].join("|");
    const groups = new Map();
    rows.forEach((r) => {
      const k = keyOf(r); if (!groups.has(k)) groups.set(k, new Map());
      const g = groups.get(k); if (!g.has(r.section)) g.set(r.section, []); g.get(r.section).push(r);
    });
    const rankDigits = (r) => (/^을/.test(r.rank) ? r.rank.replace(/\D/g, "") : null);
    const out = [];
    groups.forEach((bySec) => {
      const lists = [...bySec.values()], n = Math.max(...lists.map((l) => l.length));
      for (let i = 0; i < n; i++) {
        const same = lists.map((l) => l[i]).filter(Boolean), r = same[0];
        const books = {};
        same.forEach((x) => { if (x.book && !books[x.book]) books[x.book] = { date: x.date, rank: rankDigits(x) }; });
        if (Object.keys(books).length) r.books = books;
        const ds = [...new Set(same.map((x) => x.date))].sort();
        if (ds.length > 1) { r.memo = `${r.memo} · 등기부별 접수일 ${Object.entries(books).map(([b, p]) => `${b} ${p.date}`).join(" / ")}`.replace(/^ · /, ""); r.date = ds[0]; r.dateVaries = ds; }
        out.push(r);
      }
    });
    return out.sort((a, b) => (a.date < b.date ? -1 : a.date > b.date ? 1 : 0));
  }

  function matchCategory(text, list) {
    if (!text || !list) return "";
    const t = text.replace(/\s/g, "");
    let best = "";
    for (const c of list) { const cc = c.replace(/\s/g, ""); if (cc && t.includes(cc) && cc.length > best.length) best = c; }
    return best;
  }

  /** 경매 진행 상황 → 현재 최저가 · 지금까지 유찰 횟수 (모델 입력)
   *  ① 기일 내역 줄 "(신건|N차) 날짜 최저가 원 결과": 유찰 줄 수 = 유찰 횟수, 마지막 회차의 최저가 = 현재 최저가
   *     (이미 매각된 문서면 매각된 회차의 최저가와 그 앞 유찰 횟수 → 테스트셋과 같은 기준)
   *  ② '최저가(49%) 414,540,000원' · '최저매각가격' 표시 → 최저가
   *  ③ '유찰 2회' 표시, 없으면 최저가 ÷ 감정가 = 0.7^k 로 계산 (대구지법 30% 기준, 딱 맞을 때만)
   *  texts: 줄 글자 배열. 반환 {minPrice, fails, failsFrom, rounds, warnings} */
  const RESULT_KW = /(유찰|매각|변경|취하|기각|연기|정지|불허|미진행|진행|낙찰|입찰)/;
  function parseProgress(texts, apsl) {
    const warnings = [], rounds = [];
    // 기일 행 = "날짜 바로 뒤에 최저가 금액" (+ 결과). 옥션원은 기일 표가 왼쪽 물건 표와 같은 높이라 한 줄에 섞여 읽히므로
    // 줄 전체가 아니라 날짜 바로 뒤 금액만 봄 (등기부 줄은 날짜 뒤에 권리종류가 와서 걸리지 않음)
    const ROUND_G = /(?:(신건|\d+\s*차)\s*)?(\d{4})\s*[.\-\/]\s*(\d{1,2})\s*[.\-\/]\s*(\d{1,2})\.?\s*(?:\([^)\d]{0,3}\)\s*)?([\d]{1,3}(?:,\d{3}){2,})\s*원\s*(유찰|매각|변경|취하|기각|연기|정지|불허|미진행|진행|낙찰)?/g;
    const loose = [];                          // 결과 칸이 빈 줄 (다음 기일) — 기일 내역 뒤에 이어질 때만 씀
    texts.forEach((t) => {
      for (const x of t.matchAll(ROUND_G)) {
        const min = num(x[5]); if (!(min >= 1e6)) continue;
        const res = x[6] || "";
        (res ? rounds : loose).push({ date: ymd(x[2], x[3], x[4]), min, result: res, label: x[1] || "", text: t });
      }
    });
    if (rounds.length) {
      const lastD = rounds.map((r) => r.date).sort().pop(), lastMin = Math.min(...rounds.map((r) => r.min));
      loose.filter((r) => r.date > lastD && r.min <= lastMin).forEach((r) => rounds.push(r));
    }
    // 같은 날짜가 두 번 읽히면(표 머리·본문) 한 번만
    const seen = new Set(), R = rounds.filter((r) => { const k = r.date + "|" + r.min; if (seen.has(k)) return false; seen.add(k); return true; })
      .sort((a, b) => (a.date < b.date ? -1 : a.date > b.date ? 1 : 0));
    const flat = texts.join("\n").replace(/[ \t]/g, "");
    let m = flat.match(/최저(?:매각)?(?:가격|가)(?:\(\d+(?:\.\d+)?%\))?[:：]?([\d,]{7,})원/);
    const labelMin = m ? num(m[1]) : null;
    m = flat.match(/유찰\(?(\d+)회\)?/);
    const labelFails = m ? +m[1] : null;

    let minPrice = null, fails = null, failsFrom = "";
    const live = R.filter((r) => !/변경|취하|기각|연기|정지|미진행/.test(r.result));
    if (live.length) {
      const last = live[live.length - 1];
      minPrice = last.min;
      fails = live.filter((r) => r !== last && /유찰/.test(r.result)).length;
      failsFrom = "기일 내역";
      if (labelMin && Math.abs(labelMin - minPrice) / labelMin > 0.01) warnings.push(`기일 내역의 마지막 최저가(${minPrice.toLocaleString()}원)와 '최저가' 표시(${labelMin.toLocaleString()}원)가 달라요. 1단계에서 현재 최저가를 확인하세요.`);
    } else if (labelMin) {
      minPrice = labelMin;
    }
    if (fails === null && labelFails !== null) { fails = labelFails; failsFrom = "'유찰 N회' 표시"; }
    if (fails === null && minPrice && apsl) {
      const k = Math.round(Math.log(minPrice / apsl) / Math.log(0.7));
      if (k >= 0 && Math.abs(apsl * Math.pow(0.7, k) - minPrice) / minPrice < 0.015) { fails = k; failsFrom = "최저가 ÷ 감정가 (0.7배씩)"; }
      else warnings.push(`최저가가 감정가의 ${(minPrice / apsl * 100).toFixed(0)}%인데 유찰 횟수를 알아내지 못했어요. 1단계에서 유찰 횟수를 넣어주세요.`);
    }
    if (minPrice && apsl && minPrice > apsl * 1.001) { warnings.push(`읽은 최저가(${minPrice.toLocaleString()}원)가 감정가보다 커요. 넣지 않았어요.`); minPrice = null; }
    if (!minPrice && /가림\(최저가\)|기일·유찰이력/.test(flat)) warnings.push("이 문서는 최저가와 기일·유찰 이력이 가려진 예측용 문서예요. 1단계에서 그 회차의 최저가나 유찰 횟수를 직접 넣어야 검증된 기준(진행 중)으로 계산돼요. 비워 두면 경매 시작 전 기준(검증 안 됨)으로 계산해요.");
    else if (!minPrice) warnings.push("현재 최저가를 찾지 못했어요. 1단계에서 현재 최저가나 유찰 횟수를 넣지 않으면 경매 시작 전 기준(검증 안 됨)으로 계산해요.");
    return { minPrice, fails, failsFrom, rounds: R, warnings };
  }

  /** cats: {bld_strct: [...], bld_purps: [...]} — 모델 범주 이름과 맞춤 */
  function parseAuction1(items, cats) {
    const lines = toLines(items);
    const all = lines.map((L) => L.text).join("\n");
    const flat = lines.map((L) => L.flat).join("\n");
    const F = {}, found = [], warnings = [];
    const set = (k, v, label) => { if (v !== null && v !== undefined && v !== "" && !Number.isNaN(v)) { F[k] = v; found.push(label); } };

    if (!/옥션원|auction1/.test(all)) warnings.push("옥션원 인쇄 양식이 아닌 것 같아요. 읽은 값을 꼭 확인하세요.");
    let m = flat.match(/(\d{4})타경(\d+)/);
    const caseNo = m ? `${m[1]}타경${m[2]}` : "";
    m = flat.match(/([가-힣]+지방법원)([가-힣]*(?:본원|지원))?/);
    const court = m ? [m[1], m[2]].filter(Boolean).join(" ") : "";
    const locL = lines.find((L) => /^소재지/.test(L.flat));
    const addr = locL ? locL.text.replace(/^소\s*재\s*지\s*/, "").trim() : "";
    const kindL = lines.find((L) => /^물건종별/.test(L.flat));
    const kindRaw = kindL ? (kindL.flat.match(/^물건종별([가-힣·/]+?)감정가/) || [])[1] || "" : "";

    m = flat.match(/합계([\d,]+)원/) || flat.match(/^([\d,]{7,})원$/m);
    const apsl = m ? num(m[1]) : null;
    m = flat.match(/매각기일:(\d{4})\.(\d{2})\.(\d{2})/);
    const saleDate = m ? ymd(m[1], m[2], m[3]) : null;
    m = flat.match(/매각:([\d,]+)원/);
    const sale = m ? num(m[1]) : null;
    // 토지 감정가 (등기부별 배당에 씀): 토지 표의 소계(건물 줄 전) → 토지 줄 하나면 그 금액 → 없으면 '감정가 대비 토지 N%'
    let landApsl = null;
    const t0 = lines.findIndex((L) => /^목록지번/.test(L.flat));
    for (let k = t0 + 1; t0 >= 0 && k < lines.length; k++) {
      const t = lines[k].flat;
      if (/^(건물|현황|제시외|기계)/.test(t)) break;
      const mm = t.match(/^면적소계.*소계([\d,]+)원/); if (mm) { landApsl = num(mm[1]); break; }
      const m1 = /^토지/.test(t) && lines[k].text.match(/(?:^|\s)(\d{1,3}(?:,\d{3}){2,})원\s*$/); if (m1) landApsl = (landApsl || 0) + num(m1[1]);
    }
    m = flat.match(/감정가대비토지([\d.]+)%/);
    if (!(landApsl > 0) && m && apsl) landApsl = Math.round(apsl * Number(m[1]) / 100);
    if (!(landApsl > 0 && apsl && landApsl < apsl)) landApsl = null;
    // '토지면적 전체: 9386㎡ / 지분: 9336㎡' (지분 매각)이면 매각 대상인 지분 면적
    m = (/토지면적전체:/.test(flat) && flat.match(/지분:([\d.,]+)㎡/)) || flat.match(/토지면적(?:전체:)?([\d.,]+)㎡/) || flat.match(/대지권([\d.,]+)㎡/);
    const land = m ? num(m[1]) : null;
    if (/토지면적전체:/.test(flat) && /지분:/.test(flat)) warnings.push("토지 지분 매각이에요. 토지면적은 지분 면적으로 넣었어요.");
    m = flat.match(/건물면적([\d.,]+)㎡/);
    const bld = m ? num(m[1]) : null;
    // 공시지가: 토지 표의 '(255,600원)' — 여러 필지면 면적으로 가중평균
    const lpVals = [...flat.matchAll(/\(([\d,]+)원\)/g)].map((x) => num(x[1])).filter((v) => v > 0);
    const lp = lpVals.length ? lpVals[0] : null;
    if (lpVals.length > 1 && new Set(lpVals).size > 1) warnings.push(`필지가 여러 개예요 (공시지가 ${lpVals.length}개). 첫 필지 값(${lpVals[0].toLocaleString()}원/㎡)을 넣었어요.`);
    m = flat.match(/사용승인일(\d{4})\.(\d{1,2})\.(\d{1,2})/) || flat.match(/사용승인:(\d{4})\.(\d{1,2})\.\n?(\d{1,2})/);
    const approve = m ? ymd(m[1], m[2], m[3]) : null;
    m = flat.match(/층수지상(\d+)층/) || flat.match(/지상(\d+)층/);
    const flr = m ? +m[1] : null;
    m = flat.match(/연면적([\d.,]+)㎡/);
    const tot = m ? num(m[1]) : null;
    // 주건축물 중 면적이 가장 큰 동의 구조, 건축물정보의 용도
    const mains = lines.filter((L) => /^(주건축물|전유\/주)/.test(L.flat)).map((L) => ({ L, area: num((L.flat.match(/([\d.,]+)㎡$/) || [])[1]) || 0 }));
    mains.sort((a, b) => b.area - a.area);
    const strct = mains.length ? matchCategory(mains[0].L.text, cats && cats.bld_strct) : "";
    m = flat.match(/용도([가-힣0-9]+?)사용승인일/);
    const purps = matchCategory(m ? m[1] : mains.length ? mains[0].L.text : "", cats && cats.bld_purps);
    m = flat.match(/말소기준권리:(\d{4})\.(\d{2})\.(\d{2})/);
    const baseDate = m ? ymd(m[1], m[2], m[3]) : null;
    const tenantsNone = /조사된임차(내역|인)없음/.test(flat);
    const tenants = tenantsNone ? [] : parseTenantTable(lines);
    const hasTenantTable = !tenantsNone && /임차인현황/.test(flat) && !tenants.length;

    const parts = addr.split(/\s+/);
    const sido = parts[0] || "";
    let sgg = parts[1] || "";
    if (parts[1] && /시$/.test(parts[1]) && parts[2] && /구$/.test(parts[2])) sgg = `${parts[1]} ${parts[2]}`;
    const landOnly = !(bld > 0);
    const usage6 = usageOf(kindRaw, landOnly);

    set("name", [court.replace("지방법원", "지법"), caseNo, kindRaw].filter(Boolean).join(" "), "사건");
    set("apsl", apsl, "감정가"); set("usage6", usage6, `용도 (${kindRaw || "?"})`);
    set("month", saleDate ? saleDate.slice(0, 7) : null, "매각기일"); set("sale", sale, "매각가");
    const prog = parseProgress(lines.map((L) => L.text), apsl);
    set("minPrice", prog.minPrice, "현재 최저가"); set("fails", prog.fails, `유찰 횟수 (${prog.failsFrom})`);
    warnings.push(...prog.warnings);
    set("sido", sido, "시도"); set("sgg", sgg, "시군구"); set("land", land, "토지면적"); set("lp", lp, "공시지가");
    if (!landOnly) {
      F.hasBld = true;
      set("bld", bld, "건물면적"); set("approve", approve, "사용승인일"); set("flr", flr, "지상층수"); set("tot", tot, "연면적");
      set("strct", strct, "구조"); set("purps", purps, "주용도"); set("landApsl", landApsl, "토지 감정가");
    }
    set("baseDate", baseDate, "말소기준권리");
    if (!usage6) warnings.push(`물건종별 '${kindRaw}'을 용도에 맞추지 못했어요. 1단계에서 용도를 골라주세요.`);
    if (!apsl) warnings.push("감정가를 찾지 못했어요.");

    const rights = parseRegisters(lines);
    if (!rights.length) warnings.push("등기부 표를 찾지 못했어요. 2단계에서 직접 넣어주세요.");
    const sb = rights.filter((r) => r.who === "신보" && ["근저당권설정", "가압류", "전세권설정"].includes(r.type));
    if (sb.length > 1) warnings.push(`신보 권리가 ${sb.length}건(${sb.map((r) => r.kind).join("·")}) 있어요. 같은 구상채권이 근저당과 가압류로 겹쳐 잡혔을 수 있으니, 4단계에서 신보 채권액을 확인하세요.`);
    const forced = rights.filter((r) => r.type === "경매개시결정" && /강제/.test(r.kind));
    if (forced.length) warnings.push("강제경매 사건이에요. 신청 채권자의 가압류가 1건이면 그 가압류 금액 대신 청구금액으로 계산해요. 가압류가 없는 일반채권자라면 청구금액을 '가압류' 줄로 넣으세요.");
    if (rights.some((r) => r.type === "압류" && r.amount === null && (r.who === "세무서" || r.who === "지자체"))) warnings.push("세금 압류 금액이 없어요. 숨은 선순위일 수 있어 5단계 체납세금 시나리오로 계산해요 (지자체 압류는 당해세일 수 있음).");
    if (hasTenantTable) warnings.push("임차인 표를 읽지 못했어요. 3단계에서 보증금·전입일·확정일자·배당요구를 직접 넣어주세요.");
    tenants.filter((t) => t.claimFromDoc).forEach((t) => warnings.push(`임차인 ${t.name}: '배당요구: 없음'이지만 문건처리내역상 배당요구일이 있어 '배당요구 함'으로 넣었어요.`));
    if (tenants.length) { found.push(`임차인 ${tenants.length}명`); warnings.push(`임차인 ${tenants.length}명을 읽었어요. 보증금·월세·전입일·확정일자·배당요구가 맞는지 3단계에서 확인하세요.`); }
    rights.filter((r) => r.dateVaries && r.type === "근저당권설정").slice(0, 1).forEach((r) => warnings.push(`토지·건물 등기부의 근저당 접수일이 달라요 (${r.dateVaries.join(" / ")}). 공동담보로 보고 한 건으로 합쳐 이른 날짜로 넣었어요. 그 사이에 등기된 다른 권리가 있으면 순위를 확인하세요.`));
    rights.filter((r) => r.transferOf).forEach((r) => warnings.push(`'${r.kind}'(${r.name})은 새 근저당이 아니라 원래 근저당 순위(${r.date})로 넣었어요. 원래 근저당 최고액에 이 금액이 포함돼 있으면 겹칠 수 있어요.`));
    if (sale) found.push("(실제 매각가는 비교용)");
    return { fields: F, rights, tenants, tenantsNone, warnings, found, source: "옥션원", meta: { caseNo, court, addr, kindRaw, saleDate } };
  }

  /** 옥션원 임차인현황 표: 이름 · 전입(사업등록) · 확정일자 · 배당요구 · 보증금 · 월세
   *  임차인마다 '전입일자:/사업등록:' 줄에서 시작하는 묶음으로 나눠 읽음 (금액·날짜가 다른 임차인 것과 섞이지 않게) */
  function parseTenantTable(lines) {
    const i0 = lines.findIndex((L) => /^임차인현황/.test(L.flat));
    if (i0 < 0) return [];
    const sec = [];
    for (let i = i0 + 1; i < lines.length; i++) {
      const L = lines[i];
      if (/^(기타사항|임차인수|\*|☞|등기부현황|건물등기부|토지등기부|건축물정보|https?:)/.test(L.flat) || /등기부\(/.test(L.flat)) break;
      sec.push(L);
    }
    const ANCHOR = /(전입|사업등록|사업자등록)(?:일자|일)?[:：]/;
    const blocks = [];
    sec.forEach((L) => { if (ANCHOR.test(L.flat)) blocks.push([L]); else if (blocks.length) blocks[blocks.length - 1].push(L); });
    const d = (m, k = 1) => (m && m[k] ? ymd(m[k], m[k + 1], m[k + 2]) : null);
    return blocks.map((B, i) => {
      const f = B.map((L) => L.flat).join("\n"), all = f.replace(/\n/g, "");
      const mv = f.match(/(전입|사업등록|사업자등록)(?:일자|일)?[:：](?:(\d{4})\.(\d{1,2})\.(\d{1,2}))?/);
      const conf = d(f.match(/확정일자[:：](\d{4})\.(\d{1,2})\.(\d{1,2})/));
      const claimDate = d(f.match(/배당요구[:：](\d{4})\.(\d{1,2})\.(\d{1,2})/));
      // '배당요구: 없음'이어도 문건처리내역상 배당요구일이 있으면 배당요구를 한 것
      // ('문건처리내역상 배당' / '요구일: 2024.07.25' 처럼 칸 안에서 줄이 나뉘어도 읽음)
      const docClaim = /문건처리내역/.test(all) ? d(f.match(/요구일[:：](\d{4})\.(\d{1,2})\.(\d{1,2})/)) : null;
      const dep = f.match(/보([\d,]+)원/), rent = f.match(/월([\d,]+)원/);
      const nameL = B.find((L) => /확정일자/.test(L.flat)) || B[0];
      const name = (nameL.text.match(/^([가-힣A-Za-z()㈜○●*]+)(?=\s)/) || [])[1] || "";
      return { no: i + 1, name: name && !/^(전입|사업|확정|배당)/.test(name) ? name : `임차인${i + 1}`, kind: /사업/.test(mv[1]) ? "상가" : "주거",
        moved: d(mv, 2), confirmed: conf, claim: claimDate || docClaim ? "Y" : "N", claimFromDoc: !claimDate && !!docClaim,
        deposit: dep ? num(dep[1]) : null, rent: rent ? num(rent[1]) : 0 };
    });
  }

  /* ───────── 양식 자유 문서 ───────── */
  const DATE = /(\d{4})\s*[.\-\/년]\s*(\d{1,2})\s*[.\-\/월]\s*(\d{1,2})\s*일?/;
  const AMT = /([\d]{1,3}(?:,\d{3})+|\d{5,})\s*원/;
  const RIGHT_KW = /(근저당권?(?:설정|이전|일부이전)?|저당권|전세권(?:설정)?|임차권(?:등기)?|가압류|(?:[가-힣]*지분)?압류|(?:임의|강제)?경매(?:개시)?(?:결정)?|소유권(?:이전|보존)?)/;
  const CLAIM_KW = /(채권계산서|배당요구|교부청구|권리신고|채권신고)/;
  const dateOf = (t) => { const m = String(t || "").match(DATE); return m ? ymd(m[1], m[2], m[3]) : null; };
  const amtOf = (t) => { const all = [...String(t || "").matchAll(new RegExp(AMT.source, "g"))]; return all.length ? num(all[all.length - 1][1]) : null; };

  /** 줄 목록 → 입력값. 표 머리글이나 칸 위치에 기대지 않고 줄마다 내용으로 판단 */
  function parseGeneric(lines, cats) {
    const F = {}, found = [], warnings = [], rights = [], claims = [], tenants = [];
    const set = (k, v, label) => { if (v !== null && v !== undefined && v !== "" && !Number.isNaN(v) && F[k] === undefined) { F[k] = v; found.push(label); } };
    const texts = lines.map((L) => (typeof L === "string" ? L : L.text)).map((t) => t.replace(/\s+/g, " ").trim()).filter(Boolean);
    const all = texts.join("\n"), flat = all.replace(/[ \t]/g, "");
    let m;
    // 사건
    m = flat.match(/(\d{4})타경(\d+)/);
    const caseNo = m ? `${m[1]}타경${m[2]}` : ((all.match(/사건\s*번호\s*[:：]?\s*([^\n(]+)/) || [])[1] || "").trim();
    m = flat.match(/([가-힣]+지방법원)([가-힣]*(?:본원|지원))?/);
    const court = m ? [m[1], m[2]].filter(Boolean).join(" ") : "";
    const kv = (re) => { for (const t of texts) { const x = t.match(re); if (x) return { t, x }; } return null; };
    let r = kv(/^(?:물건\s*종[류별]|물건\s*용도|용도)\s*[:：]?\s*([가-힣·/()]+)/);
    const kindRaw = r ? r.x[1].replace(/\(.*\)/, "") : "";
    r = kv(/^소\s*재\s*지\s*[:：]?\s*(.+)$/);
    const addr = r ? r.x[1].replace(/\(가상\)/g, "").trim() : "";
    r = kv(/감\s*정\s*(?:평\s*가\s*)?(?:가|액|금액)(?:\s*\(.*?\))?\s*[:：]?\s*([\d,]{5,})\s*원/);
    const apsl = r ? num(r.x[1]) : null;
    r = kv(/^(?:매각\s*가|매각\s*대금|매각\s*금액|낙찰\s*가|매각)(?:\s*\(.*?\))?\s*[:：]?\s*([\d,]{5,})\s*원/);
    const sale = r ? num(r.x[1]) : null;
    r = kv(/집행\s*비용(?:\s*\(.*?\))?\s*[:：]?\s*([\d,]+)\s*원/);
    const cost = r ? num(r.x[1]) : null;
    r = kv(/(?:토지\s*면적|대지\s*면적|대\s*지\s*권)\s*[:：]?\s*([\d.,]+)\s*(?:㎡|m2|제곱미터)/);
    const land = r ? num(r.x[1]) : null;
    r = kv(/(?:건물\s*면적|전용\s*면적)\s*[:：]?\s*([\d.,]+)\s*(?:㎡|m2|제곱미터)/);
    const bld = r ? num(r.x[1]) : null;
    r = kv(/연\s*면\s*적\s*[:：]?\s*([\d.,]+)\s*(?:㎡|m2)/);
    const tot = r ? num(r.x[1]) : null;
    r = kv(/공시\s*지가[^\d\n]*([\d,]{3,})\s*원/);
    const lp = r ? num(r.x[1]) : null;
    r = kv(/사용\s*승인(?:일)?\s*[:：]?\s*(.+)/);
    const approve = r ? dateOf(r.x[1]) : null;
    r = kv(/매각\s*기일\s*[:：]?\s*(.+)/);
    const saleDate = r ? dateOf(r.x[1]) : null;
    r = kv(/경매\s*개시\s*결정(?:\s*등기)?(?:일)?\s*[:：]?\s*(\d{4}.+)/);
    const auctionDate = r ? dateOf(r.x[1]) : null;
    r = kv(/(?:최선순위\s*(?:설정|담보)?\s*(?:일|일자)?|말소\s*기준\s*(?:권리|등기)?)\s*[:：]?\s*(\d{4}.+)/);
    const baseDate = r ? dateOf(r.x[1]) : null;
    const tenantsNone = /조사된임차(인|내역)(이)?없음|임차인(이)?없음|임차인없음/.test(flat);

    // 권리·신고 줄
    let section = "";
    texts.forEach((t) => {
      if (/^\d+\.\s*/.test(t) || /^(등기부|채권신고|배당요구|임차인|권리)/.test(t.replace(/\s/g, ""))) section = t.replace(/\s/g, "");
      const d = dateOf(t), a = amtOf(t);
      if (CLAIM_KW.test(t) && a !== null && !/^[\d.]+\s/.test(t)) {
        const holder = t.split(/\s+/)[0];
        claims.push({ holder, kind: (t.match(CLAIM_KW) || [])[1], amount: a, date: d, text: t });
        return;
      }
      const k = t.match(RIGHT_KW);
      if (!d || !k || /^(매각|감정|사용승인|경매개시결정\s*등기)/.test(t)) return;
      if (/채권신고|배당요구/.test(section) && !/등기부/.test(section)) return;
      const kind = k[1];
      const type = typeOf(kind);
      const after = t.slice(t.indexOf(kind) + kind.length);
      let holder = after.replace(AMT, "").replace(/(청구\s*금액|채권\s*최고액|금액|최고액)\s*[:：]?/g, "").replace(/[\d,]{4,}/g, "").trim();
      holder = holder.split(/\s{2,}|\s(?=말소|소멸|인수|비고)/)[0].trim();
      const gu = (t.match(/(갑|을)\s*구/) || [])[1];
      rights.push({ gu: gu ? gu + "구" : null, date: d, kind, type, name: holder, who: whoOf(holder),
        amount: type === "경매개시결정" ? null : a, claimAmount: type === "경매개시결정" ? a : null, memo: kind !== type ? kind : "" });
    });
    if (auctionDate && !rights.some((x) => x.type === "경매개시결정")) rights.push({ date: auctionDate, kind: "경매개시결정", type: "경매개시결정", name: "", who: "은행·금융기관", amount: null, memo: "문서의 경매개시결정 등기일" });

    // 채권계산서·배당요구·교부청구: 노트북 규칙은 등기부 금액(근저당은 최고액÷1.2)을 쓰므로 메모로만 붙임
    claims.forEach((c) => {
      const hit = rights.find((x) => x.name && (x.name.includes(c.holder) || c.holder.includes(x.name)) && x.type !== "경매개시결정" && x.type !== "소유권이전");
      if (hit) { hit.memo = [hit.memo, `${c.kind} 신고 ${c.amount.toLocaleString()}원`].filter(Boolean).join(" · "); hit.claimed = c.amount; return; }
      const who = whoOf(c.holder);
      if (["세무서", "지자체"].includes(who) || /세무서|구청|시청|군청/.test(c.holder)) rights.push({ date: c.date, kind: "교부청구", type: "압류", name: c.holder, who: who === "개인·기타 법인" ? "지자체" : who, amount: c.amount, memo: `${c.kind} (등기 없음)` });
      else if (who === "근로복지공단" || /근로자|임금|퇴직/.test(c.text)) rights.push({ date: c.date, kind: c.kind, type: "가압류", name: c.holder, who: "근로복지공단", amount: c.amount, memo: `${c.kind} · 임금 (최우선으로 계산)` });
      else if (who === "건강보험공단") rights.push({ date: c.date, kind: c.kind, type: "압류", name: c.holder, who, amount: c.amount, memo: `${c.kind} (등기 없음)` });
      else warnings.push(`'${c.holder}'의 ${c.kind}(${c.amount.toLocaleString()}원)는 등기부 권리와 짝을 못 찾아 넣지 않았어요. 필요하면 2단계에서 추가하세요.`);
    });
    const withClaim = rights.filter((x) => x.claimed && x.type === "근저당권설정");
    if (withClaim.length) warnings.push(`근저당 채권계산서 신고액이 있어요 (${withClaim.map((x) => `${x.name} ${(x.claimed / 1e8).toFixed(2)}억`).join(", ")}). 계산은 노트북 규칙대로 채권최고액 ÷ 1.2로 해요. 필요하면 5단계 비율을 바꾸세요.`);

    // 임차인 (라벨이 있는 줄)
    texts.forEach((t) => {
      if (!/보증금/.test(t) || !/(전입|사업자?\s*등록)/.test(t)) return;
      const g = (re) => { const x = t.match(re); return x ? x[1] : null; };
      tenants.push({ no: tenants.length + 1, name: g(/^(\S+)/) || "", kind: /사업/.test(t) ? "상가" : "주거",
        deposit: num(g(/보증금\s*[:：]?\s*([\d,]+)/)), rent: num(g(/(?:월세|차임)\s*[:：]?\s*([\d,]+)/)) || 0,
        moved: dateOf(g(/(?:전입|사업자?\s*등록)(?:일)?\s*[:：]?\s*([\d.\-\/년월일 ]{8,14})/)), confirmed: dateOf(g(/확정\s*일자\s*[:：]?\s*([\d.\-\/년월일 ]{8,14})/)),
        claim: /배당\s*요구\s*[:：]?\s*(Y|예|있음|함|\d{4})/.test(t) ? "Y" : "N" });
    });

    const parts = addr.split(/\s+/);
    const SIDO_FULL = { "서울": "서울특별시", "부산": "부산광역시", "대구": "대구광역시", "인천": "인천광역시", "광주": "광주광역시", "대전": "대전광역시", "울산": "울산광역시",
      "세종": "세종특별자치시", "경기": "경기도", "강원": "강원특별자치도", "충북": "충청북도", "충남": "충청남도", "전북": "전북특별자치도", "전남": "전라남도", "경북": "경상북도", "경남": "경상남도", "제주": "제주특별자치도" };
    const sido = SIDO_FULL[parts[0]] || parts[0] || "";
    let sgg = parts[1] || "";
    if (parts[1] && /시$/.test(parts[1]) && parts[2] && /구$/.test(parts[2])) sgg = `${parts[1]} ${parts[2]}`;
    const landOnly = !(bld > 0) && /토지|대지|임야|전|답/.test(kindRaw);
    const usage6 = usageOf(kindRaw, landOnly || !(bld > 0) && !/공장|창고|상가|근린|아파트|주택|빌라|오피스텔/.test(kindRaw));

    set("name", [court.replace("지방법원", "지법"), caseNo, kindRaw].filter(Boolean).join(" ") || null, "사건");
    set("apsl", apsl, "감정가"); set("usage6", usage6, `용도 (${kindRaw || "?"})`); set("month", saleDate ? saleDate.slice(0, 7) : null, "매각기일");
    set("sale", sale, "매각가"); set("costFixed", cost, "집행비용"); set("sido", sido, "시도"); set("sgg", sgg, "시군구");
    const prog = parseProgress(texts, apsl);
    set("minPrice", prog.minPrice, "현재 최저가"); set("fails", prog.fails, `유찰 횟수 (${prog.failsFrom})`);
    if (apsl) warnings.push(...prog.warnings);
    set("land", land, "토지면적"); set("lp", lp, "공시지가");
    if (bld > 0) { F.hasBld = true; set("bld", bld, "건물면적"); set("approve", approve, "사용승인일"); set("tot", tot, "연면적"); }
    set("baseDate", baseDate, "최선순위 설정일");
    if (rights.length) found.push(`권리 ${rights.length}건`);
    if (tenants.length) found.push(`임차인 ${tenants.length}명`);
    if (!apsl) warnings.push(sale ? "감정가가 없어요. 낙찰가 예측 없이, 문서의 매각가로 배당표만 계산해요." : "감정가를 찾지 못했어요. 1단계에서 넣어주세요.");
    if (kindRaw && !usage6) warnings.push(`물건 종류 '${kindRaw}'을 용도에 맞추지 못했어요. 1단계에서 골라주세요.`);
    if (!rights.length) warnings.push("등기부 권리를 찾지 못했어요. 2단계에서 직접 넣어주세요.");
    if (rights.some((x) => x.type === "압류" && x.amount === null && ["세무서", "지자체"].includes(x.who))) warnings.push("세금 압류 금액이 없어요. 5단계 체납세금 시나리오로 계산해요.");
    if (sido && !Object.values(SIDO_FULL).includes(sido)) warnings.push(`시도 '${sido}'를 알아보지 못했어요. 1단계에서 골라주세요.`);
    return { fields: F, rights: rights.sort((a, b) => (a.date || "9999") < (b.date || "9999") ? -1 : 1), tenants, tenantsNone, warnings, found,
      source: "일반 문서", meta: { caseNo, court, addr, kindRaw, saleDate } };
  }

  /** PDF 조각 또는 붙여넣은 글 → 결과. 옥션원이면 정밀 인식, 아니면 내용 인식 */
  function parseDocument(input, cats) {
    const lines = typeof input === "string"
      ? input.split(/\r?\n/).map((t) => ({ text: t.replace(/\t+/g, "  ").trim(), flat: t.replace(/\s+/g, "") })).filter((L) => L.text)
      : toLines(input);
    const isA1 = Array.isArray(input) && lines.some((L) => /옥션원|auction1/.test(L.text)) && lines.some((L) => /^No접수권리종류/.test(L.flat));
    return isA1 ? parseAuction1(input, cats) : parseGeneric(lines, cats);
  }

  /** 여러 문서 결과 합치기: 물건 칸은 먼저 읽은 값 우선, 권리·임차인은 합치고 중복 제거 */
  function mergeResults(list) {
    const out = { fields: {}, rights: [], tenants: [], tenantsNone: false, warnings: [], found: [], sources: [], meta: {} };
    const seen = new Set();
    list.forEach(({ res, file }) => {
      for (const [k, v] of Object.entries(res.fields)) if (out.fields[k] === undefined) out.fields[k] = v;
      res.rights.forEach((r) => { const k = [r.date, r.type, (r.name || "").replace(/\s/g, ""), r.amount].join("|"); if (!seen.has(k)) { seen.add(k); out.rights.push(r); } });
      if (!out.tenants.length) out.tenants = res.tenants || [];
      out.tenantsNone = out.tenantsNone || res.tenantsNone;
      res.warnings.forEach((w) => { if (!out.warnings.includes(w)) out.warnings.push(w); });
      res.found.forEach((f) => { if (!out.found.includes(f)) out.found.push(f); });
      for (const [k, v] of Object.entries(res.meta || {})) if (!out.meta[k] && v) out.meta[k] = v;
      out.sources.push(`${file} (${res.source})`);
    });
    if (out.tenants.length) out.tenantsNone = false;
    out.rights.sort((a, b) => ((a.date || "9999") < (b.date || "9999") ? -1 : (a.date || "9999") > (b.date || "9999") ? 1 : 0));
    return out;
  }

  const api = { parseDocument, mergeResults, parseAuction1, parseGeneric, parseProgress, toLines, usageOf, whoOf, typeOf };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.DocParser = api;
})(typeof window !== "undefined" ? window : globalThis);
