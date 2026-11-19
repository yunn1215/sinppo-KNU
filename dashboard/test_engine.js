// 파이썬 = JS 일치 확인 (최종 시뮬레이터):  node dashboard/test_engine.js   (먼저 python pipeline/step7_시뮬레이터.py)
const fs = require("fs"), path = require("path");
const E = require("./engine.js");
const OUT = path.join(__dirname, "..", "output", "07_시뮬레이터");
const DATA = JSON.parse(fs.readFileSync(path.join(OUT, "simulator_data.json"), "utf8"));
const REF = JSON.parse(fs.readFileSync(path.join(OUT, "simulator_ref.json"), "utf8"));
let fail = 0;

// ⓪ 손계산 테스트 (step3_dividend.hand_tests 와 같은 값)
{
  const C = (id, amount, date, secured) => (secured === undefined ? { id, name: id, amount } : { id, name: id, amount, date, secured });
  const G = (name, claims, extra = {}) => ({ name, claims, ...extra });
  const at = (price, groups, id) => E.waterfall([price], groups).paid[id][0];
  const cases = [];
  const base = [G("집행비용", [C("비용", 10e6)]), G("근저당", [C("은행", 540e6)]), G("일반", [C("신보", 300e6), C("캐피탈", 200e6)])];
  cases.push(["T1 낙찰가 6억 → 신보", at(600e6, base, "신보"), 30e6], ["T2 낙찰가 5억 → 신보", at(500e6, base, "신보"), 0]);
  const same = [G("같은날", [C("A", 200e6), C("B", 400e6)])];
  cases.push(["T3 같은 날 A", at(300e6, same, "A"), 100e6], ["T3 같은 날 B", at(300e6, same, "B"), 200e6]);
  cases.push(["T4 소액임차 1/2 한도", at(100e6, [G("최우선", [C("임1", 28e6), C("임2", 28e6)], { caps: [[0.5, [0, 1]]] })], "임1"), 25e6]);
  cases.push(["T5 가압류 안분 예외", at(400e6, [G("2020", [C("신보", 300e6), C("은행", 300e6)])], "신보"), 200e6]);
  const pool = [G("풀", [C("A", 200e6, "2019-01-01", false), C("B", 300e6, "2020-01-01", true), C("C", 100e6, "2021-01-01", false)], { pool: true })];
  cases.push(["T6 안분 후 흡수 A", at(300e6, pool, "A"), 100e6], ["T6 안분 후 흡수 B", at(300e6, pool, "B"), 200e6], ["T6 안분 후 흡수 C", at(300e6, pool, "C"), 0]);
  cases.push(["T7 소액 한도 = 배당할 돈의 1/2", at(100e6, [G("비용", [C("E", 10e6)]), G("최우선", [C("임1", 28e6), C("임2", 28e6)], { caps: [[0.5, [0, 1]]] })], "임1"), 22.5e6]);
  const late = [G("비용", [C("E", 10e6)]), G("신보근저당", [C("신보", 300e6)]), G("세금", [C("T", 100e6)])];
  cases.push(["T8 세금이 늦을 때", at(350e6, late, "신보"), 300e6], ["T8 세금이 빠를 때", at(350e6, [late[0], late[2], late[1]], "신보"), 240e6]);
  // T9 일부이전 + 최고액: 은행 근저당 최고액 10억 안에 신보 일부이전 2억, 유동화회사(양도인 A은행) 경매 청구 9억
  //    → 은행 몫 = min(9억, 최고액 10억 − 신보 2억) = 8억 (1억은 담보 밖), 신보 2억은 원래 근저당 순위 그대로. 신보 채권액 2억
  const rt = [{ type: "근저당권설정", who: "은행·금융기관", name: "A은행", date: "2020-01-01", amount: 1e9, valid: true, memo: "" },
    { type: "근저당권설정", who: "신보", name: "신용보증기금", date: "2024-04-01", amount: 2e8, valid: true, memo: "A은행근저당권일부이전 · 확정채권일부대위변제" },
    { type: "경매개시결정", who: "개인·기타 법인", name: "B유동화", date: "2024-06-01", amount: 9e8, valid: true, memo: "임의경매 · 양도인A은행" },
    { type: "경매개시결정", who: "신보", name: "신용보증기금", date: "2024-05-01", amount: 2e8, valid: true, memo: "임의경매" }];
  const S9 = { sinboMode: "general", sinboAmount: 0, useActualSinbo: true, costRate: 0, mortgageRatio: 1.2, absorb: true };
  for (const ps of ["pro_rata", "bank_first"]) {
    const b9 = E.buildCase({ apsl: 1e9, rights: rt, tenants: [] }, { ...S9, partialSubrogation: ps }, DATA.small);
    const t9 = E.table(b9, 2e9, 0);
    const sb = t9.rows.find((r) => r.sinbo), bank = t9.rows.find((r) => r.name.startsWith("A은행"));
    cases.push([`T9 (${ps}) 은행 몫 = 최고액 − 신보 몫`, bank.claim, 8e8], [`T9 (${ps}) 신보`, sb.claim, 2e8],
      [`T9 (${ps}) 신보 순위 = 원래 근저당`, sb.group.includes("2020-01-01") ? 1 : 0, 1], [`T9 (${ps}) 신보 채권액`, b9.B, 2e8]);
  }
  // T10 토지·건물 등기부별 배당 (민법 368조, 경매대가 = 그 부동산 몫 − 선순위 분담액, 반복 계산 · 대법원 2001다66291)
  //     6번 25억: 비율 3.70억 / 은행 먼저 0원,  3번 22억: 비율 9.48억  (fixtures_split.json, 1백만 원 이내)
  const FX = JSON.parse(fs.readFileSync(path.join(__dirname, "fixtures_split.json"), "utf8"));
  for (const f of Object.values(FX)) for (const [ps, want] of Object.entries(f.want)) {
    const b = E.buildCase({ apsl: f.apsl, region: "그 밖의 지역", shares: { 토지: f.land, 건물: f.apsl - f.land }, rights: f.rights, tenants: [] },
      { ...DATA.settings, sinboAmount: 0, partialSubrogation: ps, taxFirst: false }, DATA.small);
    const got = E.table(b, f.price, 0).rows.filter((r) => r.sinbo).reduce((a2, r) => a2 + r.paid, 0);
    cases.push([`T10 ${f.name} ${f.price / 1e8}억 ${ps === "pro_rata" ? "비율" : "은행 먼저"} (${(got / 1e8).toFixed(3)}억)`, Math.round(got / 1e6), Math.round(want / 1e6)]);
  }
  const bad = cases.filter(([, got, want]) => Math.abs(got - want) > 1);
  bad.forEach(([l, got, want]) => console.log("  ❌", l, got, want));
  if (bad.length) fail++;
  console.log(`⓪ 손계산 테스트 ${cases.length}개: ${bad.length ? "❌ " + bad.length + "개 실패" : "모두 통과 ✅"}`);
}

// ① 회차표 (최저가율 · 매각확률 · log 배수 P10/P50/P90): 배당 50건, 결정적 계산이라 정확히 같아야 함
{
  let maxd = 0, n = 0;
  for (const r of REF.pred) {
    const { f } = E.buildFeatures2(r.prop, DATA.rone, DATA.model);
    const rt = E.roundTable(DATA.model, f, r.rto[0]);
    rt.forEach((x, k) => {
      const d = Math.max(Math.abs(x.rto - r.rto[k]), Math.abs(x.p - r.p[k]), ...x.P.map((v, j) => Math.abs(v - r.P[k][j])));
      if (d > maxd) maxd = d;
      if (d > 1e-9 && n++ < 4) console.log("  회차표 불일치", r.key, k + 1, x.p, r.p[k], x.P, r.P[k]);
    });
  }
  if (maxd > 1e-9) fail++;
  console.log(`① 회차표 ${REF.pred.length}건 × ${DATA.model.maxRounds}회차: 최대 차이 ${maxd.toExponential(2)} ${maxd <= 1e-9 ? "✅" : "❌"}`);
  // 화면 입력(현재 최저가)으로 시작 최저가율이 같은지
  let md = 0;
  for (const r of REF.pred) md = Math.max(md, Math.abs(E.predictAuction2(DATA.model, DATA.rone, r.prop, 200).startRto - r.rto[0]));
  console.log(`   현재 최저가 → 시작 최저가율: 최대 차이 ${md.toExponential(2)} ${md <= 1e-9 ? "✅" : "❌"}`);
  if (md > 1e-9) fail++;
}
// ①' 회차별 몬테카를로: 난수만 달라 근사 비교 (파이썬·JS 각 2만 회)
{
  let w50 = 0, w1090 = 0, wu = 0;
  for (const r of REF.pred) {
    const g = E.predictAuction2(DATA.model, DATA.rone, r.prop, 20000, 7);
    w50 = Math.max(w50, Math.abs(g.p[1] - r.mc[1]));
    w1090 = Math.max(w1090, Math.abs(g.p[0] - r.mc[0]), Math.abs(g.p[2] - r.mc[2]));
    wu = Math.max(wu, Math.abs(g.unsold - r.unsold));
  }
  const ok = w50 < 0.01 && w1090 < 0.02 && wu < 0.01;
  if (!ok) fail++;
  console.log(`①' 몬테카를로 최종 낙찰가율 (JS vs 파이썬): P50 최대 차이 ${(w50 * 100).toFixed(2)}%p · P10/P90 ${(w1090 * 100).toFixed(2)}%p · 미매각 ${(wu * 100).toFixed(2)}%p ${ok ? "✅" : "❌"}`);
  const t0 = Date.now(); E.predictAuction2(DATA.model, DATA.rone, REF.pred[0].prop);
  console.log(`   화면 1회 계산 시간 (${DATA.model.nSim.toLocaleString()}회) ${Date.now() - t0}ms`);
}

// ② 배당: 실제 매각가 × 세금 시나리오 중간값으로 배당표 비교
const S0 = DATA.settings;
const taxMid = (() => { const t = S0.taxScenarios.slice().sort((a, b) => a - b), n = t.length; return n % 2 ? t[(n - 1) / 2] : (t[n / 2 - 1] + t[n / 2]) / 2; })();
let ok = 0, bad = 0;
for (const c of DATA.cases) {
  const ref = REF.cases.find((x) => x.key === c.key);
  const S = { ...S0, sinboAmount: c.apsl * S0.sinboRate, sinboDate: null };
  const built = E.buildCase(c, S, DATA.small);
  const tx = built.taxNeeded ? taxMid : 0;
  const t = E.table(built, c.sale, tx);
  const got = t.rows.map((r) => [Math.round(r.claim), Math.round(r.paid)]).concat([[-1, Math.round(t.remain)]]);
  const want = ref.table.map((r) => [Math.round(r[1]), Math.round(r[2])]);
  const sinbo = t.rows.filter((r) => r.sinbo).reduce((s, r) => s + r.paid, 0);
  // 파이썬 round()는 .5를 짝수로, JS Math.round는 위로 → 1원 이내면 같은 값
  const same = got.length === want.length && got.every((r, i) => Math.abs(r[0] - want[i][0]) <= 1 && Math.abs(r[1] - want[i][1]) <= 1)
    && Math.abs(sinbo - ref.sinbo_actual) < 1 && Math.abs(built.B - ref.sinbo_claim) < 1;
  if (same) ok++; else { bad++; fail++; if (bad < 4) console.log("배당 불일치", c.key, "\n JS ", JSON.stringify(got), "\n PY ", JSON.stringify(want)); }
}
console.log(`② 배당표 ${DATA.cases.length}건: 일치 ${ok} / 불일치 ${bad} ${bad ? "❌" : "✅"}`);

// ③ 몬테카를로: 난수가 달라 값이 정확히 같지는 않음 → 요약값이 가까운지
let worst = 0;
for (const c of DATA.cases) {
  const ref = REF.cases.find((x) => x.key === c.key);
  const S = { ...S0, sinboAmount: c.apsl * S0.sinboRate, sinboDate: null };
  const built = E.buildCase(c, S, DATA.small);
  const sm = E.simulate(built, c.apsl, ref.p, S0.taxScenarios, 10000, 2026).summary;
  worst = Math.max(worst, Math.abs(sm.zero - ref.mc_zero));
}
console.log(`③ 몬테카를로 0원 확률 (JS vs 파이썬, 난수만 다름): 최대 차이 ${(worst * 100).toFixed(1)}%p ${worst < 0.03 ? "✅" : "⚠️"}`);
const t0 = Date.now(); const c = DATA.cases[0];
E.simulate(E.buildCase(c, { ...S0, sinboAmount: c.apsl * 0.3 }, DATA.small), c.apsl, [0.3, 0.5, 0.7], S0.taxScenarios);
console.log(`   1만 번 계산 시간 ${Date.now() - t0}ms`);
process.exit(fail ? 1 : 0);
