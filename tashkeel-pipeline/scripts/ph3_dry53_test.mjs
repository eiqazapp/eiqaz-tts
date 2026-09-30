// ph3_dry53_test.mjs — offline dry test for engine v5.3 (no API, no quota).
// Verifies:
//  A) SYSTEM_PROMPT byte-identity with the approved v5/v5.1/v5.2 prompt
//     (sha256 == 60130f6b… recorded in the approved smoke meta).
//  B) v5.3 retry scheduling: batched retry rounds for rounds 1-2, then the
//     FOCUSED individual drain (tashkeelSingle) as the unit's last budget
//     round — hard unit recovered on the individual call; budget parity
//     (attempts max 4); punct_unswap accumulation; record schema.
//  C) The individual call's user message is the v5 wording: single sentence,
//     no numbering, attempt note, start-fresh hint on the last round.
//  D) Unit failing even the individual drain -> failed record attempts=4.
//  E) Batch-call API-error parity: units re-queued (attempts=0), recovered
//     in a retry round (attempts=1) — same record shape as v5.1/v5.2.
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import crypto from 'node:crypto';
import { SELF } from './ph3_engine_v53.mjs';

const ROOT = '/home/z/my-project';
const DIR = `${ROOT}/work/dry53`;
fs.rmSync(DIR, { recursive: true, force: true });
fs.mkdirSync(DIR, { recursive: true });

const APPROVED_PROMPT_SHA =
  '60130f6baeca1b56d69a14e90d72f8f11a27f6d7c5b2e08832230800b1678149';

let failures = 0;
const ok = (cond, label) => {
  console.log(`${cond ? 'PASS' : 'FAIL'}  ${label}`);
  if (!cond) failures++;
};

// ---------- A) prompt identity ----------
const sha = crypto.createHash('sha256').update(SELF.SYSTEM_PROMPT).digest('hex');
ok(sha === APPROVED_PROMPT_SHA, `A1 system prompt sha256 == approved (${sha.slice(0, 12)}…)`);
ok(SELF.MAX_RETRY_ROUNDS === 3, 'A2 per-unit retry budget unchanged (3 rounds / max 4 attempts)');

// ---------- helpers ----------
const runEngine = (inputPath, outPath, batch, retryBatch, planPath, extraEnv = {}) => {
  const logPath = planPath.replace(/\.json$/, '_calls.log');
  fs.rmSync(logPath, { force: true });
  const r = spawnSync('node', [
    `${ROOT}/scripts/ph3_engine_v53.mjs`, inputPath, outPath, String(batch),
  ], {
    cwd: ROOT,
    encoding: 'utf8',
    env: {
      ...process.env,
      RETRY_BATCH: String(retryBatch),
      PH3_FAKE_LLM: `${ROOT}/scripts/ph3_fake_llm.mjs`,
      PH3_FAKE_PLAN: planPath,
      PH3_FAKE_LOG: logPath,
      ...extraEnv,
    },
  });
  return { stdout: r.stdout, stderr: r.stderr, status: r.status,
           calls: fs.existsSync(logPath)
             ? fs.readFileSync(logPath, 'utf8').split('\n')
                 .filter((l) => l.trim()).map((l) => JSON.parse(l))
             : [] };
};
const readOut = (p) => fs.readFileSync(p, 'utf8').split('\n')
  .filter((l) => l.trim()).map((l) => JSON.parse(l));

// ---------- B) v5.3 scheduling: batch rounds then individual drain ----------
const input1 = [
  { id: 'u1', text: 'الراجل ده جاي' },
  { id: 'u2', text: 'الراجل ده جاي, ماشي?' },
  { id: 'u3', text: 'الوضع ده كويس' },
  { id: 'u4', text: 'الواحد بيقرا كتير' },
];
const in1 = `${DIR}/input1.json`;
fs.writeFileSync(in1, JSON.stringify(input1));

const plan1 = [
  // call 1 — first-pass batch (u1,u2,u3): u1 valid; u2 raw (density fail); u3 tanween off-list
  '1. اِلرَّاجِل دَهْ جَايْ\n2. الراجل ده جاي, ماشي?\n3. اِلْوُضِعْ دَهْ كُوَيْسً',
  // call 2 — retry batch (u2,u3): both fixed; u2 swaps punct to Arabic (unswap must fire)
  '1. اِلرَّاجِلَ دَهْ جَايْ، مَاشِي؟\n2. اِلْوُضِعْ دَهْ كُوَيْسْ',
  // call 3 — first-pass batch (u4): skeleton error (extra alef in كتيرا)
  '1. اِلْوَاحِدْ بِيْقِرَا كَتِيرًا',
  // call 4 — batched retry round (u4): still failing (attempts=2 -> retryQ)
  '1. اِلْوَاحِدْ بِيْقِرَا كَتِيرًا',
  // call 5 — batched retry round (u4): still failing (attempts=3 -> finalQ)
  '1. اِلْوَاحِدْ بِيْقِرَا كَتِيرًا',
  // call 6 — FINAL INDIVIDUAL drain (u4): fixed (v5-style single line, no numbering)
  'اِلْوَاحِدْ بِيْقِرَا كَتِيرْ',
];
const plan1Path = `${DIR}/plan1.json`;
fs.writeFileSync(plan1Path, JSON.stringify(plan1));

const out1 = `${DIR}/out1.jsonl`;
const r1 = runEngine(in1, out1, 3, 2, plan1Path);
console.log('--- engine run 1 (stdout tail) ---');
console.log(r1.stdout.split('\n').filter(Boolean).slice(-8).join('\n'));
ok(r1.status === 0, 'B1 engine exit code 0');

const recs = readOut(out1);
const by = Object.fromEntries(recs.map((r) => [r.id, r]));
ok(recs.length === 4, `B2 exactly 4 records (got ${recs.length})`);
ok(by.u1 && by.u1.status === 'ok' && by.u1.attempts === 1, 'B3 u1 ok attempts=1');
ok(by.u2 && by.u2.status === 'ok' && by.u2.attempts === 2, 'B4 u2 ok attempts=2 (one batched retry round)');
ok(by.u2 && by.u2.fixes && by.u2.fixes.punct_unswap === 2,
  `B5 u2 punct_unswap=2 (got ${JSON.stringify(by.u2?.fixes)})`);
ok(by.u3 && by.u3.status === 'ok' && by.u3.attempts === 2, 'B6 u3 ok attempts=2');
ok(by.u4 && by.u4.status === 'ok' && by.u4.attempts === 4,
  `B7 u4 ok attempts=4 — recovered on the individual drain (got ${by.u4?.status}/${by.u4?.attempts})`);
ok(by.u4 && typeof by.u4.out === 'string' && by.u4.out.includes('بِيْقِرَا كَتِيرْ'),
  'B8 u4 out = the individual-round output');
const keysOk = recs.every((r) => ['id', 'raw', 'converted', 'out', 'attempts',
  'status', 'fixes', 'checks', 'reason'].every((k) => k in r));
ok(keysOk, 'B9 record schema = v5.1 schema');

// call structure from the fake log
const calls = r1.calls;
ok(calls.length === 6, `B10 total LLM calls = 6 (got ${calls.length})`);
const sysAll = calls.every((c) =>
  c.messages.find((m) => m.role === 'system')?.content === SELF.SYSTEM_PROMPT);
ok(sysAll, 'B11 every call (incl. individual drain) uses the byte-identical system prompt');

// ---------- C) the individual call is the v5 single-retry shape ----------
const c6 = calls[5]?.messages.find((m) => m.role === 'user')?.content ?? '';
ok(c6.includes('إضافة التشكيل إلى هذه الجملة فقط'),
  'C1 individual user msg: v5 single-sentence task wording');
ok(c6.includes('أعد سطرًا واحدًا بلا رقم'),
  'C2 individual user msg: one line, no numbering');
ok(c6.includes('الواحد بيقرا كتير'), 'C3 individual user msg carries the raw sentence');
ok(c6.includes('محاولة 3'), 'C4 individual user msg: attempt note (3)');
ok(c6.includes('تجاهل محاولاتك السابقة تمامًا'),
  'C5 individual user msg: start-fresh hint on the last round');
ok(!/\n\s*1\./.test(c6), 'C6 individual user msg has no numbered-list contract');
const c5 = calls[4]?.messages.find((m) => m.role === 'user')?.content ?? '';
ok(c5.includes('الجملة: الواحد بيقرا كتير') && c5.includes('1.'),
  'C7 call 5 is still a BATCHED retry round (numbered list)');
ok(r1.stdout.includes('final individual drain: u4'),
  'C8 engine logs the final individual drain');

// meta + token stats
const meta1 = JSON.parse(fs.readFileSync(`${DIR}/out1_meta.json`, 'utf8'));
ok(meta1.prompt_sha256 === APPROVED_PROMPT_SHA, 'C9 meta prompt_sha256 == approved');
ok(meta1.token_stats.first_calls === 2 && meta1.token_stats.retry_calls === 4,
  `C10 token_stats calls: first=2 retry=4 (got ${meta1.token_stats.first_calls}/${meta1.token_stats.retry_calls})`);
ok((meta1.engine ?? '').includes('v5.3'), 'C11 meta engine string = v5.3');

// resume: nothing to do
const before = readOut(out1).length;
const r1b = runEngine(in1, out1, 3, 2, plan1Path);
ok(r1b.stdout.includes('nothing to do'), 'C12 resume: nothing to do');
ok(readOut(out1).length === before, 'C13 resume: output file unchanged');

// ---------- D) unit failing even the individual drain ----------
const input2 = [
  { id: 'w1', text: 'الواحد بيقرا كتير' },
];
const in2 = `${DIR}/input2.json`;
fs.writeFileSync(in2, JSON.stringify(input2));
const plan2 = [
  'الواحد بيقرا كتير',          // call 1 — first pass: raw (density fail)
  '1. اِلْوَاحِدْ بِيْقِرَا كَتِيرًا',  // call 2 — batched retry: skeleton fail
  '1. اِلْوَاحِدْ بِيْقِرَا كَتِيرًا',  // call 3 — batched retry: skeleton fail
  'اِلْوَاحِدْ بِيْقِرَا كَتِيرًا',   // call 4 — individual drain: skeleton fail
];
const plan2Path = `${DIR}/plan2.json`;
fs.writeFileSync(plan2Path, JSON.stringify(plan2));
const out2 = `${DIR}/out2.jsonl`;
const r2 = runEngine(in2, out2, 1, 2, plan2Path);
ok(r2.status === 0, 'D1 engine exit code 0');
const recs2 = readOut(out2);
ok(recs2.length === 1 && recs2[0].status === 'failed' && recs2[0].attempts === 4,
  `D2 w1 failed attempts=4 after exhausting the individual drain (got ${recs2[0]?.status}/${recs2[0]?.attempts})`);
ok(recs2[0].checks && Object.values(recs2[0].checks).every((x) => x === false),
  'D3 w1 checks all false');
ok(r2.calls.length === 4, `D4 calls = 4 (got ${r2.calls.length})`);

// ---------- E) batch-level API error -> retry round recovery ----------
const input3 = [
  { id: 'v1', text: 'الراجل ده جاي' },
  { id: 'v2', text: 'الوضع ده كويس' },
];
const in3 = `${DIR}/input3.json`;
fs.writeFileSync(in3, JSON.stringify(input3));
const plan3 = [
  'THROW:simulated batch API failure',
  '1. اِلرَّاجِل دَهْ جَايْ\n2. اِلْوُضِعْ دَهْ كُوَيْسْ',
];
const plan3Path = `${DIR}/plan3.json`;
fs.writeFileSync(plan3Path, JSON.stringify(plan3));
const out3 = `${DIR}/out3.jsonl`;
const r3 = runEngine(in3, out3, 2, 2, plan3Path);
ok(r3.status === 0, 'E1 engine exit code 0 after batch error');
const recs3 = readOut(out3);
ok(recs3.length === 2 && recs3.every((x) => x.status === 'ok'),
  `E2 both units ok after batch-level failure (got ${JSON.stringify(recs3.map((x) => x.status))})`);
ok(recs3.every((x) => x.attempts === 1),
  'E3 attempts=1 (v5.1/v5.2 parity: batch error consumed no per-unit attempt)');
ok(r3.calls.length === 2, `E4 calls: 1 failed batch + 1 retry round (got ${r3.calls.length})`);
ok(r3.calls[1]?.messages.find((m) => m.role === 'user')?.content
  .includes('سبب الرفض: simulated batch API failure'),
  'E5 retry round feedback carries the batch error as rejection reason');

console.log(`\n${failures === 0 ? 'ALL DRY TESTS PASSED' : failures + ' FAILURES'}`);
process.exit(failures === 0 ? 0 : 1);
