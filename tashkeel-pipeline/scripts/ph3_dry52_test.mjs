// ph3_dry52_test.mjs — offline dry test for engine v5.2 (no API, no quota).
// Verifies:
//  A) SYSTEM_PROMPT byte-identity with the approved v5/v5.1 prompt
//     (sha256 == 60130f6b… recorded in the approved smoke meta).
//  B) Retry-rebatching mechanics with a deterministic fake LLM:
//     failures queue up, retry rounds batch them together with per-sentence
//     feedback (prev output + reason), per-unit budget (max 4 attempts),
//     punct_unswap accumulation across rounds, failed-record shape,
//     token/call accounting, resume (done-set) semantics.
//  C) Batch-call API-error parity: units re-queued (attempts=0), recovered
//     in a retry round (attempts=1) — same record shape as v5.1.
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import crypto from 'node:crypto';
import { SELF } from './ph3_engine_v52.mjs';

const ROOT = '/home/z/my-project';
const DIR = `${ROOT}/work/dry52`;
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

// ---------- helpers ----------
const runEngine = (inputPath, outPath, batch, retryBatch, planPath, extraEnv = {}) => {
  const logPath = planPath.replace(/\.json$/, '_calls.log');
  fs.rmSync(logPath, { force: true });
  const r = spawnSync('node', [
    `${ROOT}/scripts/ph3_engine_v52.mjs`, inputPath, outPath, String(batch),
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

// ---------- B) main mechanics ----------
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
  // call 2 — retry round (u2,u3): both fixed; u2 swaps punct to Arabic (unswap must fire)
  '1. اِلرَّاجِلَ دَهْ جَايْ، مَاشِي؟\n2. اِلْوُضِعْ دَهْ كُوَيْسْ',
  // call 3 — first-pass batch (u4): skeleton error (extra alef in كتيرا)
  '1. اِلْوَاحِدْ بِيْقِرَا كَتِيرًا',
  // calls 4,5,6 — retry rounds for u4: persistent failure until budget ends
  '1. اِلْوَاحِدْ بِيْقِرَا كَتِيرًا',
  '1. اِلْوَاحِدْ بِيْقِرَا كَتِيرًا',
  '1. اِلْوَاحِدْ بِيْقِرَا كَتِيرًا',
];
const plan1Path = `${DIR}/plan1.json`;
fs.writeFileSync(plan1Path, JSON.stringify(plan1));

const out1 = `${DIR}/out1.jsonl`;
const r1 = runEngine(in1, out1, 3, 2, plan1Path);
console.log('--- engine run 1 (stdout tail) ---');
console.log(r1.stdout.split('\n').filter(Boolean).slice(-6).join('\n'));
ok(r1.status === 0, 'B1 engine exit code 0');

const recs = readOut(out1);
const by = Object.fromEntries(recs.map((r) => [r.id, r]));
ok(recs.length === 4, `B2 exactly 4 records (got ${recs.length})`);
ok(by.u1 && by.u1.status === 'ok' && by.u1.attempts === 1, 'B3 u1 ok attempts=1');
ok(by.u2 && by.u2.status === 'ok' && by.u2.attempts === 2, 'B4 u2 ok attempts=2 (one retry round)');
ok(by.u2 && by.u2.fixes && by.u2.fixes.punct_unswap === 2,
  `B5 u2 punct_unswap=2 (got ${JSON.stringify(by.u2?.fixes)})`);
ok(by.u3 && by.u3.status === 'ok' && by.u3.attempts === 2, 'B6 u3 ok attempts=2');
ok(by.u4 && by.u4.status === 'failed' && by.u4.attempts === 4,
  `B7 u4 failed attempts=4 (got ${by.u4?.status}/${by.u4?.attempts})`);
ok(by.u4 && typeof by.u4.reason === 'string' && by.u4.reason.includes('الهيكل'),
  'B8 u4 reason = skeleton violation');
ok(by.u4 && typeof by.u4.out === 'string' && by.u4.out.length > 0,
  'B9 u4 failed record keeps last (gate-failing) output — v5.1 parity');
ok(by.u4 && Object.values(by.u4.checks).every((x) => x === false),
  'B10 u4 checks all false');
const keysOk = recs.every((r) => ['id', 'raw', 'converted', 'out', 'attempts',
  'status', 'fixes', 'checks', 'reason'].every((k) => k in r));
ok(keysOk, 'B11 record schema = v5.1 schema');

// call structure from the fake log
const calls = r1.calls;
ok(calls.length === 6, `B12 total LLM calls = 6 (got ${calls.length})`);
const sysAll = calls.every((c) =>
  c.messages.find((m) => m.role === 'system')?.content === SELF.SYSTEM_PROMPT);
ok(sysAll, 'B13 every call (incl. retry rounds) uses the byte-identical system prompt');
const call2user = calls[1]?.messages.find((m) => m.role === 'user')?.content ?? '';
ok(call2user.includes('المخرج السابق (مرفوض): الراجل ده جاي, ماشي?'),
  'B14 retry user msg carries u2 prev output');
ok(call2user.includes('المخرج السابق (مرفوض): اِلْوُضِعْ دَهْ كُوَيْسً'),
  'B15 retry user msg carries u3 prev output');
ok(call2user.includes('سبب الرفض') && call2user.includes('تنوين'),
  'B16 retry user msg carries per-sentence rejection reason');
ok(call2user.includes('1. الجملة: الراجل ده جاي, ماشي?')
  && call2user.includes('2. الجملة: الوضع ده كويس'),
  'B17 retry user msg re-states raw sentences with their numbers');
const call6user = calls[5]?.messages.find((m) => m.role === 'user')?.content ?? '';
ok(call6user.includes('تجاهل محاولاتك السابقة تمامًا'),
  'B18 3rd retry round hides prev output (start-fresh hint) — v5.1 parity');
ok(!call6user.includes('المخرج السابق (مرفوض): اِلْوَاحِدْ'),
  'B19 3rd retry round does not show the rejected output');

// meta + token stats
const meta1 = JSON.parse(fs.readFileSync(`${DIR}/out1_meta.json`, 'utf8'));
ok(meta1.prompt_sha256 === APPROVED_PROMPT_SHA, 'B20 meta prompt_sha256 == approved');
ok(typeof meta1.token_stats === 'object', 'B21 meta carries token_stats');
ok(meta1.token_stats.first_calls === 2 && meta1.token_stats.retry_calls === 4,
  `B22 token_stats calls: first=2 retry=4 (got ${meta1.token_stats.first_calls}/${meta1.token_stats.retry_calls})`);
ok(meta1.retry_batch_size === 2 && meta1.max_retry_rounds === 3,
  'B23 meta records retry config');

// resume: nothing to do
const before = readOut(out1).length;
const r1b = runEngine(in1, out1, 3, 2, plan1Path);
ok(r1b.stdout.includes('nothing to do'), 'B24 resume: nothing to do');
ok(readOut(out1).length === before, 'B25 resume: output file unchanged');

// ---------- C) batch-level API error -> retry round recovery ----------
const input2 = [
  { id: 'v1', text: 'الراجل ده جاي' },
  { id: 'v2', text: 'الوضع ده كويس' },
];
const in2 = `${DIR}/input2.json`;
fs.writeFileSync(in2, JSON.stringify(input2));
const plan2 = [
  'THROW:simulated batch API failure',
  '1. اِلرَّاجِل دَهْ جَايْ\n2. اِلْوُضِعْ دَهْ كُوَيْسْ',
];
const plan2Path = `${DIR}/plan2.json`;
fs.writeFileSync(plan2Path, JSON.stringify(plan2));
const out2 = `${DIR}/out2.jsonl`;
const r2 = runEngine(in2, out2, 2, 2, plan2Path);
ok(r2.status === 0, 'C1 engine exit code 0 after batch error');
const recs2 = readOut(out2);
ok(recs2.length === 2 && recs2.every((x) => x.status === 'ok'),
  `C2 both units ok after batch-level failure (got ${JSON.stringify(recs2.map((x) => x.status))})`);
ok(recs2.every((x) => x.attempts === 1),
  'C3 attempts=1 (v5.1 parity: batch error consumed no per-unit attempt)');
ok(r2.calls.length === 2, `C4 calls: 1 failed batch + 1 retry round (got ${r2.calls.length})`);
ok(r2.calls[1]?.messages.find((m) => m.role === 'user')?.content
  .includes('سبب الرفض: simulated batch API failure'),
  'C5 retry round feedback carries the batch error as rejection reason');

console.log(`\n${failures === 0 ? 'ALL DRY TESTS PASSED' : failures + ' FAILURES'}`);
process.exit(failures === 0 ? 0 : 1);
