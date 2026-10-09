#!/usr/bin/env node
/**
 * parity_textpipe.mjs — اختبار تكافؤ خط النص JS مقابل بايثون (Eiqaz v1)
 * ====================================================================
 * 1) فحص فوري للتطبيع (expected_normalize.json — بلا أي نموذج)
 * 2) فحص توكن-بموكن كامل (expected_tokens.json — وضع manual الحتمي)
 *
 * التشغيل:  node web-exp/tests/parity_textpipe.mjs
 * الخروج:   0 = PASS كامل · 1 = فشل حالة أو أكثر
 */
'use strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, '..');

await import(join(ROOT, 'models', 'textpipe_data.js'));
await import(join(ROOT, 'textpipe.js'));
const TP = globalThis.TextPipe;

let pass = 0, fail = 0;
const failures = [];

// ---- 1) فحص التطبيع ------------------------------------------------------
const normExpected = JSON.parse(
  readFileSync(join(HERE, 'expected_normalize.json'), 'utf-8'));
for (const c of normExpected) {
  try {
    const got = TP.normalizeText(c.raw, c.dialect);
    if (got === c.normalized) {
      pass++;
    } else {
      fail++;
      failures.push(['normalize', c.raw + ' [' + c.dialect + ']',
        c.normalized, got]);
      console.log(`FAIL  normalize  ${c.raw} [${c.dialect}]`);
      console.log(`        py: ${c.normalized}`);
      console.log(`        js: ${got}`);
    }
  } catch (e) {
    fail++;
    failures.push(['normalize-err', c.raw, '', String(e)]);
    console.log(`ERROR normalize ${c.raw} — ${e.message}`);
  }
}
console.log(`[normalize] ${normExpected.length - failures.filter(f => f[0].startsWith('normalize')).length}/${normExpected.length} PASS`);

// ---- 2) فحص التوكنز --------------------------------------------------------
const expected = JSON.parse(
  readFileSync(join(HERE, 'expected_tokens.json'), 'utf-8'));
for (const item of expected) {
  if (item.error) {
    console.log(`SKIP  ${item.id} — بايثون نفسه أخفق: ${item.error}`);
    continue;
  }
  try {
    const res = await TP.prepareTextRich(item.text, 'manual',
      item.dialect, null);
    const toks = item.dialect === 'msa'
      ? TP.toksMs(res.text) : TP.toksEgy(res.text);
    const ids = TP.tokensToIds(toks);
    const okPrepared = res.text === item.prepared;
    const okTokens = JSON.stringify(toks) === JSON.stringify(item.tokens);
    const okIds = JSON.stringify(ids) === JSON.stringify(item.ids);
    if (okPrepared && okTokens && okIds) {
      pass++;
      console.log(`PASS  ${item.id}  (${ids.length} ids)`);
    } else {
      fail++;
      failures.push(['tokens', item.id, item.tokens, toks]);
      console.log(`FAIL  ${item.id}  prepared=${okPrepared} `
        + `tokens=${okTokens} ids=${okIds}`);
      if (!okPrepared) {
        console.log('  py:', item.prepared);
        console.log('  js:', res.text);
      }
    }
  } catch (e) {
    fail++;
    failures.push(['tokens-err', item.id, '', String(e)]);
    console.log(`ERROR ${item.id} — ${e.message}`);
  }
}

// تفاصيل أول فشل توكنز
for (const f of failures.filter(x => x[0] === 'tokens').slice(0, 2)) {
  console.log('\n--- تفاصيل', f[1], '---');
  const pyT = f[2] || [];
  const jsT = f[3] || [];
  for (let i = 0; i < Math.max(pyT.length, jsT.length); i++) {
    if (pyT[i] !== jsT[i]) {
      console.log(`أول اختلاف توكن عند ${i}: py=${JSON.stringify(pyT[i])}`
        + ` js=${JSON.stringify(jsT[i])}`);
      console.log('سياق py:', pyT.slice(Math.max(0, i - 4), i + 4));
      console.log('سياق js:', jsT.slice(Math.max(0, i - 4), i + 4));
      break;
    }
  }
}

console.log(`\n=== النتيجة: ${pass} PASS / ${fail} FAIL ===`);
process.exit(fail === 0 ? 0 : 1);
