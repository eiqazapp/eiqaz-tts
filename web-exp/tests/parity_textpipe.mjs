#!/usr/bin/env node
/**
 * parity_textpipe.mjs — اختبار تكافؤ خط النص JS مقابل بايثون
 * ====================================================================
 * يقارن توكن-بموكن وid-بid مخرجات textpipe.js مع المرجع المولَّد من
 * inference/infer.py نفسه (tests/expected_tokens.json) على النصوص
 * الذهبية — الإثبات الوحيد المطلوب لسلامة النقل.
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

// تحميل السكريبتات الكلاسيكية (تسجل على globalThis)
await import(join(ROOT, 'models', 'textpipe_data.js'));
await import(join(ROOT, 'textpipe.js'));
const TP = globalThis.TextPipe;

const expected = JSON.parse(
  readFileSync(join(HERE, 'expected_tokens.json'), 'utf-8'));

let pass = 0, fail = 0;
const failures = [];

for (const item of expected) {
  if (item.error) {
    console.log(`SKIP  ${item.id} — بايثون نفسه أخفق: ${item.error}`);
    continue;
  }
  try {
    const res = TP.prepareTextRich(item.text, 'never', item.dialect, 'auto',
      null);
    // نفس مسار الإنتاج: تمرير أفعال العلامات والهياكل native للمرمِّز
    const acts = (res.qafActions && Object.keys(res.qafActions).length)
      ? res.qafActions : null;
    const nat = (res.qafNative && res.qafNative.length)
      ? new Set(res.qafNative) : null;
    let toks;
    if (item.dialect === 'msa') {
      toks = TP.msaSynthesisTokens(res.text, 'auto', acts);
    } else {
      toks = TP.toksEgy(res.text, 'auto', acts, nat);
    }
    const ids = TP.tokensToIds(toks);

    const okPrepared = res.text === item.prepared;
    const okTokens = JSON.stringify(toks) === JSON.stringify(item.tokens);
    const okIds = JSON.stringify(ids) === JSON.stringify(item.ids);
    if (okPrepared && okTokens && okIds) {
      pass++;
      console.log(`PASS  ${item.id}  (${ids.length} ids)`);
    } else {
      fail++;
      failures.push({
        id: item.id,
        prepared_match: okPrepared,
        tokens_match: okTokens,
        ids_match: okIds,
        js_prepared: res.text,
        py_prepared: item.prepared,
        js_tokens: toks,
        py_tokens: item.tokens,
      });
      console.log(`FAIL  ${item.id}  prepared=${okPrepared} `
        + `tokens=${okTokens} ids=${okIds}`);
    }
  } catch (e) {
    fail++;
    failures.push({ id: item.id, error: String(e) });
    console.log(`ERROR ${item.id} — ${e.message}`);
  }
}

// تفاصيل أول فشلين للتشخيص
for (const f of failures.slice(0, 2)) {
  console.log('\n--- تفاصيل', f.id, '---');
  if (f.error) { console.log(f.error); continue; }
  if (!f.prepared_match) {
    console.log('PY :', f.py_prepared);
    console.log('JS :', f.js_prepared);
  }
  if (!f.tokens_match) {
    const pyT = f.py_tokens || [];
    const jsT = f.js_tokens || [];
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
}

console.log(`\n=== النتيجة: ${pass} PASS / ${fail} FAIL ===`);
process.exit(fail === 0 ? 0 : 1);
