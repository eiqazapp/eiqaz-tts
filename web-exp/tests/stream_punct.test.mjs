#!/usr/bin/env node
/**
 * stream_punct.test.mjs — اختبار انحدار حدود الجمل في محاكاة البث
 * ====================================================================
 * قبل [EIQAZ-PUNCT-BATCH]: القصّ عند نهايات الجمل فقط (.!؟?…) —
 *   نصُّ الفواصل «،» وحده لا يُنتج أي جملة كاملة أثناء الوصول → لا بث.
 * بعد: الحدود كل علامات الترقيم (.!؟?… + ،؛,;) — كما طلب المستخدم.
 *
 * يستخرج الـregex الفعلي من app.js نفسه (يفشل إن أُعيد القصر)، ثم يحاكي
 * حلقة وصول الدفعات حرفيًا كما في startStream ويطابق النتائج المتوقعة.
 *
 * التشغيل:  node web-exp/tests/stream_punct.test.mjs
 * الخروج:   0 = PASS كامل · 1 = فشل حالة أو أكثر
 */
'use strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, '..');

const appSrc = readFileSync(join(ROOT, 'app.js'), 'utf-8');
const htmlSrc = readFileSync(join(ROOT, 'index.html'), 'utf-8');
const serveSrc = readFileSync(join(ROOT, 'serve.py'), 'utf-8');

let pass = 0, fail = 0;
function check(name, cond, extra) {
  if (cond) { pass++; console.log('  [PASS] ' + name); }
  else { fail++; console.log('  [FAIL] ' + name + (extra ? ' — ' + extra : '')); }
}

console.log('=== 1) استخراج regex الحدود من app.js ===');
const mSplit = appSrc.match(/const parts = tail\.split\((\/[^\n]*?)\);/);
const mDone = appSrc.match(/const lastIsComplete = (\/[^\n]*?\/)\.test\(tail\);/);
check('regex القصّ موجود في app.js', !!mSplit);
check('regex الاكتمال موجود في app.js', !!mDone);
const splitRe = mSplit ? new RegExp(mSplit[1].slice(1, -1)) : /(?<=[.!؟?…])\s+/;
const doneRe = mDone ? new RegExp(mDone[1].slice(1, -1)) : /[.!؟?…]\s*$/;

for (const ch of ['،', '؛', ',', ';']) {
  check(`حدود الجملة تشمل «${ch}»`, splitRe.source.includes(ch));
  check(`اختبار الاكتمال يشمل «${ch}»`, doneRe.source.includes(ch));
}
for (const ch of ['.', '!', '؟', '?', '…']) {
  check(`حدود الجملة ما زالت تشمل «${ch}»`, splitRe.source.includes(ch));
}

console.log('=== 2) محاكاة حلقة وصول الدفعات (منطق startStream حرفيًا) ===');
function simulateStream(raw, batchWords = 6) {
  const words = raw.split(/\s+/).filter(Boolean);
  const batches = [];
  for (let i = 0; i < words.length; i += batchWords) {
    batches.push(words.slice(i, i + batchWords).join(' '));
  }
  let arrived = '';
  let synthPointer = 0;
  const pending = [];
  let streamedDuringArrival = 0;
  for (let i = 0; i < batches.length; i++) {
    arrived += (i ? ' ' : '') + batches[i];
    const tail = arrived.slice(synthPointer);
    const parts = tail.split(splitRe);
    const complete = parts.slice(0, -1);
    const lastIsComplete = doneRe.test(tail);
    if (complete.length || lastIsComplete) {
      if (lastIsComplete) complete.push(parts[parts.length - 1]);
      for (const c of complete) {
        const cc = c.trim();
        if (cc) { pending.push(cc); streamedDuringArrival++; }
        synthPointer += c.length + 1;
      }
    }
  }
  const leftover = arrived.slice(synthPointer).trim();
  if (leftover) pending.push(leftover);
  return { pending, streamedDuringArrival, leftover };
}

// نص المستخدم الأصلي (فواصل ، فقط — كان لا يُبثّ إطلاقًا)
const commaText =
  'النموذج اللغوي بيبعت النص على دفعات، والمحرك لازم يتكلم فورًا من أول دفعة، ' +
  'والصوت لازم يفضل شغال بدون انتظار، والمقاطعة لازم توقف كل حاجة في نفس اللحظة، ' +
  'وفي الآخر نقيس كل الأزمنة ونقارن النسخ.';
const r1 = simulateStream(commaText);
check('نص الفواصل «،» يُبثّ أثناء الوصول (كان 0)', r1.streamedDuringArrival === 5,
  'فعلي: ' + r1.streamedDuringArrival);
check('نص الفواصل: 5 جمل كاملة إجمالًا', r1.pending.length === 5,
  'فعلي: ' + r1.pending.length);
check('نص الفواصل: لا بقايا في الدفعة الأخيرة', r1.leftover === '');
check('الجملة الأولى سليمة', r1.pending[0] === 'النموذج اللغوي بيبعت النص على دفعات،');

// فواصل بلا نهاية ختامية — آخر جزء يُدفق في النهاية (سلوك أصيل)
const commaNoEnd = 'أول جزء، تاني جزء، تالت جزء، رابع جزء';
const r2 = simulateStream(commaNoEnd);
check('فواصل بلا نقطة ختامية: 3 أثناء الوصول', r2.streamedDuringArrival === 3,
  'فعلي: ' + r2.streamedDuringArrival);
check('الفاصلة الأخيرة (بلا ترقيم بعدها) تُدفق في النهاية', r2.pending.length === 4);

// نص النقاط (يجب أن يستمر بالعمل كما كان)
const dotText = 'خلينا نبدأ. الموضوع بسيط. جرب بنفسك. واسمع الفرق. قبل وبعد.';
const r3 = simulateStream(dotText);
check('نص النقاط: 5 جمل كما كان', r3.pending.length === 5 && r3.leftover === '');

// علامات {ق} تبقى داخل الجملة (لا قصّ داخلها)
const qafText = 'قسّمنا قطعة{ق} قماش على رقم{ج} أطفال وكل واحد قال{ء} شكرًا. القرآن{ق} فيه قصص{ق} كتير.';
const r4 = simulateStream(qafText);
check('نص {ق}: جملتان (عند النقطتين)', r4.pending.length === 2, JSON.stringify(r4.pending));
check('علامات {ق}/{ج}/{ء} سليمة داخل الجملة',
  r4.pending[0] === 'قسّمنا قطعة{ق} قماش على رقم{ج} أطفال وكل واحد قال{ء} شكرًا.'
  && r4.pending[1] === 'القرآن{ق} فيه قصص{ق} كتير.');

// فاصلة منقوطة وعلامة استفهام
const r5 = simulateStream('أهلاً، إزيك؟ عامل إيه؛ كويس.');
check('مزيج ، ؟ ؛ .: 4 جمل', r5.pending.length === 4, JSON.stringify(r5.pending));

// نص بلا أي ترقيم: دفعة واحدة في النهاية (سلوك أصيل)
const r6 = simulateStream('كلام من غير أي ترقيم خالص هنا');
check('بلا ترقيم: دفعة واحدة نهائية (كما كان)', r6.pending.length === 1
  && r6.streamedDuringArrival === 0);

console.log('=== 3) فتح قفل صوت الهاتف ([EIQAZ-AUDIO-WARMUP]) ===');
check('warmupAudio() معرّفة في app.js', /function warmupAudio\(\)/.test(appSrc));
check('توليد كامل يسخّن الصوت داخل النقر',
  /btnGenerate\.addEventListener\('click', \(\) => \{ warmupAudio\(\); generateFull\(\); \}\)/.test(appSrc));
check('محاكاة البث تسخّن الصوت داخل النقر',
  /btnStream\.addEventListener\('click', \(\) => \{ warmupAudio\(\); startStream\(\); \}\)/.test(appSrc));
check('الإعادة تسخّن الصوت داخل النقر',
  /btnReplay\.addEventListener\('click', \(\) => \{ warmupAudio\(\); replay\(\); \}\)/.test(appSrc));
check('احتياط play() عبر Web Audio في الوضع الكامل',
  /els\.fullAudio\.play\(\)\.catch\(\(\) => \{[\s\S]{0,120}?scheduleWave\(final, myEpoch\)/.test(appSrc));
check('index.html يحوي [EIQAZ-AUDIO-UNLOCK]',
  htmlSrc.includes('EIQAZ-AUDIO-UNLOCK') && htmlSrc.includes('__IQZ_AUDIO_UNLOCK__'));
check('الحقن قبل تحميل app.js',
  htmlSrc.indexOf('EIQAZ-AUDIO-UNLOCK') < htmlSrc.indexOf('src="app.js"'));

console.log('=== 4) serve.py (بانر فوري + كاش ذكي + favicon) ===');
check('line_buffering مفعّلة (بانر فوري في Git Bash)',
  serveSrc.includes('line_buffering=True'));
check('رؤوس كاش ذكية [IQZ_CACHE]', serveSrc.includes('IQZ_CACHE')
  && serveSrc.includes("max-age=86400' if _heavy else 'no-cache'"));
check('لا يعود no-store (كان يعيد تنزيل النماذج كل زيارة)',
  !serveSrc.includes("'no-store'"));
check('favicon.ico يُرد 204', /favicon\.ico[\s\S]{0,200}send_response\(204\)/.test(serveSrc));
check('الخادم ما زال متعدد الخيوط', serveSrc.includes('ThreadingHTTPServer'));
check('COOP/COEP ما زالت تُرسَل', serveSrc.includes('Cross-Origin-Opener-Policy'));

console.log('');
console.log('=== النتيجة: ' + pass + ' PASS / ' + fail + ' FAIL ===');
process.exit(fail ? 1 : 0);
