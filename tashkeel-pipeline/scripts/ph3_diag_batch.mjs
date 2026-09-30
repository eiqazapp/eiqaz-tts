// ph3_engine.mjs — Path C tashkeel engine v5 (guide v2.1, validated, resumable)
// Usage: node scripts/ph3_engine.mjs <input.json> <output.jsonl> [batch_size]
// input.json: [{"id": "...", "text": "...", ...}]
// output.jsonl lines: {"id","raw","converted","out","attempts","status",
//   "checks":{...},"fixes":{al_kasra,taa_sukun_strip,tier_a,rabbena},...}
// Gates: skeleton+punct exact (two-sided diacritic strip), tanween closed
// list (191 structures, prefix-strippable), ة carries no mark (exempt inside
// Tier A spans — قُوّةَ), density (anti-echo).
// v5 (guide v2.1): deterministic Tier A map + رَبِّنَا/بِرَبِّنَا enforcement
// AFTER al_kasra (map restores اَلْحَمْدُ), quarantine skip (6 units, §ل-5),
// letter-fix exceptions (صالة/ردّ — §ك-5), full re-validation after
// every enforcement step.
import ZAI from 'z-ai-web-dev-sdk';
import fs from 'node:fs';
import crypto from 'node:crypto';
import { loadMaps, applyTierA, applyRabbena, taaCheckWithExempt, restoreLetters }
  from './ph3_maps.mjs';

const [,, IN_PATH, OUT_PATH, BATCH_ARG] = process.argv;
const BATCH_SIZE = BATCH_ARG ? parseInt(BATCH_ARG, 10) : 50;
const MAX_SINGLE_RETRIES = 3;

const GUIDE_PATH = '/home/z/my-project/work/tashkeel_guide_v2.md';
const TANWEEN_LIST_PATH = '/home/z/my-project/work/tanween_closed_list.json';
const TIER_A_MAP_PATH = '/home/z/my-project/work/tier_a_map.json';
const LETTER_FIXES_PATH = '/home/z/my-project/work/ph1_letter_fixes.json';
const QUARANTINE_PATH = '/home/z/my-project/work/ph1_quarantine.json';
const GUIDE = fs.readFileSync(GUIDE_PATH, 'utf8');
const TANWEEN_ADVERBS = new Set(
  JSON.parse(fs.readFileSync(TANWEEN_LIST_PATH, 'utf8')).words);
const MAPS = loadMaps(TIER_A_MAP_PATH);
const LETTER_FIXES = new Map(
  JSON.parse(fs.readFileSync(LETTER_FIXES_PATH, 'utf8')).fixes
    .map((f) => [f.id, f]));
const QUARANTINED = new Set(
  JSON.parse(fs.readFileSync(QUARANTINE_PATH, 'utf8')).units
    .map((u) => u.utt));

const STRIP = /[\u064B-\u0652\u0653-\u0655\u0670]/g;
const convertPunct = (t) => t.replace(/\u060C/g, ',').replace(/\u061F/g, '?');

const FEWSHOT = `أمثلة مرجعية للمطلوب (خام ← مشكول):
- خام: انه بقى بيزنس مودل ناجح ومش بس بيوفر فرص عمل للشباب
- مشكول: اِنُّهْ بَقَى بِيزْنِسْ مُودِلْ نَاجِحْ وِمُشْ بَسْ بِيُوَفِّرْ فُرَصْ عَمَلْ لِلشَّبَابْ

- خام: هي دي فكرة اوية جدا.
- مشكول: هِيَ دِي فِكْرَة اوِّيَة جِدَاً.

- خام: بيقول مهمة اوي إن الواحد ما يعزلش نفسه وقت الأزمة دي
- مشكول: بِيْقُولْ مُهِمَّة اوِّي إِنَّ الْوَاحِدْ مَا يِعْزِلْشْ نَفْسُهْ وَقْتْ اِلْأَزْمَة دِيْ.

- خام: التسرع ده ممكن يخليك تشتري بسعر أغلى
- مشكول: اِلتَّسْرِعْ دَهْ مُمْكِنْ يِخْلِيكْ تِشْتَرِي بِسْعْرْ أَغْلَى

`;

const SYSTEM_PROMPT = FEWSHOT + GUIDE + `

---

## عقد الإخراج (إلزامي — يُنفَّذ حرفيًا)

- ستستقبل جملًا مرقّمة (1. 2. 3. ...). أعد كل جملة مشكولةً بالكامل كما تُنطق
  بالمصرية، بنفس الترقيم الرقمي وبنفس الترتيب — سطر واحد لكل جملة، ولا شيء غيرها.
- ممنوع: أي شرح أو مقدمة أو خاتمة أو علامات تنصيص أو تنسيق ماركداون — أسطر
  مرقّمة فقط.
- انسخ كل حرف ومسافة ورقم ورمز ترقيم كما ورد حرفيًا (بما فيها . , ? ! : " ( ) %
  والأرقام العربية واللاتينية والتطويل) وأضِف الحركات فقط وفق الدليل أعلاه.
- كل «ال» التعريف في بداية كلمة تُرسم «اِلْ» بكسرة تحت الألف دائمًا:
  اِلنَّاسْ، اِلْمَصَادِرْ، اِلتَّقْرِيرْ (بالشدة على الحرف الشمسي).
- التشكيل الجزئي الموجود في الجملة تلميح نطق فقط؛ قرارك النهائي يتبع الدليل.
- التنوين: فقط على الظروف المنصوصة في قائمة §ب (يُرسم بعد الألف: جِدَاً،
  فِعْلَاً، طَبْعَا، تَمَامَا). أي كلمة أخرى تنتهي بسكون أو بلا حركة.`;

// ---------------- validators ----------------
function firstDiff(a, b) {
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) if (a[i] !== b[i]) return i;
  return a.length === b.length ? -1 : n;
}

function wordAt(s, i) {
  let st = s.lastIndexOf(' ', i); st = st < 0 ? 0 : st + 1;
  let en = s.indexOf(' ', i); en = en < 0 ? s.length : en;
  return s.slice(st, en);
}

function skeletonCheck(out, convertedRaw) {
  const a = out.replace(STRIP, '');
  const b = convertedRaw.replace(STRIP, '');
  if (a === b) return { ok: true };
  const n = Math.min(a.length, b.length);
  const diffPos = [];
  for (let i = 0; i < n; i++) if (a[i] !== b[i]) diffPos.push(i);
  const lenNote = a.length === b.length ? '' :
    ` (طول الهيكل مختلف: مخرجك ${a.length} مقابل ${b.length} في الأصل — يوجد حرف زائد أو ناقص أو إعادة ترتيب)`;
  const words = [];
  for (const i of diffPos) {
    const w = wordAt(b, i);
    if (w && w.trim() && !words.includes(w)) words.push(w);
    if (words.length >= 4) break;
  }
  const i = diffPos[0] ?? 0;
  const ctx = (s, i) => `${s.slice(Math.max(0, i - 12), i)}[${s[i] ?? '∅'}]${s.slice(i + 1, i + 13)}`;
  const wordList = words.length
    ? `الكلمات المطلوبة بأحرفها حرفيًا: ${words.map((w) => `«${w}»`).join(' و')} — انقل أحرف كل منها كما هي بلا أي زيادة أو نقصان أو تبديل أو إعادة ترتيب (أضف الحركات فقط)، حتى لو بدت لك أخطاءً إملائية عامية.`
    : '';
  return {
    ok: false,
    reason: `اختلاف الهيكل عند الموضع ${i}${lenNote}: الأصل «${ctx(b, i)}» مقابل مخرجك «${ctx(a, i)}». ${wordList}`,
  };
}

function taaMarbutaCheck(out) {
  // §ج-2: ة carries no HARAKA (fatha/damma/kasra/shadda/tanween) — sukun is
  // tolerated (phoneme-neutral) and stripped later by post-fix.
  const m = out.match(/\u0629[\u064B-\u064E\u064F\u0650\u0651]/);
  if (m) {
    const i = out.indexOf(m[0]);
    return { ok: false, reason: `التاء المربوطة تحمل حركة في «${out.slice(Math.max(0, i - 10), i + 12)}» — القاعدة §ج-2: ة في نهاية الكلمة بلا فتحة/ضمة/كسرة/شدة/تنوين. احذف حركتها.` };
  }
  return { ok: true };
}

function tanweenCheck(out) {
  if (/[\u064C\u064D]/.test(out))
    return { ok: false, reason: 'يحوي تنوين ضم أو كسر — ممنوع مطلقًا. استخدم السكون أو حركة عادية.' };
  for (const w of out.split(/\s+/)) {
    if (w.includes('\u064B')) {
      let skel = w.replace(STRIP, '').replace(/[^\u0621-\u064A\u0640]/g, '');
      if ((skel[0] === 'و' || skel[0] === 'ف') && skel.length > 3) skel = skel.slice(1);
      if (!TANWEEN_ADVERBS.has(skel))
        return { ok: false, reason: `تنوين على كلمة خارج القائمة المغلقة: «${w}». التنوين مسموح فقط للظروف المنطوقة (جدا، فعلا، طبعا، تماما، مثلا، غالبا، دايما، أصلا، تقريبا، خصوصا، أولا، أبدا، فورا، أحيانا...). احذف التنوين واجعل نهاية الكلمة سكونًا أو بلا حركة.` };
    }
  }
  return { ok: true };
}

function densityCheck(out) {
  const words = out.split(/\s+/).filter((w) => /[\u0621-\u064A]/.test(w));
  const marks = (out.match(/[\u064B-\u0652\u0653-\u0655\u0670]/g) || []).length;
  const need = Math.max(5, Math.ceil(1.2 * words.length));
  if (marks < need)
    return { ok: false, reason: `المخرج شبه خالٍ من الحركات (${marks} حركة لـ${words.length} كلمة والحد الأدنى ${need}) — هذه مهمة تشكيل كامل وليست نسخًا. أضف الحركات لكل كلمة وفق الدليل.` };
  return { ok: true };
}

function validate(out, convertedRaw) {
  const sk = skeletonCheck(out, convertedRaw);
  if (!sk.ok) return sk;
  const tw = tanweenCheck(out);
  if (!tw.ok) return tw;
  const tt = taaMarbutaCheck(out);
  if (!tt.ok) return tt;
  return densityCheck(out);
}

// Final validation after Tier A / Rabbena enforcement: same gates, but the
// taa-marbuta check exempts map-replaced tokens (قُوّةَ is the only ة-with-
// haraka frozen form — guide §ك exception).
function validateFinal(out, convertedRaw, exemptTokenIdx) {
  const sk = skeletonCheck(out, convertedRaw);
  if (!sk.ok) return sk;
  const tw = tanweenCheck(out);
  if (!tw.ok) return tw;
  const tt = taaCheckWithExempt(out, exemptTokenIdx);
  if (!tt.ok) return tt;
  return densityCheck(out);
}

// Deterministic post-fixes (all diacritic-only, skeleton-safe, logged):
//  1) al_kasra: kasra under word-initial bare/fatha ال (article) and under
//     و/ف/ب prefix + ال — with corpus-derived exclusion skeletons.
//  2) taa_sukun_strip: remove sukun from ة (cosmetic normalization §ج-2).
const AL_EXCLUDE = new Set(['ال', 'الف', 'الفين', 'الله']);
const WAL_EXCLUDE = new Set(['والله', 'والد', 'والدة', 'والدي', 'والدك', 'والدنا']);
const FAL_EXCLUDE = new Set(['فالح', 'فالحة', 'فالحين']);
const BAL_EXCLUDE = new Set(['بال', 'بالي', 'بالك', 'باله', 'بالنا', 'بالم',
  'بالمي', 'بالكم', 'بالهم']);

function applyPostFixes(out) {
  const fixes = { al_kasra: 0, taa_sukun_strip: 0 };
  const words = out.split(/(\s+)/); // keep separators
  for (let i = 0; i < words.length; i += 2) {
    let w = words[i];
    if (!w) continue;
    const skel = w.replace(STRIP, '').replace(/[^\u0621-\u064A\u0640]/g, '');
    // word-initial: optional و/ف/ب prefix + optional fatha + ا + optional fatha + ل
    const m = w.match(/^([وفب]?)([\u064E]?)ا([\u064E]?)ل/);
    if (m) {
      const pfx = m[1];
      const pfxMark = m[2];
      const alMark = m[3];
      let apply = false;
      if (!pfx) {
        apply = !AL_EXCLUDE.has(skel);
      } else {
        const excl = pfx === 'و' ? WAL_EXCLUDE : pfx === 'ف' ? FAL_EXCLUDE
          : BAL_EXCLUDE;
        apply = !excl.has(skel);
      }
      if (apply) {
        if (!pfx && pfxMark === '' && alMark === '') {          // ال -> اِل
          w = 'ا\u0650' + w.slice(1); fixes.al_kasra++;
        } else if (!pfx && pfxMark === '' && alMark === '\u064E') { // اَل -> اِل
          w = 'ا\u0650' + w.slice(2); fixes.al_kasra++;
        } else if (pfx && pfxMark === '' && alMark === '') {     // وال -> وِال
          w = pfx + '\u0650' + w.slice(1); fixes.al_kasra++;
        } else if (pfx && pfxMark === '\u064E' && alMark === '\u064E') { // وَاَل -> وِال
          w = pfx + '\u0650' + 'ا' + w.slice(4); fixes.al_kasra++;
        } else if (pfx && pfxMark === '\u064E') {                // وَال -> وِال
          w = pfx + '\u0650' + w.slice(2); fixes.al_kasra++;
        } else if (pfx && pfxMark === '' && alMark === '\u064E') { // واَل -> وِال
          w = pfx + '\u0650' + 'ا' + w.slice(3); fixes.al_kasra++;
        }
      }
    }
    // strip sukun from ة (cosmetic)
    const stripped = w.replace(/\u0629\u0652/g, () => { fixes.taa_sukun_strip++; return '\u0629'; });
    words[i] = stripped;
  }
  return { out: words.join(''), fixes };
}

// ---------------- LLM calls ----------------
async function callLLM(zai, messages) {
  const MAX_API_TRIES = 5;
  let lastErr;
  for (let t = 1; t <= MAX_API_TRIES; t++) {
    try {
      const completion = await zai.chat.completions.create({
        messages,
        thinking: { type: 'disabled' },
      });
      const content = completion.choices?.[0]?.message?.content;
      if (!content || !content.trim()) throw new Error('empty completion');
      return { content, model: completion.model ?? null };
    } catch (e) {
      lastErr = e;
      const is429 = /429|Too many/i.test(String(e.message));
      if (!is429 || t === MAX_API_TRIES) throw e;
      const wait = Math.min(60000, 3000 * 2 ** (t - 1)); // 3s,6s,12s,24s
      console.log(`    [429] waiting ${wait / 1000}s before retry ${t + 1}/${MAX_API_TRIES}`);
      await new Promise((r) => setTimeout(r, wait));
    }
  }
  throw lastErr;
}

function parseNumbered(text, expectedN) {
  let t = text.trim();
  t = t.replace(/^```[a-z]*\n?/i, '').replace(/\n?```\s*$/i, '');
  const lines = t.split(/\r?\n/).filter((l) => l.trim());
  const byNum = new Map();
  const plain = [];
  const RE = /^\s*(\d+|[٠-٩]+)\s*[.\-)]\s*(.+)$/;
  for (const l of lines) {
    const m = l.match(RE);
    if (m) {
      const n = m[1].match(/[٠-٩]/)
        ? [...m[1]].reduce((a, c) => a * 10 + '٠١٢٣٤٥٦٧٨٩'.indexOf(c), 0)
        : parseInt(m[1], 10);
      byNum.set(n, m[2].trim());
    } else plain.push(l.trim());
  }
  if (byNum.size >= Math.max(1, Math.floor(expectedN * 0.5))) {
    const arr = [];
    for (let i = 1; i <= expectedN; i++) arr.push(byNum.get(i) ?? null);
    return { mode: 'numbered', lines: arr };
  }
  if (plain.length === expectedN) return { mode: 'plain', lines: plain };
  return { mode: 'unparsed', lines: null };
}

async function tashkeelBatch(zai, items) {
  const numbered = items.map((p, i) => `${i + 1}. ${p.converted}`).join('\n');
  const { content, model } = await callLLM(zai, [
    { role: 'system', content: SYSTEM_PROMPT },
    { role: 'user', content: `المهمة: إضافة التشكيل (الفتحة/الضمة/الكسرة/السكون/الشدة) إلى الجمل التالية كما تُنطق بالمصرية — هذا تحويل وليس نسخًا للنص كما ورد.

قاعدتان حاسمتان:
1) انقل أحرف كل كلمة كما هي حتى لو بدت لك خطأ إملائيًا عاميًا («ده» تُكتب «دَهْ» وليست دي أو ديه؛ «اوي» تُكتب «اوِّي»؛ «بتاعت» تبقى «بِتَاعِتْ») — يُمنع إضافة أو حذف أو تبديل أي حرف.
2) كل «ال» أول الكلمة تُرسم «اِلْ» بكسرة تحت الألف: اِلنَّاسْ، اِلْمَصَادِرْ، اِلتَّقْرِيرْ — وبعد و/ب/ف: وِالْقَلَقْ، بِالزَّبَطْ.

وفحص ثالث حاسم: انقل علامات الترقيم كما وردت حرفيًا — إن جاءت الفاصلة لاتينية «,» فأبقها لاتينية (لا تبدلها بـ«،» العربية)، وإن جاء الاستفهام «?» لاتينيًا فأبقه كما هو.

شكّل الجمل التالية بنفس ترقيمها وترتيبها (سطر مرقّم لكل جملة، بلا أي شرح):

${numbered}` },
  ]);
  return { content, model };
}

async function tashkeelSingle(zai, item, prevOut, reason, attempt) {
  const { content } = await callLLM(zai, [
    { role: 'system', content: SYSTEM_PROMPT },
    { role: 'user', content: `المهمة: إضافة التشكيل إلى هذه الجملة فقط كما تُنطق بالمصرية — تحويل وليس نسخًا. أعد سطرًا واحدًا بلا رقم وبلا أي شرح، وانقل أحرف الكلمات كما هي بلا أي تصحيح إملائي:

${item.converted}

ملاحظة (محاولة ${attempt}): مخرجك السابق رُفض لأن: ${reason}${attempt > 2 ? '\n(تجاهل محاولاتك السابقة تمامًا وابدأ من الصفر بهذه الجملة فقط.)' : `\nالمخرج السابق كان: ${prevOut ?? '(فارغ)'}`}
صحّح المخالفة المذكورة فقط، واحتفظ بكل الحروف والترقيم والمسافات كما هي (الفاصلة اللاتينية , والاستفهام ? يبقيان كما وردا)، وأضف الحركات وفق الدليل (كل «ال» أول كلمة = اِلْ بكسرة تحت الألف).` },
  ]);
  return content.trim();
}

// ---------------- main ----------------
async function main() {
  const input0 = JSON.parse(fs.readFileSync(IN_PATH, 'utf8'));
  // Letter-fix exceptions (guide v2.1 §ك-5 — approved 2026-09-30): swap the
  // transcription error BEFORE anything else; skeleton reference = corrected
  // text; the swap is recorded per-unit (letter_fix + orig_text).
  const input = input0.map((p) => {
    const fix = LETTER_FIXES.get(p.id);
    if (!fix) return p;
    return { ...p, text: fix.corrected, orig_text: p.text,
             letter_fix: { from: fix.word_before, to: fix.word_after,
                           reason: fix.reason, approved: fix.approved } };
  });
  const done = new Set();
  if (fs.existsSync(OUT_PATH)) {
    for (const line of fs.readFileSync(OUT_PATH, 'utf8').split('\n')) {
      if (!line.trim()) continue;
      try { done.add(JSON.parse(line).id); } catch {}
    }
  }
  const todoAll = input.filter((p) => !done.has(p.id));
  // Quarantine (§ل-5 — approved 2026-09-30): quarantined units get a record
  // but NO automated tashkeel, no LLM call, no use in any output.
  const quarantined = todoAll.filter((p) => QUARANTINED.has(p.id));
  const todo = todoAll.filter((p) => !QUARANTINED.has(p.id));
  console.log(`input=${input.length} done=${done.size} quarantined=${quarantined.length} todo=${todo.length} batch=${BATCH_SIZE} tierA=${MAPS.meta.n_entries}forms/${MAPS.meta.total_corpus_positions}pos`);
  if (todo.length === 0 && quarantined.length === 0) { console.log('nothing to do'); return; }

  const zai = await ZAI.create();
  let modelSeen = null;
  const out = fs.createWriteStream(OUT_PATH, { flags: 'a' });

  for (const q of quarantined) {
    out.write(JSON.stringify({
      id: q.id, raw: q.orig_text ?? q.text,
      converted: convertPunct(q.orig_text ?? q.text),
      out: null, attempts: 0, status: 'quarantined', fixes: {}, checks: {},
      reason: 'حجر معتمد من المستخدم 2026-09-30 (§ل-5) — لا استخدام آلي حتى قرار حالة بحالة',
    }) + '\n');
    console.log(`  ${q.id}: quarantined (skipped, no LLM)`);
  }

  const DIAG = [];
  for (let b0 = 0; b0 < Math.min(todo.length, BATCH_SIZE); b0 += BATCH_SIZE) {
    const batch = todo.slice(b0, b0 + BATCH_SIZE).map((p) => ({
      ...p, converted: convertPunct(p.text),
    }));
    console.log(`batch ${Math.floor(b0 / BATCH_SIZE) + 1}: units ${b0 + 1}-${b0 + batch.length}`);
    let parsed = null, batchError = null;
    let DIAG_RAW = null;
    try {
      const { content, model } = await tashkeelBatch(zai, batch);
      modelSeen = model ?? modelSeen;
      DIAG_RAW = content;
      globalThis.__DIAG_RAW = content;
      parsed = parseNumbered(content, batch.length);
    } catch (e) { batchError = e.message; DIAG_RAW = String(e.message); }
    if (!parsed || !parsed.lines) {
      console.log(`  batch-level parse failure: ${batchError ?? 'unparsed'} -> per-item fallback`);
    }

    for (let i = 0; i < batch.length; i++) {
      const item = batch[i];
      let punctUnswap = 0;
      const unswapPunct = (t) => {
        if (!t) return t;
        const n = (t.match(/[\u060C\u061F]/g) || []).length;
        if (n > 0) punctUnswap += n;
        return t.replace(/\u060C/g, ',').replace(/\u061f/g, '?');
      };
      // letter-restore (§أ-1/§و enforcement): undo the model's orthographic
      // normalization (alef-hamza forms / digit script / ة-ه) back to the RAW
      // letters before validation — so retries target REAL issues.
      let letterFixes = 0;
      const restore = (t) => {
        if (!t) return t;
        const hr = restoreLetters(t, item.converted);
        if (hr.count > 0) { letterFixes += hr.count; return hr.out; }
        return t;
      };
      let outText = restore(unswapPunct(parsed?.lines?.[i] ?? null));
      let attempts = outText ? 1 : 0;
      let v = outText ? validate(outText, item.converted) : { ok: false, reason: batchError ?? 'تعذر التحليل ضمن الدفعة' };
      DIAG.push({ id: item.id, batchLine: parsed?.lines?.[i] ?? null,
                  outAfterRestore: outText, ok: v.ok,
                  reason: v.ok ? null : v.reason,
                  letterRestore: letterFixes, punctUnswap });
      if (true) { // DIAG: skip retries entirely
        const rec = {
          id: item.id, raw: item.text, converted: item.converted,
          out: v.ok ? outText : (outText ?? null), attempts,
          status: v.ok ? 'ok' : 'failed', fixes: {},
          checks: {}, reason: v.ok ? null : v.reason,
        };
        out.write(JSON.stringify(rec) + '\n');
        console.log(`  ${item.id}: ${rec.status} (DIAG no-retry)`);
        continue;
      }
      while (!v.ok && attempts <= MAX_SINGLE_RETRIES) {
        attempts++;
        try {
          outText = restore(unswapPunct(await tashkeelSingle(zai, item, outText, v.reason, attempts - 1)));
        } catch (e) { outText = null; v = { ok: false, reason: `API error: ${e.message}` }; continue; }
        v = validate(outText, item.converted);
      }
      // deterministic post-fixes (diacritic-only, skeleton-safe)
      let fixes = punctUnswap ? { punct_unswap: punctUnswap } : {};
      if (letterFixes) fixes.letter_restore = letterFixes;
      if (v.ok && outText) {
        const f = applyPostFixes(outText);
        if (f.fixes.al_kasra || f.fixes.taa_sukun_strip) {
          const cand = f.out;
          const v2 = validate(cand, item.converted); // safety re-check
          if (v2.ok) { outText = cand; fixes = { ...fixes, ...f.fixes }; }
        }
      }
      // Tier A + Rabbena enforcement (guide v2.1 §ك) — AFTER al_kasra so the
      // map restores اَلْحَمْدُ if the fixer rewrites اَلْ to اِلْ. Full
      // re-validation afterwards (taa exempted on replaced tokens only); on
      // unexpected failure we keep the pre-map text (already gate-passed).
      if (v.ok && outText) {
        const ta = applyTierA(outText, MAPS);
        const rb = applyRabbena(ta.out, MAPS);
        if (rb.out !== outText) {
          const exempt = new Set([...ta.tokenIdx, ...rb.tokenIdx]);
          const v3 = validateFinal(rb.out, item.converted, exempt);
          if (v3.ok) {
            if (ta.forms.length) fixes.tier_a = ta.forms;
            if (rb.forms.length) fixes.rabbena = rb.forms;
            outText = rb.out;
          } else {
            console.log(`    [map-revert] ${item.id}: ${v3.reason.slice(0, 90)}`);
          }
        }
      }
      const rec = {
        id: item.id, raw: item.text, converted: item.converted,
        ...(item.letter_fix ? { orig_text: item.orig_text,
                                letter_fix: item.letter_fix } : {}),
        out: v.ok ? outText : (outText ?? null),
        attempts, status: v.ok ? 'ok' : 'failed',
        fixes,
        checks: { skeleton: v.ok, tanween: v.ok, taa: v.ok, density: v.ok },
        reason: v.ok ? null : v.reason,
      };
      out.write(JSON.stringify(rec) + '\n');
      console.log(`  ${item.id}: ${rec.status} (attempts=${attempts}${Object.keys(fixes).length ? `, fixes=${JSON.stringify(fixes)}` : ''})`);
    }
  }
  out.end();

  const meta = {
    engine: 'ph3_engine.mjs v5 (guide v2.1 — approved 2026-09-30)',
    guide_path: GUIDE_PATH,
    guide_sha256: crypto.createHash('sha256').update(GUIDE).digest('hex'),
    prompt_sha256: crypto.createHash('sha256').update(SYSTEM_PROMPT).digest('hex'),
    tanween_list_path: TANWEEN_LIST_PATH,
    tanween_list_sha256: crypto.createHash('sha256')
      .update(fs.readFileSync(TANWEEN_LIST_PATH)).digest('hex'),
    tier_a_map_path: TIER_A_MAP_PATH,
    tier_a_map_sha256: crypto.createHash('sha256')
      .update(fs.readFileSync(TIER_A_MAP_PATH)).digest('hex'),
    tier_a_meta: MAPS.meta,
    letter_fixes_path: LETTER_FIXES_PATH,
    quarantine_path: QUARANTINE_PATH,
    n_quarantined_written: quarantined.length,
    model: modelSeen, batch_size: BATCH_SIZE,
    max_single_retries: MAX_SINGLE_RETRIES,
    finished: new Date().toISOString(), punct_rule: 'U+060C->, U+061F->?',
    gates: ['skeleton+punct two-sided', 'tanween closed list 191 structures',
            'taa no-haraka (sukun stripped; Tier A spans exempt — قُوّةَ)',
            'density>=1.2/w floor5',
            'post-fixes: al_kasra + taa_sukun_strip',
            'enforcement: letter_restore (§أ-1/§و) + Tier A map + rabbena + re-validation'],
  };
  fs.writeFileSync(OUT_PATH.replace(/\.jsonl?$/, '') + '_diag.json',
    JSON.stringify({ rawBatchContent: globalThis.__DIAG_RAW ?? null, nLines: DIAG.length,
                     diag: DIAG }, null, 1));
  fs.writeFileSync(OUT_PATH.replace(/\.jsonl?$/, '') + '_meta.json',
    JSON.stringify(meta, null, 1));
}

main().catch((e) => { console.error('FATAL:', e); process.exit(1); });
