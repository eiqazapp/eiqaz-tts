// ph3_engine_v53.mjs — Path C tashkeel engine v5.3 (guide v2.1, validated, resumable)
// Usage: node scripts/ph3_engine_v53.mjs <input.json> <output.jsonl> [batch_size]
//
// v5.3 (2026-10-01, smoke-evidence change): FINAL INDIVIDUAL DRAIN. The v5.2
// smoke comparison vs the approved v5 run showed identical quality metrics on
// passing units (ratio median 0.877 = approved) and 3.29x call economy, BUT
// weaker recovery of hard units (4/13 failed vs 1/13 approved): batched retry
// rounds alone did not break through units where the LLM returns the sentence
// undiacritized (density gate). v5.3 keeps the v5.2 batched retry rounds for
// the first two rounds and spends the unit's LAST budget round as a v5-style
// FOCUSED individual retry (tashkeelSingle — single sentence, one line, no
// numbering; byte-identical wording to the approved v5 engine's single retry).
// Per-unit budget is unchanged (max 4 attempts); only the scheduling of the
// last round differs. SYSTEM_PROMPT and all validators/gates remain
// byte-identical to v5/v5.1/v5.2 (sha256 60130f6b…).
// input.json: [{"id": "...", "text": "...", ...}]
// output.jsonl lines: {"id","raw","converted","out","attempts","status",
//   "checks":{...},"fixes":{al_kasra,taa_sukun_strip,tier_a,rabbena},...}
// Gates: skeleton+punct exact (two-sided diacritic strip), tanween closed
// list (191 structures, prefix-strippable), ة carries no mark (exempt inside
// Tier A spans — قُوّةَ), density (anti-echo).
// v5.1 (2026-09-30, full-run diagnostic): tanweenCheck و/ف prefix strip
// is now conditional — the full skeleton is checked against the closed list
// FIRST, and the prefix is stripped only if the stripped form is in the list.
// Bug symptom (v5): sanctioned ف-initial adverbs (فعلا، فورا) lost their ف
// and were wrongly rejected, contradicting the approved list (§ب) — the list
// itself contains فعلا/فورا/وفعلا. No other logic touched.
// v5.2 (2026-09-30, economy change — user-approved direction): RETRY
// REBATCHING. Failed units are no longer retried one-by-one (each retry paid
// the full ~11k-token system prompt). Instead, failures accumulate in a
// retry queue and are re-sent TOGETHER in retry-batches of RETRY_BATCH_SIZE
// (default 5) with per-sentence feedback (previous rejected output + reason).
// SYSTEM_PROMPT is byte-identical to v5.1 (verified via prompt_sha256 vs the
// approved smoke meta 60130f6b…). All validators, Tier A / rabbena
// enforcement, letter-restore, post-fixes, quarantine and resume semantics
// are unchanged. Per-unit budget unchanged: 1 first-pass + up to 3 retry
// rounds (attempts max 4; parse-failure units max 4 rounds as in v5.1).
// Token accounting: usage-based when the backend exposes it, plus char
// counts and call counts for estimation otherwise.
import fs from 'node:fs';
import crypto from 'node:crypto';
import { loadMaps, applyTierA, applyRabbena, taaCheckWithExempt, restoreLetters }
  from './ph3_maps.mjs';

const [,, IN_PATH, OUT_PATH, BATCH_ARG] = process.argv;
const BATCH_SIZE = BATCH_ARG ? parseInt(BATCH_ARG, 10) : 50;
const RETRY_BATCH_SIZE = parseInt(process.env.RETRY_BATCH || '5', 10);
const MAX_RETRY_ROUNDS = 3; // same per-unit budget as v5.1 MAX_SINGLE_RETRIES

// PH3_WORK_DIR: portable work-dir override (e.g. GitHub Actions runner);
// unset → the canonical local layout. Same files, same bytes, same hashes.
// PH3_GUIDE_FILE: guide filename override (default tashkeel_guide_v2.md).
// Used by the Groq condensed-prompt experiment ONLY (on_demand TPM limits
// are below the full prompt); meta records prompt_mode loudly so condensed
// units are distinguishable and revertible at merge time.
const WORK_DIR = process.env.PH3_WORK_DIR || '/home/z/my-project/work';
const GUIDE_FILE = process.env.PH3_GUIDE_FILE || 'tashkeel_guide_v2.md';
const GUIDE_PATH = `${WORK_DIR}/${GUIDE_FILE}`;
const TANWEEN_LIST_PATH = `${WORK_DIR}/tanween_closed_list.json`;
const TIER_A_MAP_PATH = `${WORK_DIR}/tier_a_map.json`;
const LETTER_FIXES_PATH = `${WORK_DIR}/ph1_letter_fixes.json`;
const QUARANTINE_PATH = `${WORK_DIR}/ph1_quarantine.json`;
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
      // v5.1 bugfix: strip the و/ف conjunction prefix ONLY when the full
      // skeleton is not itself in the closed list AND the stripped form is.
      // (v5 wrongly stripped ف from فعلا/فورا themselves.)
      if (!TANWEEN_ADVERBS.has(skel)
          && (skel[0] === 'و' || skel[0] === 'ف') && skel.length > 3
          && TANWEEN_ADVERBS.has(skel.slice(1))) {
        skel = skel.slice(1);
      }
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
// Token accounting (v5.2): usage-based when the backend exposes it; char
// counts + call counts always recorded for estimation otherwise.
const TOK = {
  first_calls: 0, retry_calls: 0, rate_limited: 0,
  usage_present: 0, prompt_tokens: 0, completion_tokens: 0,
  chars: { system: SYSTEM_PROMPT.length, user_first: 0, out_first: 0,
           user_retry: 0, out_retry: 0 },
};

// TEST-ONLY hook (zero effect when env unset): offline dry tests load a fake
// LLM to exercise the retry-rebatching mechanics deterministically without
// spending API quota. Production must never set PH3_FAKE_LLM.
let FAKE = null;
if (process.env.PH3_FAKE_LLM) {
  FAKE = (await import(process.env.PH3_FAKE_LLM)).fakeLLM;
}

async function callLLM(zai, messages, kind) {
  const MAX_API_TRIES = 5;
  let lastErr;
  for (let t = 1; t <= MAX_API_TRIES; t++) {
    try {
      const completion = FAKE
        ? await FAKE(messages)
        : await zai.chat.completions.create({
            messages,
            thinking: { type: 'disabled' },
          });
      const content = completion.choices?.[0]?.message?.content;
      if (!content || !content.trim()) throw new Error('empty completion');
      // token accounting
      if (kind === 'retry') {
        TOK.retry_calls++;
        TOK.chars.user_retry += messages
          .filter((m) => m.role === 'user').map((m) => m.content).join('\n').length;
        TOK.chars.out_retry += content.length;
      } else {
        TOK.first_calls++;
        TOK.chars.user_first += messages
          .filter((m) => m.role === 'user').map((m) => m.content).join('\n').length;
        TOK.chars.out_first += content.length;
      }
      if (completion.usage && typeof completion.usage.total_tokens === 'number') {
        TOK.usage_present++;
        TOK.prompt_tokens += completion.usage.prompt_tokens ?? 0;
        TOK.completion_tokens += completion.usage.completion_tokens ?? 0;
        console.log(`    [tok] ${kind}#${TOK.first_calls + TOK.retry_calls} ` +
          `usage prompt=${completion.usage.prompt_tokens} ` +
          `completion=${completion.usage.completion_tokens}`);
      }
      return { content, model: completion.model ?? null };
    } catch (e) {
      lastErr = e;
      const is429 = /429|Too many/i.test(String(e.message));
      if (is429) TOK.rate_limited++;
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
  ], 'first');
  return { content, model };
}

// v5.2: retry-batch call — ONE call for up to RETRY_BATCH_SIZE failed units,
// with per-sentence feedback (previous rejected output + rejection reason).
// The system prompt and the output contract are identical to the first-pass
// batch call; only the user message carries the per-sentence feedback.
async function tashkeelRetryBatch(zai, entries) {
  const listed = entries.map((e, i) => {
    const prevNote = e.attempts >= 3
      ? 'المخرج السابق: (تجاهل محاولاتك السابقة تمامًا وابدأ من الصفر بهذه الجملة فقط.)'
      : `المخرج السابق (مرفوض): ${e.prevOut ?? '(فارغ)'}`;
    return `${i + 1}. الجملة: ${e.item.converted}\n   ${prevNote}\n   سبب الرفض: ${e.reason}`;
  }).join('\n\n');
  const { content, model } = await callLLM(zai, [
    { role: 'system', content: SYSTEM_PROMPT },
    { role: 'user', content: `المهمة: إضافة التشكيل (الفتحة/الضمة/الكسرة/السكون/الشدة) إلى الجمل التالية كما تُنطق بالمصرية — هذا تحويل وليس نسخًا للنص كما ورد. هذه جمل سبقت محاولة تشكيلها ورُفض مخرجها؛ لكل جملة أمامك مخرجها السابق المرفوض وسبب الرفض.

قاعدتان حاسمتان:
1) انقل أحرف كل كلمة كما هي حتى لو بدت لك خطأ إملائيًا عاميًا («ده» تُكتب «دَهْ» وليست دي أو ديه؛ «اوي» تُكتب «اوِّي»؛ «بتاعت» تبقى «بِتَاعِتْ») — يُمنع إضافة أو حذف أو تبديل أي حرف.
2) كل «ال» أول الكلمة تُرسم «اِلْ» بكسرة تحت الألف: اِلنَّاسْ، اِلْمَصَادِرْ، اِلتَّقْرِيرْ — وبعد و/ب/ف: وِالْقَلَقْ، بِالزَّبَطْ.

وفحص ثالث حاسم: انقل علامات الترقيم كما وردت حرفيًا — إن جاءت الفاصلة لاتينية «,» فأبقها لاتينية (لا تبدلها بـ«،» العربية)، وإن جاء الاستفهام «?» لاتينيًا فأبقه كما هو.

صحّح لكل جملة المخالفة المذكورة في سبب الرفض فقط، وأعد الجمل مشكولةً بالكامل بنفس الترقيم الرقمي وبنفس الترتيب — سطر واحد لكل جملة، ولا شيء غيره:

${listed}` },
  ], 'retry');
  return { content, model };
}

// v5.3: focused single-sentence retry — byte-identical user-message wording
// to the approved v5 engine's tashkeelSingle (proven to recover hard units in
// the approved smoke run: 3/4 hard units passed on this exact call shape).
async function tashkeelSingle(zai, item, prevOut, reason, attempt) {
  const { content, model } = await callLLM(zai, [
    { role: 'system', content: SYSTEM_PROMPT },
    { role: 'user', content: `المهمة: إضافة التشكيل إلى هذه الجملة فقط كما تُنطق بالمصرية — تحويل وليس نسخًا. أعد سطرًا واحدًا بلا رقم وبلا أي شرح، وانقل أحرف الكلمات كما هي بلا أي تصحيح إملائي:

${item.converted}

ملاحظة (محاولة ${attempt}): مخرجك السابق رُفض لأن: ${reason}${attempt > 2 ? '\n(تجاهل محاولاتك السابقة تمامًا وابدأ من الصفر بهذه الجملة فقط.)' : `\nالمخرج السابق كان: ${prevOut ?? '(فارغ)'}`}
صحّح المخالفة المذكورة فقط، واحتفظ بكل الحروف والترقيم والمسافات كما هي (الفاصلة اللاتينية , والاستفهام ? يبقيان كما وردا)، وأضف الحركات وفق الدليل (كل «ال» أول كلمة = اِلْ بكسرة تحت الألف).` },
  ], 'retry');
  return { content: content.trim(), model };
}

// ---------------- main ----------------
// ---------------- Kaggle Model Proxy backend (env-gated) ----------------
// KAGGLE_PROXY_KEY مضبوط → كل نداءات LLM تذهب إلى وكيل Kaggle بدل z-ai.
// غير مضبوط → سلوك المحرك مطابق تمامًا للنسخة المعتمدة (لا فرق).
// التوافق: callLLM يقرأ completion.choices[0].message.content وcompletion.usage
// وcompletion.model — والوكيل يعيدها جميعًا (OpenAI-style).
function makeKaggleClient(model, key, url) {
  const endpoint = String(url).replace(/\/$/, '') + '/openapi/chat/completions';
  return {
    chat: {
      completions: {
        create: async (req) => {
          const body = { ...req, model };
          delete body.thinking; // الوكيل OpenAI-style — لا حقل تفكير
          if (!body.max_tokens) body.max_tokens = 8192;
          const res = await fetch(endpoint, {
            method: 'POST',
            headers: {
              Authorization: `Bearer ${key}`,
              'Content-Type': 'application/json',
            },
            body: JSON.stringify(body),
          });
          const txt = await res.text();
          if (!res.ok) {
            throw new Error(`kaggle proxy HTTP ${res.status}: ${txt.slice(0, 300)}`);
          }
          return JSON.parse(txt);
        },
      },
    },
  };
}

// ---------------- Groq backend (env-gated) ----------------
// GROQ_API_KEY مضبوط → كل نداءات LLM تذهب إلى Groq (OpenAI-compatible).
// المنطقة HK محجوبة عند Groq (403 عبر Cloudflare) لذا يُشغَّل هذا المسار من
// GitHub Actions (خوادم US). العميل يتحمل 429 فترات أطول داخليًا (حدود TPM
// في Groq تحتاج انتظارًا يتجاوز backoff المحرك) وينظف كتل <think> الدفاعية.
// temperature 0.1: مهمة نسخ حروف حرفيًا — الحتمية تقلل أخطاء الزيادة/النقصان.
function makeGroqClient(model, key) {
  // GROQ_BASE_URL: testability override (local mock server); unset → Groq.
  const base = process.env.GROQ_BASE_URL || 'https://api.groq.com';
  const endpoint = `${base.replace(/\/$/, '')}/openai/v1/chat/completions`;
  const MAX_TRIES = 8;
  return {
    chat: {
      completions: {
        create: async (req) => {
          const body = { ...req, model, temperature: 0.1 };
          delete body.thinking; // OpenAI-style endpoint — لا حقل تفكير
          // GROQ_MAX_TOKENS (default 4096): Groq on_demand TPM/context can
          // force smaller completion reserves. TPM accounting includes the
          // reserve, so batch-2 mode runs at 1024.
          if (!body.max_tokens) {
            body.max_tokens = parseInt(process.env.GROQ_MAX_TOKENS || '4096', 10);
          }
          // gpt-oss are reasoning models: at default effort the reasoning
          // field consumes the whole completion budget and content comes
          // back EMPTY ("empty completion" storm). Low effort keeps short
          // reasoning + real content within small max_tokens.
          if (/^openai\/gpt-oss/.test(model)) body.reasoning_effort = 'low';
          let lastErr;
          for (let t = 1; t <= MAX_TRIES; t++) {
            let retryAfter = 0;
            try {
              const res = await fetch(endpoint, {
                method: 'POST',
                headers: {
                  Authorization: `Bearer ${key}`,
                  'Content-Type': 'application/json',
                },
                body: JSON.stringify(body),
              });
              const ra = parseInt(res.headers.get('retry-after') || '0', 10);
              if (ra > 0) retryAfter = ra * 1000;
              const txt = await res.text();
              if (!res.ok) {
                const err = new Error(`groq HTTP ${res.status}: ${txt.slice(0, 300)}`);
                err.status = res.status;
                throw err;
              }
              const parsed = JSON.parse(txt);
              const msg = parsed.choices?.[0]?.message;
              // defensive: بعض نماذج التفكير تدمج <think> في المحتوى — انزعها
              // (parseNumbered لن يفهمها أصلًا)
              if (msg && typeof msg.content === 'string') {
                msg.content = msg.content
                  .replace(/[\s\S]*?<\/think>\s*/g, '')
                  .replace(/^<\|channel\|>final<\|message\|>\s*/g, '')
                  .trim();
              }
              return parsed;
            } catch (e) {
              lastErr = e;
              const is429 = e.status === 429 || /429|rate.?limit/i.test(String(e.message));
              if (!is429 || t === MAX_TRIES) throw e;
              const wait = retryAfter || Math.min(65000, 20000 * t);
              console.log(`    [groq-429] waiting ${Math.round(wait / 1000)}s (try ${t}/${MAX_TRIES})`);
              await new Promise((r) => setTimeout(r, wait));
            }
          }
          throw lastErr;
        },
      },
    },
  };
}

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
  console.log(`input=${input.length} done=${done.size} quarantined=${quarantined.length} todo=${todo.length} batch=${BATCH_SIZE} retryBatch=${RETRY_BATCH_SIZE} tierA=${MAPS.meta.n_entries}forms/${MAPS.meta.total_corpus_positions}pos`);
  if (todo.length === 0 && quarantined.length === 0) { console.log('nothing to do'); return; }

  // Backend selection (env-gated; unset → z-ai exactly as approved):
  //   GROQ_API_KEY > KAGGLE_PROXY_KEY > z-ai-web-dev-sdk (dynamic import —
  //   the SDK is unavailable outside this workspace, e.g. on Actions runners)
  let zai;
  let modelSeen = null;
  if (process.env.GROQ_API_KEY) {
    const gm = process.env.GROQ_MODEL || 'llama-3.3-70b-versatile';
    zai = makeGroqClient(gm, process.env.GROQ_API_KEY);
    console.log(`[groq] backend=on model=${gm}`);
  } else if (process.env.KAGGLE_PROXY_KEY) {
    zai = makeKaggleClient(
      process.env.KAGGLE_MODEL || 'google/gemini-3.1-flash-lite-preview',
      process.env.KAGGLE_PROXY_KEY,
      process.env.KAGGLE_PROXY_URL || 'https://mp-staging.kaggle.net/models',
    );
    console.log(`[kaggle] backend=on model=${process.env.KAGGLE_MODEL
      || 'google/gemini-3.1-flash-lite-preview'}`);
  } else {
    zai = await (await import('z-ai-web-dev-sdk')).default.create();
  }
  const out = fs.createWriteStream(OUT_PATH, { flags: 'a' });

  const writeRec = (rec) => {
    out.write(JSON.stringify(rec) + '\n');
    console.log(`  ${rec.id}: ${rec.status} (attempts=${rec.attempts}${Object.keys(rec.fixes).length ? `, fixes=${JSON.stringify(rec.fixes)}` : ''})`);
  };

  for (const q of quarantined) {
    writeRec({
      id: q.id, raw: q.orig_text ?? q.text,
      converted: convertPunct(q.orig_text ?? q.text),
      out: null, attempts: 0, status: 'quarantined', fixes: {}, checks: {},
      reason: 'حجر معتمد من المستخدم 2026-09-30 (§ل-5) — لا استخدام آلي حتى قرار حالة بحالة',
    });
  }

  // Parse one LLM line for one unit (punct un-swap + letter-restore +
  // validation) — identical pipeline to v5.1's per-attempt handling.
  // fallbackReason: the batch/round-level error (v5.1 passed it as the
  // rejection reason when the whole call failed).
  const parseAttempt = (item, line, fallbackReason) => {
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
    const outText = restore(unswapPunct(line));
    const v = outText ? validate(outText, item.converted)
      : { ok: false, reason: fallbackReason ?? 'تعذر التحليل ضمن الدفعة' };
    return { outText, v, punctUnswap, letterFixes };
  };

  // Finalize an OK unit (deterministic post-fixes + Tier A + rabbena) —
  // identical chain to v5.1's inline logic.
  const finalizeOk = (item, outText, punctUnswap, letterFixes) => {
    let fixes = punctUnswap ? { punct_unswap: punctUnswap } : {};
    if (letterFixes) fixes.letter_restore = letterFixes;
    // deterministic post-fixes (diacritic-only, skeleton-safe)
    const f = applyPostFixes(outText);
    if (f.fixes.al_kasra || f.fixes.taa_sukun_strip) {
      const cand = f.out;
      const v2 = validate(cand, item.converted); // safety re-check
      if (v2.ok) { outText = cand; fixes = { ...fixes, ...f.fixes }; }
    }
    // Tier A + Rabbena enforcement (guide v2.1 §ك) — AFTER al_kasra so the
    // map restores اَلْحَمْدُ if the fixer rewrites اَلْ to اِلْ. Full
    // re-validation afterwards (taa exempted on replaced tokens only); on
    // unexpected failure we keep the pre-map text (already gate-passed).
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
    return { outText, fixes };
  };

  const buildRec = (item, outText, attempts, status, fixes, reason) => ({
    id: item.id, raw: item.text, converted: item.converted,
    ...(item.letter_fix ? { orig_text: item.orig_text,
                            letter_fix: item.letter_fix } : {}),
    out: status === 'ok' ? outText : (outText ?? null),
    attempts, status, fixes,
    checks: { skeleton: status === 'ok', tanween: status === 'ok',
              taa: status === 'ok', density: status === 'ok' },
    reason: status === 'ok' ? null : reason,
  });

  // v5.2 retry queue: failed units wait here and are re-sent TOGETHER.
  // Entry: {item, prevOut, reason, attempts, punctUnswap, letterFixes}
  const retryQ = [];
  // v5.3 final-drain queue: units whose batched rounds are exhausted but whose
  // last budget round is still pending (focused individual retry).
  const finalQ = [];

  const runRetryRound = async (entries) => {
    console.log(`retry round: ${entries.length} unit(s) [${entries.map((e) => e.item.id).join(' ')}]`);
    let parsed = null, roundError = null;
    try {
      const { content, model } = await tashkeelRetryBatch(zai, entries);
      modelSeen = model ?? modelSeen;
      parsed = parseNumbered(content, entries.length);
    } catch (e) { roundError = e.message; }
    if (!parsed || !parsed.lines) {
      console.log(`  retry-batch parse failure: ${roundError ?? 'unparsed'} -> units re-queued (budget consumed)`);
    }
    for (let i = 0; i < entries.length; i++) {
      const e = entries[i];
      e.attempts++;
      let outText = null, v, punctUnswap = 0, letterFixes = 0;
      if (roundError && !parsed) {
        v = { ok: false, reason: `API error: ${roundError}` };
      } else {
        const r = parseAttempt(e.item, parsed?.lines?.[i] ?? null);
        outText = r.outText; v = r.v;
        punctUnswap = r.punctUnswap; letterFixes = r.letterFixes;
      }
      e.punctUnswap += punctUnswap; e.letterFixes += letterFixes;
      if (v.ok && outText) {
        const fin = finalizeOk(e.item, outText, e.punctUnswap, e.letterFixes);
        writeRec(buildRec(e.item, fin.outText, e.attempts, 'ok', fin.fixes, null));
      } else {
        e.prevOut = outText; // null on API/parse failure (v5.1 parity)
        e.reason = v.reason;
        if (e.attempts < MAX_RETRY_ROUNDS) {
          retryQ.push(e); // one more BATCHED round is still within budget
        } else {
          // v5.3: last budget round is reserved for the focused individual
          // drain (tashkeelSingle) — unit is NOT failed yet.
          finalQ.push(e);
        }
      }
    }
  };

  // Flush full retry-batches as failures accumulate (keeps writes flowing);
  // leftovers are drained at the end in <= RETRY_BATCH_SIZE sub-batches.
  const drainRetries = async () => {
    while (retryQ.length >= RETRY_BATCH_SIZE) {
      await runRetryRound(retryQ.splice(0, RETRY_BATCH_SIZE));
    }
  };

  // ---- Phase 1: first pass over batches (identical batch calls to v5.1) ----
  for (let b0 = 0; b0 < todo.length; b0 += BATCH_SIZE) {
    const batch = todo.slice(b0, b0 + BATCH_SIZE).map((p) => ({
      ...p, converted: convertPunct(p.text),
    }));
    console.log(`batch ${Math.floor(b0 / BATCH_SIZE) + 1}: units ${b0 + 1}-${b0 + batch.length}`);
    let parsed = null, batchError = null;
    try {
      const { content, model } = await tashkeelBatch(zai, batch);
      modelSeen = model ?? modelSeen;
      parsed = parseNumbered(content, batch.length);
    } catch (e) { batchError = e.message; }
    if (!parsed || !parsed.lines) {
      console.log(`  batch-level parse failure: ${batchError ?? 'unparsed'} -> units queued for retry round`);
    }

    for (let i = 0; i < batch.length; i++) {
      const item = batch[i];
      const r = parseAttempt(item, parsed?.lines?.[i] ?? null, batchError);
      const outText = r.outText, v = r.v;
      const punctUnswap = r.punctUnswap, letterFixes = r.letterFixes;
      const attempts = outText ? 1 : 0;
      if (v.ok && outText) {
        const fin = finalizeOk(item, outText, punctUnswap, letterFixes);
        writeRec(buildRec(item, fin.outText, attempts, 'ok', fin.fixes, null));
      } else {
        retryQ.push({ item, prevOut: outText, reason: v.reason, attempts,
                      punctUnswap, letterFixes });
      }
    }
    await drainRetries();
  }

  // ---- Phase 2a: batch drain of remaining retry entries (v5.2 semantics) ----
  while (retryQ.length > 0) {
    await runRetryRound(retryQ.splice(0, RETRY_BATCH_SIZE));
  }

  // ---- Phase 2b (v5.3): final INDIVIDUAL drain — one focused call per unit,
  // v5-style single retry. This is the unit's last budget round: ok -> ok
  // record; otherwise a final failed record is written here. ----
  for (const e of finalQ.splice(0)) {
    console.log(`final individual drain: ${e.item.id} (attempt ${e.attempts + 1})`);
    let parsed = null, roundError = null;
    try {
      const { content, model } = await tashkeelSingle(
        zai, e.item, e.prevOut, e.reason ?? 'تعذر التحليل', e.attempts);
      modelSeen = model ?? modelSeen;
      parsed = content; // single trimmed line (no numbering contract)
    } catch (err) { roundError = err.message; }
    e.attempts++;
    let outText = null, v;
    if (roundError) {
      v = { ok: false, reason: `API error: ${roundError}` };
    } else {
      const r = parseAttempt(e.item, parsed ?? null, roundError);
      outText = r.outText; v = r.v;
      e.punctUnswap += r.punctUnswap; e.letterFixes += r.letterFixes;
    }
    if (v.ok && outText) {
      const fin = finalizeOk(e.item, outText, e.punctUnswap, e.letterFixes);
      writeRec(buildRec(e.item, fin.outText, e.attempts, 'ok', fin.fixes, null));
    } else {
      let fixes = e.punctUnswap ? { punct_unswap: e.punctUnswap } : {};
      if (e.letterFixes) fixes.letter_restore = e.letterFixes;
      writeRec(buildRec(e.item, e.prevOut, e.attempts, 'failed', fixes, v.reason));
    }
  }
  out.end();

  const totalTok = TOK.usage_present
    ? (TOK.prompt_tokens + TOK.completion_tokens) : null;
  console.log(`token stats: first_calls=${TOK.first_calls} retry_calls=${TOK.retry_calls}` +
    ` rate_limited=${TOK.rate_limited}` +
    (totalTok !== null
      ? ` measured_tokens=${totalTok} (prompt=${TOK.prompt_tokens} completion=${TOK.completion_tokens})`
      : ` (backend did not expose usage — chars: system=${TOK.chars.system} user_first=${TOK.chars.user_first} out_first=${TOK.chars.out_first} user_retry=${TOK.chars.user_retry} out_retry=${TOK.chars.out_retry})`));

  const meta = {
    engine: 'ph3_engine_v53.mjs v5.3 (guide v2.1; v5.2 batched retry rounds for rounds 1-2 + v5-style focused individual retry as the last budget round — tashkeelSingle wording byte-identical to approved v5; per-unit budget unchanged (max 4 attempts); system prompt + validators byte-identical to v5.1/v5.2)',
    ...(process.env.PH3_GUIDE_FILE ? {
      prompt_mode: 'CONDENSED-EXPERIMENT (not the approved full guide v2.1 — Groq on_demand TPM caps are below the full ~10.5k-token prompt; core rule sections أ-ح kept verbatim, list appendices enforced engine-side by gates/maps as usual)',
    } : {}),
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
    retry_batch_size: RETRY_BATCH_SIZE,
    max_retry_rounds: MAX_RETRY_ROUNDS,
    token_stats: TOK,
    finished: new Date().toISOString(), punct_rule: 'U+060C->, U+061F->?',
    gates: ['skeleton+punct two-sided', 'tanween closed list 191 structures',
            'taa no-haraka (sukun stripped; Tier A spans exempt — قُوّةَ)',
            'density>=1.2/w floor5',
            'post-fixes: al_kasra + taa_sukun_strip',
            'enforcement: letter_restore (§أ-1/§و) + Tier A map + rabbena + re-validation'],
  };
  fs.writeFileSync(OUT_PATH.replace(/\.jsonl?$/, '') + '_meta.json',
    JSON.stringify(meta, null, 1));
}

// Offline self-test exports (harmless — engine still runs when invoked
// directly by the orchestrator; imports skip main()).
export const SELF = {
  SYSTEM_PROMPT, validate, validateFinal, parseNumbered, applyPostFixes,
  BATCH_SIZE, RETRY_BATCH_SIZE, MAX_RETRY_ROUNDS,
};

const isDirect = process.argv[1] && process.argv[1].endsWith('ph3_engine_v53.mjs')
  && process.argv[2] && process.argv[3]; // real invocation carries IN/OUT args
if (isDirect) {
  main().catch((e) => { console.error('FATAL:', e); process.exit(1); });
}
