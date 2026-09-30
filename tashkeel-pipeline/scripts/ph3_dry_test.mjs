// ph3_dry_test.mjs — dry unit tests for ph3_maps.mjs (NO LLM, no API).
// Covers: every Tier A entry (replacement + skeleton safety + affix
// preservation), رَبِّنَا/بِرَبِّنَا enforcement, negatives (no false matches),
// الحمد-after-al_kasra restoration, taa exemption for قُوَّةَ.
import assert from 'node:assert';
import { loadMaps, applyTierA, applyRabbena, taaCheckWithExempt, restoreLetters } from './ph3_maps.mjs';

const STRIP = /[\u064B-\u0652\u0653-\u0655\u0670]/g;
const strip = (s) => s.replace(STRIP, '');
const maps = loadMaps();
let pass = 0, fail = 0;
const t = (name, fn) => {
  try { fn(); pass++; console.log(`PASS  ${name}`); }
  catch (e) { fail++; console.log(`FAIL  ${name} -> ${e.message}`); }
};

// ---------- 1. every Tier A entry ----------
for (const e of maps.tierA) {
  t(`tierA «${e.name}» (${e.pat.length}w)`, () => {
    const words = e.name.split(' ');
    const sim = ['قبل', ...words.slice(0, -1), words[words.length - 1] + ',', 'بعد'].join(' ');
    const r = applyTierA(sim, maps);
    // skeleton preserved (punct included)
    assert.equal(strip(r.out), strip(sim), 'skeleton changed');
    // token-level expected: قبل + frozen words (last carries the comma) + بعد
    const exp = ['قبل', ...e.rep.slice(0, -1), e.rep[e.rep.length - 1] + ',', 'بعد'];
    assert.deepEqual(r.out.split(' '), exp, `got: ${r.out}`);
    assert.ok(r.forms.includes(e.name), 'form not recorded');
    assert.equal(r.tokenIdx.size, e.pat.length, 'replaced token count');
  });
}

// ---------- 2. multi-occurrence + prefix affix ----------
// (corpus-realistic: all قدر الله forms are standalone words — scan-verified;
// prefixed و/ف forms do not occur in the 21,880 units)
t('tierA multi-occurrence + paren prefix', () => {
  const sim = '(لقدر الله) أو لا قدر الله.';
  const r = applyTierA(sim, maps);
  assert.equal(strip(r.out), strip(sim));
  assert.ok(r.out.includes('(لَقَدَرَ اللهُ)'), r.out);
  assert.ok(r.out.includes('لَا قَدَرَ اللهُ.'), r.out);
  assert.equal(r.forms.length, 2, r.forms.join('|'));
});

// ---------- 2b. diacritized interior words (LLM-realistic output) ----------
t('diacritized words match by skeleton', () => {
  const sim = 'اِنْ شَاءَ اللهِ يا ربّ خير';
  const r = applyTierA(sim, maps);
  assert.equal(r.out, 'اِنْ شَاءَ اللهُ يَا رَبِّ خير', r.out);
  assert.equal(strip(r.out), strip(sim), 'skeleton changed');
});

// ---------- 3. رَبِّنَا / بِرَبِّنَا ----------
t('rabbena: ربنا يكرم', () => {
  const r = applyRabbena('ربنا يكرم', maps);
  assert.equal(r.out, 'رَبِّنَا يكرم');
});
t('rabbena: بربنا', () => {
  const r = applyRabbena('بربنا وهي', maps);
  assert.equal(r.out, 'بِرَبِّنَا وهي');
});
t('rabbena: negatives (رب/الرب/ربح) + standalone ربنا', () => {
  // وربنا does not occur in the corpus (scan: forms = ربنا 20, رب 4, بربنا 1)
  const s = 'رب الأسرة والرب ربح و ربنا';
  const r = applyRabbena(s, maps);
  assert.ok(r.out.includes('رب الأسرة'));
  assert.ok(r.out.includes('والرب'));
  assert.ok(r.out.includes('ربح'));
  assert.ok(r.out.includes('رَبِّنَا'), r.out);
  assert.equal(r.forms.length, 1, r.forms.join('|'));
});

// ---------- 4. الحمد restoration after al_kasra corruption ----------
t('الحمد لله restored from al_kasra-corrupted form', () => {
  // simulate what the engine does: al_kasra rewrites اَلْحَمْدُ -> اِلْحَمْدُ
  const corrupted = 'اِلْحَمْدُ لِلَّهِ عالسلامة';
  const r = applyTierA(corrupted, maps);
  assert.ok(r.out.startsWith('اَلْحَمْدُ لِلَّهِ'), r.out);
  assert.equal(strip(r.out), strip(corrupted), 'skeleton changed');
});

// ---------- 5. taa exemption (قُوَّةَ) ----------
t('taa exempt inside Tier A span (قُوَّةَ)', () => {
  const r = applyTierA('لا حول ولا قوة إلا بالله', maps);
  assert.ok(r.out.includes('قُوَّةَ'), r.out);
  const chk = taaCheckWithExempt(r.out, r.tokenIdx);
  assert.ok(chk.ok, JSON.stringify(chk));
});
t('taa NOT exempt outside Tier A span', () => {
  const r = applyTierA('لا حول ولا قوة إلا بالله', maps);
  const withOther = r.out + ' وحاجةَ تانية';
  const chk = taaCheckWithExempt(withOther, r.tokenIdx);
  assert.ok(!chk.ok, 'should reject ة+fatha outside spans');
});

// ---------- 6. negatives: no false matches ----------
t('negative: محمد صلاح / صلاحية / بنحلله / كما شاء', () => {
  const s = 'محمد صلاح صلاحية بنحلله كما شاء الله ينوره';
  const r = applyTierA(s, maps);
  // ينوره ≠ ينور: no 'الله ينور' match (token is ينوره)
  assert.equal(r.forms.length, 0, 'false match: ' + r.forms.join('|'));
  assert.equal(r.out, s);
});
t('negative: bare الله alone is not Tier A', () => {
  const r = applyTierA('الله يوفقك', maps);
  assert.equal(r.forms.length, 0, r.forms.join('|'));
});
t('negative: صلاة الحديد untouched by maps (letter-fix handles it)', () => {
  const s = 'جوه صلاة الحديد دي';
  const r = applyTierA(s, maps);
  assert.equal(r.out, s);
});

// ---------- 7. dual letter forms إن/ان ----------
t('dual forms: إن vs ان شاء الله', () => {
  const r = applyTierA('إن شاء الله. ان شاء الله', maps);
  assert.ok(r.out.includes('إِنْ شَاءَ اللهُ'), r.out);
  assert.ok(r.out.includes('اِنْ شَاءَ اللهُ'), r.out);
  const r2 = applyTierA('ان شاء الله', maps);
  assert.equal(r2.out, 'اِنْ شَاءَ اللهُ', r2.out);
});

// ---------- 8. rabbena + tierA composed ----------
t('composed: والله + ربنا يكرموا', () => {
  const s = 'والله ربنا يكرموا';
  const ta = applyTierA(s, maps);
  const rb = applyRabbena(ta.out, maps);
  assert.equal(rb.out, 'وَاللهِ رَبِّنَا يكرموا');
});

// ---------- 9. hamza restore (§و enforcement) ----------
t('restoreLetters: model-normalized الأكثر → raw الاكثر', () => {
  const raw = 'وهي طبعا الاكثر انتشارا';
  const out = 'وِهِي طَبْعًا الأَكْثَر اِنْتِشَارًا';
  const r = restoreLetters(out, raw);
  assert.equal(r.count, 1, `count=${r.count}`);
  assert.ok(r.out.includes('الاَكْثَر'), r.out);
  assert.ok(!r.out.includes('الأ'), r.out);
});
t('restoreLetters: الإجتماعية → الاجتماعية (keeps model diacritics)', () => {
  const raw = 'من الظواهر الاجتماعية اللي';
  const out = 'مِنْ اِلظَّوَاهِرْ اِلْإِجْتِمَاعِيَّة اِلْلِي';
  const r = restoreLetters(out, raw);
  assert.equal(r.count, 1, `count=${r.count}`);
  assert.equal(r.out, 'مِنْ اِلظَّوَاهِرْ اِلْاِجْتِمَاعِيَّة اِلْلِي', r.out);
});
t('restoreLetters: reverse direction (raw hamza, model bare)', () => {
  const raw = 'إن الواحد ما يعزلش';
  const out = 'اِنْ اَلْوَاحِدْ مَا يِعْزِلْشْ';   // model de-hamzafied إن and spelled ال wrong
  const r = restoreLetters(out, raw);
  assert.ok(r.count >= 1, `count=${r.count}`);
  assert.ok(r.out.startsWith('اِنْ') || r.out.startsWith('إِنْ'), r.out);
});
t('restoreLetters: non-alef diff -> untouched (left to retries)', () => {
  const raw = 'نمسك ورقة و الم و نسأل';
  const out = 'نَمْسِكْ وَرْقَة وَ[ ]الم وَ نَسْأَلْ';
  const r = restoreLetters(out, raw);
  assert.equal(r.count, 0);
  assert.equal(r.out, out);
});
t('restoreLetters: no diff -> no-op', () => {
  const s = 'وِاللهِ رَبِّنَا يِكْرِمْ';
  const r = restoreLetters(s, 'والله ربنا يكرم');
  assert.equal(r.count, 0);
  assert.equal(r.out, s);
});

// ---------- 9b. letter-restore families (digits + ة/ه + mixed) ----------
t('restoreLetters: digits ٢٠٢٤ → 2024', () => {
  const raw = 'صدر في ديسمبر 2024 وعنوانه';
  const out = 'صَدَرْ فِي دِيسَمْبَرْ ٢٠٢٤ وِعَنْوَانُهُ';
  const r = restoreLetters(out, raw);
  assert.equal(r.count, 4, `count=${r.count}`);
  assert.ok(r.out.includes('2024'), r.out);
  assert.ok(!r.out.includes('٢٠٢٤'), r.out);
});
t('restoreLetters: الوحدة (MSA ة) → raw الوحده (ه)', () => {
  const raw = 'وعنوانه الوحده معبر';
  const out = 'وِعَنْوَانُهُ اَلْوَحْدَة مَعْبَر';
  const r = restoreLetters(out, raw);
  assert.equal(r.count, 1, `count=${r.count}`);
  assert.ok(strip(r.out).includes('الوحده'), r.out);
  assert.ok(!strip(r.out).includes('الوحدة'), r.out);
});
t('restoreLetters: mixed families in one sentence (all restored)', () => {
  const raw = 'اساسي على تقرير في 2024 الوحده';   // raw: bare alef + ASCII digits + ه
  const out = 'أَسَاسِي عَلَى تَقْرِيرْ فِي ٢٠٢٤ الْوَحْدَة';
  const r = restoreLetters(out, raw);
  assert.ok(r.count >= 6, `count=${r.count}`);
  assert.ok(strip(r.out).includes('اساسي'), r.out);      // hamza restored
  assert.ok(r.out.includes('2024'), r.out);               // digits restored
  assert.ok(strip(r.out).includes('الوحده'), r.out);      // ه restored
});
t('restoreLetters: metathesis (length differs) -> not fixable', () => {
  const raw = 'من الظواهر الاجتماعية';
  const out = 'مِنْ اِلظَّاهِرَاتْ اِلْاجْتِمَاعِيَّةْ';   // ظاهرات = ظواهر + ت (len differs)
  const r = restoreLetters(out, raw);
  assert.equal(r.count, 0);
  assert.equal(r.out, out);
});
t('restoreLetters: digit VALUE change -> not restorable', () => {
  const raw = 'في 2024';
  const out = 'فِي ٢٠٢٥';   // ٥ = 5 ≠ 4
  const r = restoreLetters(out, raw);
  assert.equal(r.count, 0);
});

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
