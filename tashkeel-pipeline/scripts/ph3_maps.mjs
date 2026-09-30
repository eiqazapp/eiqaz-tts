// ph3_maps.mjs — deterministic Tier A + Rabbena enforcement (guide v2.1 §ك).
// Pure word-level skeleton matching; affix-preserving (punctuation/parens kept);
// letters never change (map entries are letter-faithful by construction and
// verified by scripts/ph2_build_maps.py).
// Exported: loadMaps(), applyTierA(), applyRabbena(), validateFinal()
import fs from 'node:fs';

const STRIP = /[\u064B-\u0652\u0653-\u0655\u0670]/g;
const strip = (w) => w.replace(STRIP, '');

const MAP_PATH = '/home/z/my-project/work/tier_a_map.json';

export function loadMaps(mapPath = MAP_PATH) {
  const doc = JSON.parse(fs.readFileSync(mapPath, 'utf8'));
  const mk = (e) => ({
    name: e.raw,
    pat: e.raw.split(/\s+/).map(strip),
    rep: e.out.split(/\s+/),
  });
  return {
    tierA: doc.entries.map(mk),
    rabbena: doc.rabbena.map(mk),
    meta: {
      version: doc.version,
      total_corpus_positions: doc.total_corpus_positions,
      n_entries: doc.entries.length,
    },
  };
}

// split a token into pre/core/post on the SKELETON level: diacritics are
// stripped first (they belong to the replaced core), punctuation/affixes are
// kept. This is what makes diacritized words like «اِلْحَمْدُ» match their
// skeleton «الحمد» (the naive first-letter-run approach would truncate the
// core at the first interior haraka).
function affixes(word) {
  const skel = word.replace(STRIP, '');
  const m = skel.match(/^([^\u0621-\u064A\u0640]*)([\u0621-\u064A\u0640]+)([\s\S]*)$/);
  if (!m) return null;
  return { pre: m[1], core: m[2], post: m[3] };
}

// Apply patterns word-wise. Returns { out, tokenIdx (Set of parts-indices
// whose core got replaced), forms (applied pattern names) }.
// tokenIdx indices refer to split(/(\s+)/) even positions of the RETURNED
// string (token count/boundaries are unchanged by replacement).
export function applyPatterns(outText, patterns) {
  const parts = outText.split(/(\s+)/);
  const tok = []; // {pi (parts index), text, core, pre, post}
  for (let i = 0; i < parts.length; i += 2) {
    if (!parts[i]) continue;
    const a = affixes(parts[i]);
    tok.push({ pi: i, text: parts[i], core: a ? a.core : null,
               pre: a ? a.pre : '', post: a ? a.post : '' });
  }
  const tokenIdx = new Set();
  const forms = [];
  for (const pat of patterns) {
    const k = pat.pat.length;
    for (let s = 0; s + k <= tok.length; s++) {
      let ok = true;
      for (let j = 0; j < k && ok; j++) {
        if (tok[s + j].core !== pat.pat[j]) ok = false;
      }
      if (!ok) continue;
      for (let j = 0; j < k; j++) {
        const t = tok[s + j];
        parts[t.pi] = t.pre + pat.rep[j] + t.post;
        tokenIdx.add(t.pi);
      }
      forms.push(pat.name);
      s += k - 1; // skip replaced span
    }
  }
  return { out: parts.join(''), tokenIdx, forms };
}

export function applyTierA(outText, maps) {
  return applyPatterns(outText, maps.tierA);
}

export function applyRabbena(outText, maps) {
  return applyPatterns(outText, maps.rabbena);
}

// Letter-restore (guide §أ-1/§و enforcement): the LLM tends to
// "orthographically normalize" the raw text — three observed families:
//   1. bare alef <-> hamza forms (الاكثر → الأكثر، الاجتماعية → الإجتماعية)
//   2. ASCII digits -> Arabic-Indic digits (2024 → ٢٠٢٤)
//   3. colloquial ه <-> MSA ة (الوحده → الوحدة)
// This deterministic pass copies the RAW letter back at every diff position
// whose char pair belongs to one of these families (per-position; other
// diffs are left for the targeted retry loop). All restores are 1:1 char,
// length-preserving, model's diacritics kept. Requires equal skeleton
// lengths (otherwise unaligned -> not fixable here).
const ALEF_FAMILY = new Set(['\u0627', '\u0623', '\u0625', '\u0622']); // ا أ إ آ
const AR_DIGIT = { '\u0660': '0', '\u0661': '1', '\u0662': '2', '\u0663': '3',
  '\u0664': '4', '\u0665': '5', '\u0666': '6', '\u0667': '7', '\u0668': '8',
  '\u0669': '9' };
const isMark = (ch) => {
  const c = ch.codePointAt(0);
  return (c >= 0x064B && c <= 0x0655) || c === 0x0670;
};

function restorablePair(a, b) {
  if (ALEF_FAMILY.has(a) && ALEF_FAMILY.has(b)) return true;
  if ((AR_DIGIT[a] && AR_DIGIT[a] === b) || (AR_DIGIT[b] && AR_DIGIT[b] === a))
    return true;
  if ((a === '\u0629' && b === '\u0647') || (a === '\u0647' && b === '\u0629'))
    return true; // ة <-> ه
  return false;
}

const isDigitCh = (ch) => /[0-9\u0660-\u0669]/.test(ch);
const digitVal = (ch) => (AR_DIGIT[ch] !== undefined ? AR_DIGIT[ch] : ch);
const sameValueDigitPair = (a, b) =>
  isDigitCh(a) && isDigitCh(b) && digitVal(a) === digitVal(b);

export function restoreLetters(outText, convertedRaw) {
  const aSkel = [], mapA = [];
  for (let i = 0; i < outText.length; i++) {
    if (isMark(outText[i])) continue;
    aSkel.push(outText[i]);
    mapA.push(i);
  }
  const bSkel = [];
  for (const ch of convertedRaw) if (!isMark(ch)) bSkel.push(ch);
  if (aSkel.length !== bSkel.length) return { out: outText, count: 0 };
  // digit runs: restorable only as a WHOLE number (all positions pairwise
  // same-value digits) — prevents hybrid monsters like «202٥» when the model
  // also changed one digit's value.
  const digitOk = new Array(aSkel.length).fill(false);
  let p = 0;
  while (p < aSkel.length) {
    if (isDigitCh(aSkel[p]) || isDigitCh(bSkel[p])) {
      let q = p;
      while (q < aSkel.length && (isDigitCh(aSkel[q]) || isDigitCh(bSkel[q]))) q++;
      let ok = true;
      for (let k = p; k < q; k++) {
        if (!sameValueDigitPair(aSkel[k], bSkel[k])) { ok = false; break; }
      }
      if (ok) for (let k = p; k < q; k++) digitOk[k] = true;
      p = q;
    } else p++;
  }
  const diffs = [];
  for (let k = 0; k < aSkel.length; k++) {
    if (aSkel[k] === bSkel[k]) continue;
    const a = aSkel[k], b = bSkel[k];
    if ((ALEF_FAMILY.has(a) && ALEF_FAMILY.has(b)) ||
        (a === '\u0629' && b === '\u0647') || (a === '\u0647' && b === '\u0629') ||
        digitOk[k]) {
      diffs.push(k);
    }
    // non-restorable diffs stay -> validation fails -> targeted retry
  }
  if (!diffs.length) return { out: outText, count: 0 };
  const chars = outText.split('');
  for (const k of diffs) chars[mapA[k]] = bSkel[k];
  return { out: chars.join(''), count: diffs.length };
}

// Final validation with taa-marbuta exemption for Tier A-replaced tokens
// (قُوَّةَ is the ONLY ة-with-haraka form in the frozen list — §ك).
// tokenIdx = Set of parts indices (even) replaced by the maps.
export function taaCheckWithExempt(out, exemptIdx) {
  const parts = out.split(/(\s+)/);
  for (let i = 0; i < parts.length; i += 2) {
    const w = parts[i];
    if (!w || exemptIdx.has(i)) continue;
    const m = w.match(/\u0629[\u064B-\u064E\u064F\u0650\u0651]/);
    if (m) {
      return {
        ok: false,
        reason: `التاء المربوطة تحمل حركة في «${w}» — القاعدة §ج-2: ة بلا حركة (الاستثناء الوحيد: داخل صيغ Tier A المجمدة §ك).`,
      };
    }
  }
  return { ok: true };
}
