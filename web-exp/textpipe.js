/**
 * textpipe.js — منفذ JavaScript لخط معالجة النص العربي (Eiqaz v1)
 * ===========================================================================
 * نقل حرفي (line-by-line) للمسار الإنتاجي الجديد بعد التوحيد:
 *   inference/lib/eqz_text.py   (التطبيع: أرقام→كلمات، ترقيم، نقحرة لاتينية)
 *   inference/lib/eqz_tokens.py (سياسة القاف v1: q/v/hamza + EGY_SOUND_MAP)
 *   inference/infer.py          (prepare_text_rich + _final_clean + الترميز)
 *   + نفس التفقيم باكوالتير من tts_arabic (phonetise_buckwalter.py)
 *
 * الصحة مضمونة باختبار تكافؤ توكن-بموكن مقابل بايثون
 * (tests/parity_textpipe.mjs و tests/browser_e2e.py) على النصوص الذهبية.
 *
 * الفرق الموثق الوحيد عن بايثون: وضع «egyptian» الكامل يطبق في بايثون
 * طبقة det_tashkeel الصارمة فوق catt — لم تُنقل للمتصفح (انظر
 * prepareTextRich) — الأوضاع manual وfusha مكافئة تمامًا.
 *
 * يحتاج قبله: <script src="models/textpipe_data.js"> (globalThis.TEXTPIPE_DATA)
 * يعرّف: globalThis.TextPipe
 */

(function () {
  'use strict';

  const D = globalThis.TEXTPIPE_DATA;
  if (!D) throw new Error('TEXTPIPE_DATA غير محمّلة — أدرج models/textpipe_data.js قبل textpipe.js');

  // ========================================================================
  // ثوابت من eqz_tokens.py (مولّدة من بايثون نفسه في textpipe_data.js)
  // ========================================================================
  const EGY_SOUND_MAP = D.EGY_SOUND_MAP;      // {'j':'v','^':'t','*':'d'}
  const TRAIN_MAX_TOKENS = D.TRAIN_MAX_TOKENS;
  const FORCED_Q = new Set(D.FORCED_Q_SKELETONS);
  const FORCED_G = new Set(D.FORCED_G_SKELETONS);
  const QAF_DIAC = new Set('auiFNK~o'.split(''));
  const QAF_PUNCT = new Set(['.', ',', '?', '!', ';', ':', '"', "'",
    '،', '؛', '…', '(', ')', '—', '-']);
  const QAF_CLITICS = ['w', 'f', 'b', 'l', 'k'];
  const MARKER_MAP = D.QAF_MARKER_MAP;
  const QAF_PROGRESSIVE_BAQR = /^[wf]?b(?:yt|y|t)?qr>/;

  const AR_LETTERS = /[\u0621-\u063A\u0641-\u064A]/;
  const TATWEEL = '\u0640';

  // ========================================================================
  // أدوات نصية (مكافئات بايثون)
  // ========================================================================
  function pySplit(text) {
    return text.trim().split(/\s+/).filter(Boolean);
  }
  function pyStripChars(s, chars) {
    let a = 0, b = s.length;
    const cs = new Set(chars.split(''));
    while (a < b && cs.has(s[a])) a++;
    while (b > a && cs.has(s[b - 1])) b--;
    return s.slice(a, b);
  }
  function divmod(n, d) { return [Math.floor(n / d), n % d]; }

  // ========================================================================
  // eqz_text.py — التطبيع الموحد
  // ========================================================================
  const TAG_BIND_RE = /([\u0621-\u063A\u0641-\u064A\u064B-\u0652])\s+(\{[^{}]{1,2}\})/g;
  const TAG_ANY_RE = /\{[^{}]{1,2}\}/g;
  const PREFIX_MARKER_RE = /\{([\u0642\u062C\u0621\u0623\u06AFqhg])\}([\u0621-\u064A][\u0621-\u064A\u064B-\u0652]*)/g;
  const PREFIX_INSERT = { '\u0642': '\u0642', '\u062C': '\u062C',
    '\u06AF': '\u062C', '\u0621': '\u0623', '\u0623': '\u0623', 'q': '\u0642',
    'g': '\u062C', 'h': '\u0623' };
  const EASTERN_DIGITS = { '٠': '0', '١': '1', '٢': '2', '٣': '3', '٤': '4',
    '٥': '5', '٦': '6', '٧': '7', '٨': '8', '٩': '9' };

  function expandPrefixMarkers(text) {
    return text.replace(PREFIX_MARKER_RE, (m, marker, word) => {
      const insert = PREFIX_INSERT[marker] !== undefined
        ? PREFIX_INSERT[marker] : marker;
      if (word.startsWith(insert)) return word + '{' + marker + '}';
      return insert + word + '{' + marker + '}';
    });
  }

  function bindMarkersToWords(text) {
    return text.replace(TAG_BIND_RE, '$1$2');
  }

  function isMarker(tag) {
    const inner = tag.slice(1, -1);
    return [...inner].some(c => 'قءأجگqhg'.includes(c)) && inner.length <= 2;
  }

  function splitMarkers(text) {
    const out = [];
    for (const w of text.split(' ')) {
      let m = null;
      for (const mm of w.matchAll(TAG_ANY_RE)) {
        if (isMarker(mm[0])) m = mm[0];
      }
      out.push([m ? w.split(m).join('') : w, m || '']);
    }
    return out;
  }

  function rejoinMarkers(wordsWithTags) {
    return wordsWithTags.map(([w, t]) => t ? w + t : w).join(' ');
  }

  // ---- الأرقام → كلمات ----------------------------------------------------
  const EGY_UNITS = { 1: 'واحد', 2: 'اتنين', 3: 'تلاتة', 4: 'أربعة',
    5: 'خمسة', 6: 'ستة', 7: 'سبعة', 8: 'تمانية', 9: 'تسعة' };
  const EGY_TEENS = { 11: 'حداشر', 12: 'اتناشر', 13: 'تلتاشر',
    14: 'اربعتاشر', 15: 'خمستاشر', 16: 'ستاشر', 17: 'سبعتاشر',
    18: 'تمنتاشر', 19: 'تسعتاشر' };
  const EGY_TENS = { 2: 'عشرين', 3: 'تلاتين', 4: 'أربعين', 5: 'خمسين',
    6: 'ستين', 7: 'سبعين', 8: 'تمانين', 9: 'تسعين' };
  const EGY_HUNDREDS = { 1: 'مية', 2: 'ميتين', 3: 'تلاتمية', 4: 'اربعمية',
    5: 'خمسمية', 6: 'ستمية', 7: 'سبعمية', 8: 'تمانمية', 9: 'تسعمية' };
  const EGY_THOUSAND = { 1: 'ألف', 2: 'ألفين' };
  const EGY_MILLION = { 1: 'مليون', 2: 'مليونين' };
  const EGY_MILLIARD = { 1: 'مليار', 2: 'مليارين' };

  const MSA_UNITS = { 1: 'واحد', 2: 'اثنين', 3: 'ثلاثة', 4: 'أربعة',
    5: 'خمسة', 6: 'ستة', 7: 'سبعة', 8: 'ثمانية', 9: 'تسعة' };
  const MSA_TEENS = { 11: 'أحد عشر', 12: 'اثنا عشر', 13: 'ثلاثة عشر',
    14: 'أربعة عشر', 15: 'خمسة عشر', 16: 'ستة عشر', 17: 'سبعة عشر',
    18: 'ثمانية عشر', 19: 'تسعة عشر' };
  const MSA_TENS = { 2: 'عشرون', 3: 'ثلاثون', 4: 'أربعون', 5: 'خمسون',
    6: 'ستون', 7: 'سبعون', 8: 'ثمانون', 9: 'تسعون' };
  const MSA_HUNDREDS = { 1: 'مائة', 2: 'مئتان', 3: 'ثلاثمائة',
    4: 'أربعمائة', 5: 'خمسمائة', 6: 'ستمائة', 7: 'سبعمائة',
    8: 'ثمانمائة', 9: 'تسعمائة' };
  const MSA_THOUSAND = { 1: 'ألف', 2: 'ألفان' };
  const MSA_MILLION = { 1: 'مليون', 2: 'مليونان' };
  const MSA_MILLIARD = { 1: 'مليار', 2: 'ملياران' };
  const MSA_PLURALS = { th3: 'آلاف', th5: 'ألف', mn3: 'ملايين', mn5: 'مليون',
    md3: 'مليارات', md5: 'مليار' };
  const EGY_PLURALS = { th3: 'آلاف', th5: 'ألف', mn3: 'ملايين', mn5: 'مليون',
    md3: 'مليارات', md5: 'مليار' };

  const DIGIT_WORDS_EGY = Object.assign({ 0: 'صفر' }, EGY_UNITS);
  const DIGIT_WORDS_MSA = Object.assign({ 0: 'صفر' }, MSA_UNITS);

  function under100(n, egy) {
    if (n === 0) return '';
    if (n < 10) return (egy ? EGY_UNITS : MSA_UNITS)[n];
    if (n >= 11 && n <= 19) return (egy ? EGY_TEENS : MSA_TEENS)[n];
    if (n === 10) return 'عشرة';
    const [tens, unit] = divmod(n, 10);
    const t = (egy ? EGY_TENS : MSA_TENS)[tens];
    const u = unit ? (egy ? EGY_UNITS : MSA_UNITS)[unit] : '';
    return u ? (u + ' و' + t) : t;
  }

  function under1000(n, egy) {
    const out = [];
    const [h, rest] = divmod(n, 100);
    if (h) out.push((egy ? EGY_HUNDREDS : MSA_HUNDREDS)[h]);
    if (rest) out.push(under100(rest, egy));
    return out.join(' و');
  }

  // [إصلاح معزول موثق — نفس إصلاح eqz_text.py بايثون 2026-10-09]:
  // كانت بايثون تستخدم مفتاحًا صفّيًا plurals[(3,)] فيسقط أي عدد ≥ مليون
  // بKeyError — الصواب البنيوي int. النقل هنا بالسلوك الصحيح، وبايثون
  // أُصلح بالتوازي (نفس الالتزام).
  function bigScale(n, one, two, pl3, pl5, egy) {
    if (n === 1) return one;
    if (n === 2) return two;
    if (n >= 3 && n <= 10) return under100(n, egy) + ' ' + pl3;
    return under1000(n, egy) + ' ' + pl5;
  }

  function numberToArabicWords(n, egy = true) {
    if (n === 0) return 'صفر';
    const thMap = egy ? EGY_THOUSAND : MSA_THOUSAND;
    const mnMap = egy ? EGY_MILLION : MSA_MILLION;
    const mdMap = egy ? EGY_MILLIARD : MSA_MILLIARD;
    const pl = egy ? EGY_PLURALS : MSA_PLURALS;
    const parts = [];
    let [milliard, rest] = divmod(n, 10 ** 9);
    let [million, rest2] = divmod(rest, 10 ** 6);
    let [thousand, rest3] = divmod(rest2, 10 ** 3);
    if (milliard) {
      parts.push(bigScale(milliard, mdMap[1], mdMap[2], pl.md3, pl.md5, egy));
    }
    if (million) {
      parts.push(bigScale(million, mnMap[1], mnMap[2], pl.mn3, pl.mn5, egy));
    }
    if (thousand) {
      if (thousand === 1) parts.push(thMap[1]);
      else if (thousand === 2) parts.push(thMap[2]);
      else if (thousand >= 3 && thousand <= 10) {
        parts.push(under100(thousand, egy) + ' ' + pl.th3);
      } else {
        parts.push(under1000(thousand, egy) + ' ' + pl.th5);
      }
    }
    if (rest3) parts.push(under1000(rest3, egy));
    return parts.join(' و');
  }

  function digitsToWords(s, egy) {
    const dw = egy ? DIGIT_WORDS_EGY : DIGIT_WORDS_MSA;
    return [...s].map(c => dw[+c]).join(' ');
  }

  // ---- سياقات الأرقام ------------------------------------------------------
  const RE_WESTERN_GRP = /\b(\d{1,3}(?:,\d{3})+)\b/g;
  const RE_PERCENT = /(\d+(?:\.\d+)?)\s*[%٪]/g;
  const RE_DECIMAL = /(?<![\d.])(\d+)\.(\d+)(?![\d.])/g;
  const RE_MULTIDOT = /\b(\d+(?:\.\d+){2,})\b/g;
  const RE_INT = /\d+/g;

  function readDecimal(intPart, fracPart, egy) {
    if (fracPart.replace(/0+$/, '') === '5') {
      const base = intPart ? under1000(+intPart, egy) : '';
      return (base + ' ونص').trim();
    }
    const intW = intPart ? under1000(+intPart, egy) : 'صفر';
    const fracW = digitsToWords(fracPart, egy);
    return intW + ' فاصل ' + fracW;
  }

  function convertNumbersIn(text, egy) {
    text = text.replace(RE_WESTERN_GRP,
      (m, g1) => g1.split(',').join(''));
    text = text.replace(RE_PERCENT, (m, num) => {
      let words;
      if (num.includes('.')) {
        const [a, b] = num.split('.');
        words = readDecimal(a, b, egy);
      } else {
        words = under1000(+num, egy) || 'صفر';
      }
      return words + ' في المية';
    });
    text = text.replace(RE_MULTIDOT, (m, g1) => {
      const parts = g1.split('.').map(g => digitsToWords(g, egy));
      return parts.join(' نقطة ');
    });
    text = text.replace(RE_DECIMAL,
      (m, a, b) => ' ' + readDecimal(a, b, egy) + ' ');
    text = text.replace(RE_INT,
      (m) => ' ' + numberToArabicWords(+m, egy) + ' ');
    return pySplit(text).join(' ');
  }

  // ---- نقحرة لاتينية -------------------------------------------------------
  const LATIN_DIGRAPHS = { sh: 'ش', ch: 'تش', th: 'ث', gh: 'غ', ph: 'ف',
    kh: 'خ', oo: 'و', ee: 'ي', ck: 'ك', qu: 'كو' };
  const LATIN_SINGLE = { a: 'ا', b: 'ب', c: 'ك', d: 'د', e: 'ي', f: 'ف',
    g: 'ج', h: 'ه', i: 'ي', j: 'ج', k: 'ك', l: 'ل', m: 'م', n: 'ن', o: 'و',
    p: 'ب', q: 'ك', r: 'ر', s: 'س', t: 'ت', u: 'و', v: 'ف', w: 'و',
    x: 'كس', y: 'ي', z: 'ز' };
  const LATIN_EXCEPTIONS = { ok: 'أوكي', okay: 'أوكي', wifi: 'واي فاي',
    'wi-fi': 'واي فاي', hello: 'ألو', hi: 'هاي', app: 'آب',
    email: 'إيميل', iphone: 'آيفون', android: 'أندرويد',
    windows: 'ويندوز', google: 'جوجل', youtube: 'يوتيوب',
    facebook: 'فيسبوك', deadline: 'ديدلاين', meeting: 'ميتينج',
    report: 'ريبورت', update: 'أبديت', download: 'دونلود',
    upload: 'أبلود', link: 'لينك', post: 'بوست', story: 'ستوري' };
  const LATIN_WORD = /[A-Za-z]+/g;
  const HAS_LATIN = /[A-Za-z]/;

  function translitWord(w) {
    const wl = w.toLowerCase();
    if (LATIN_EXCEPTIONS[wl] !== undefined) return LATIN_EXCEPTIONS[wl];
    const out = [];
    let i = 0;
    while (i < wl.length) {
      const two = wl.slice(i, i + 2);
      if (LATIN_DIGRAPHS[two] !== undefined) {
        out.push(LATIN_DIGRAPHS[two]);
        i += 2;
        continue;
      }
      out.push(LATIN_SINGLE[wl[i]] !== undefined ? LATIN_SINGLE[wl[i]] : '');
      i += 1;
    }
    return out.join('');
  }

  function transliterateLatin(text) {
    if (!HAS_LATIN.test(text)) return text;
    return text.replace(LATIN_WORD, (m) => translitWord(m));
  }

  // ---- الترقيم --------------------------------------------------------------
  const AR_COMMA = '\u060C';
  const AR_QMARK = '\u061F';
  const AR_SEMI = '\u061B';
  const RE_PUNCT_RUN = /([.,!?؛:])\1+/g;
  const RE_STANDALONE_PUNCT = /(?:^|\s)([.,!?؛:،؟]+)(?=\s|$)/g;

  function attachStandalonePunct(text) {
    const out = [];
    for (const w of text.split(' ')) {
      if (w && [...w].every(c => '.,!?؛:'.includes(c))) {
        if (out.length) out[out.length - 1] += w;
        else out.push(w);
      } else {
        out.push(w);
      }
    }
    return out.join(' ');
  }

  function normalizePunctuation(text) {
    text = text.split(AR_COMMA).join(',').split(AR_QMARK).join('?');
    text = text.replace(RE_PUNCT_RUN, '$1');
    return text;
  }

  function normalizeText(rawText, dialect = 'egy', egy = null) {
    if (egy === null) egy = dialect !== 'msa';
    let text = [...rawText].map(c => EASTERN_DIGITS[c] !== undefined
      ? EASTERN_DIGITS[c] : c).join('');
    text = expandPrefixMarkers(text);
    text = bindMarkersToWords(text);
    text = normalizePunctuation(text);
    text = convertNumbersIn(text, egy);
    text = attachStandalonePunct(text);
    text = transliterateLatin(text);
    return pySplit(text).join(' ');
  }

  // ---- الإبقاء على العربية + الترقيم المدرب --------------------------------
  const AR_AND_PUNCT = /[^\u0621-\u063A\u0641-\u064A\u064B-\u0652 .,?!]/g;

  function keepArabicAndPunct(text) {
    text = text.split(TATWEEL).join('');
    text = text.replace(AR_AND_PUNCT, ' ');
    return pySplit(text).join(' ');
  }

  const LEAD_PUNCT = '.,!?؛:"()«»' + AR_COMMA + AR_QMARK + AR_SEMI + '-…';
  const TAIL_PUNCT = LEAD_PUNCT + '%٪';

  function restorePunctuation(rawNorm, cattOut) {
    const rawWs = rawNorm.split(' ');
    const cattWs = cattOut.split(' ');
    if (rawWs.length !== cattWs.length) return cattOut;
    const out = [];
    for (let k = 0; k < rawWs.length; k++) {
      const r = rawWs[k];
      let i = 0, j = r.length;
      while (i < j && LEAD_PUNCT.includes(r[i])) i++;
      while (j > i && TAIL_PUNCT.includes(r[j - 1])) j--;
      out.push(r.slice(0, i) + cattWs[k] + r.slice(j));
    }
    return out.join(' ');
  }

  // ========================================================================
  // phonetise_buckwalter.py — التحويل والتفقيم (كما في النسخة السابقة
  // للمنفذ — مثبت تكافؤها توكن-بموكن على المسار الجديد أيضًا)
  // ========================================================================
  const AR2BW = {
    '\u0628': 'b', '\u0630': '*', '\u0637': 'T', '\u0645': 'm',
    '\u062A': 't', '\u0631': 'r', '\u0638': 'Z', '\u0646': 'n',
    '\u062B': '^', '\u0632': 'z', '\u0639': 'E', '\u0647': 'h',
    '\u062C': 'j', '\u0633': 's', '\u063A': 'g', '\u062D': 'H',
    '\u0642': 'q', '\u0641': 'f', '\u062E': 'x', '\u0635': 'S',
    '\u0634': '$', '\u062F': 'd', '\u0636': 'D', '\u0643': 'k',
    '\u0623': '>', '\u0621': "'", '\u0626': '}', '\u0624': '&',
    '\u0625': '<', '\u0622': '|', '\u0627': 'A', '\u0649': 'Y',
    '\u0629': 'p', '\u064A': 'y', '\u0644': 'l', '\u0648': 'w',
    '\u064B': 'F', '\u064C': 'N', '\u064D': 'K', '\u064E': 'a',
    '\u064F': 'u', '\u0650': 'i', '\u0651': '~', '\u0652': 'o',
  };

  function arabicToBuckwalter(word) {
    let res = '';
    for (const letter of word) {
      res += (AR2BW[letter] !== undefined ? AR2BW[letter] : letter);
    }
    return res;
  }

  const unambiguousConsonantMap = {
    'b': 'b', '*': '*', 'T': 'T', 'm': 'm', 't': 't', 'r': 'r', 'Z': 'Z',
    'n': 'n', '^': '^', 'z': 'z', 'E': 'E', 'h': 'h', 'j': 'j', 's': 's',
    'g': 'g', 'H': 'H', 'q': 'q', 'f': 'f', 'x': 'x', 'S': 'S', '$': '$',
    'd': 'd', 'D': 'D', 'k': 'k', '>': '<', "'": '<', '}': '<', '&': '<',
    '<': '<',
  };
  const ambiguousConsonantMap = { 'l': ['l', ''], 'w': 'w', 'y': 'y',
    'p': ['t', ''] };
  const maddaMap = { '|': [['<', 'aa'], ['<', 'AA']] };
  const vowelMap = {
    'A': [['aa', ''], ['AA', '']], 'Y': [['aa', ''], ['AA', '']],
    'w': [['uu0', 'uu1'], ['UU0', 'UU1']],
    'y': [['ii0', 'ii1'], ['II0', 'II1']],
    'a': ['a', 'A'],
    'u': [['u0', 'u1'], ['U0', 'U1']],
    'i': [['i0', 'i1'], ['I0', 'I1']],
  };
  const diacritics = ['o', 'a', 'u', 'i', 'F', 'N', 'K', '~'];
  const diacriticsWithoutShadda = ['o', 'a', 'u', 'i', 'F', 'N', 'K'];
  const emphatics = ['D', 'S', 'T', 'Z', 'g', 'x', 'q'];
  const forwardEmphatics = ['g', 'x'];
  const consonants = ['>', '<', '}', '&', "'", 'b', 't', '^', 'j', 'H', 'x',
    'd', '*', 'r', 'z', 's', '$', 'S', 'D', 'T', 'Z', 'E', 'g', 'f', 'q',
    'k', 'l', 'm', 'n', 'h', '|'];
  const punctuation = ['.', ',', '?', '!'];

  const fixedWords = {
    'h*A': ['h aa * aa', 'h aa * a'],
    'h*h': ['h aa * i0 h i0', 'h aa * i1 h'],
    'h*An': ['h aa * aa n i0', 'h aa * aa n'],
    "h&lA'": ['h aa < u0 l aa < i0', 'h aa < u0 l aa <'],
    '*lk': ['* aa l i0 k a', '* aa l i0 k'],
    'k*lk': ['k a * aa l i0 k a', 'k a * aa l i1 k'],
    '*lkm': '* aa l i0 k u1 m',
    '>wl}k': ['< u0 l aa < i0 k a', '< u0 l aa < i1 k'],
    'Th': 'T aa h a',
    'lkn': ['l aa k i0 nn a', 'l aa k i1 n'],
    'lknh': 'l aa k i0 nn a h u0',
    'lknhm': 'l aa k i0 nn a h u1 m',
    'lknk': ['l aa k i0 nn a k a', 'l aa k i0 nn a k i0'],
    'lknkm': 'l aa k i0 nn a k u1 m',
    'lknkmA': 'l aa k i0 nn a k u0 m aa',
    'lknnA': 'l aa k i0 nn a n aa',
    'AlrHmn': ['rr a H m aa n i0', 'rr a H m aa n'],
    'Allh': ['ll aa h i0', 'll aa h', 'll AA h u0', 'll AA h a', 'll AA h',
      'll A'],
    'h*yn': ['h aa * a y n i0', 'h aa * a y n'],
    'nt': 'n i1 t',
    'fydyw': 'v i0 d y uu1',
    'lndn': 'l A n d u1 n',
  };

  function isFixedWord(word, pronunciations) {
    let lastLetter = '';
    if (word.length > 0) lastLetter = word[word.length - 1];
    if (lastLetter === 'a') lastLetter = ['a', 'A'];
    else if (lastLetter === 'A') lastLetter = ['aa'];
    else if (lastLetter === 'u') lastLetter = ['u0'];
    else if (lastLetter === 'i') lastLetter = ['i0'];
    else if (unambiguousConsonantMap[lastLetter] !== undefined) {
      lastLetter = [unambiguousConsonantMap[lastLetter]];
    }
    const wordConsonants = word.replace(/[^h*Ahn'>wl}kmyTtfd]/g, '');
    if (wordConsonants in fixedWords) {
      const entry = fixedWords[wordConsonants];
      if (Array.isArray(entry)) {
        for (const pronunciation of entry) {
          if (lastLetter.includes(pronunciation.split(' ').pop())) {
            pronunciations.push(pronunciation.split(' '));
          }
        }
      } else {
        pronunciations.push(entry.split(' '));
      }
    }
  }

  function preprocessUtterance(utterance) {
    utterance = utterance.split('AF').join('F');
    utterance = utterance.split(TATWEEL).join('');
    utterance = utterance.split('o').join('');
    utterance = utterance.split('aA').join('A');
    utterance = utterance.split('aY').join('Y');
    utterance = utterance.split(' A').join(' ');
    utterance = utterance.split('F').join('an');
    utterance = utterance.split('N').join('un');
    utterance = utterance.split('K').join('in');
    utterance = utterance.split('|').join('>A');
    utterance = utterance.split('i~').join('~i');
    utterance = utterance.split('a~').join('~a');
    utterance = utterance.split('u~').join('~u');
    utterance = utterance.replace(/Ai/g, '<i');
    utterance = utterance.replace(/Aa/g, '>a');
    utterance = utterance.replace(/Au/g, '>u');
    utterance = utterance.replace(/^>([^auAw])/, '>a$1');
    utterance = utterance.replace(/ >([^auAw ])/g, ' >a$1');
    utterance = utterance.replace(/<([^i])/g, '<i$1');
    utterance = utterance.replace(/(\S)(\.|\?|,|!)/g, '$1 $2');
    return utterance.split(' ');
  }

  function processWord(word) {
    if (punctuation.includes(word)) return word;
    const pronunciations = [];
    isFixedWord(word, pronunciations);
    let emphaticContext = false;
    word = 'bb' + word + 'ee';
    const phones = [];
    for (let index = 2; index < word.length - 2; index++) {
      const letter = word[index];
      const letter1 = word[index + 1];
      const letter2 = word[index + 2];
      const letter_1 = word[index - 1];
      const letter_2 = word[index - 2];
      if ((consonants.includes(letter) || letter === 'w' || letter === 'y')
        && !(emphatics.includes(letter) || letter === 'r' || letter === 'l')) {
        emphaticContext = false;
      }
      if (emphatics.includes(letter)) emphaticContext = true;
      if (emphatics.includes(letter1)
        && !forwardEmphatics.includes(letter1)) {
        emphaticContext = true;
      }
      if (unambiguousConsonantMap[letter] !== undefined) {
        phones.push(unambiguousConsonantMap[letter]);
      }
      if (letter === 'l') {
        if ((!diacritics.includes(letter1)
          && vowelMap[letter1] === undefined) && letter2 === '~') {
          phones.push(ambiguousConsonantMap['l'][1]);
        } else {
          phones.push(ambiguousConsonantMap['l'][0]);
        }
      }
      if (letter === '~' && letter_1 !== 'w' && letter_1 !== 'y'
        && phones.length > 0) {
        phones[phones.length - 1] += phones[phones.length - 1];
      }
      if (letter === '|') {
        phones.push(maddaMap['|'][emphaticContext ? 1 : 0]);
      }
      if (letter === 'p') {
        if (diacritics.includes(letter1)) {
          phones.push(ambiguousConsonantMap['p'][0]);
        } else {
          phones.push(ambiguousConsonantMap['p'][1]);
        }
      }
      if (vowelMap[letter] !== undefined) {
        if (letter === 'w' || letter === 'y') {
          if (diacriticsWithoutShadda.includes(letter1) || letter1 === 'A'
            || letter1 === 'Y'
            || ((letter1 === 'w' || letter1 === 'y')
              && !(diacritics.includes(letter2) || letter2 === 'A'
                || letter2 === 'w' || letter2 === 'y'))
            || (diacriticsWithoutShadda.includes(letter_1)
              && (consonants.includes(letter1) || letter1 === 'e'))) {
            if ((letter === 'w' && letter_1 === 'u'
              && !['a', 'i', 'A', 'Y'].includes(letter1))
              || (letter === 'y' && letter_1 === 'i'
                && !['a', 'u', 'A', 'Y'].includes(letter1))) {
              phones.push(vowelMap[letter][emphaticContext ? 1 : 0][0]);
            } else {
              if (letter1 === 'A' && letter === 'w' && letter2 === 'e') {
                phones.push([ambiguousConsonantMap[letter],
                vowelMap[letter][0][0]]);
              } else {
                phones.push(ambiguousConsonantMap[letter]);
              }
            }
          } else if (letter1 === '~') {
            if (letter_1 === 'a'
              || (letter === 'w' && (letter_1 === 'i' || letter_1 === 'y'))
              || (letter === 'y' && (letter_1 === 'w' || letter_1 === 'u'))) {
              phones.push(ambiguousConsonantMap[letter],
                ambiguousConsonantMap[letter]);
            } else {
              phones.push(vowelMap[letter][0][0],
                ambiguousConsonantMap[letter]);
            }
          } else {
            if (emphaticContext) {
              if ((consonants.includes(letter_1) || letter_1 === 'u'
                || letter_1 === 'i') && letter1 === 'e') {
                phones.push([vowelMap[letter][1][0],
                vowelMap[letter][1][0].slice(1)]);
              } else {
                phones.push(vowelMap[letter][1][0]);
              }
            } else {
              if ((consonants.includes(letter_1) || letter_1 === 'u'
                || letter_1 === 'i') && letter1 === 'e') {
                phones.push([vowelMap[letter][0][0],
                vowelMap[letter][0][0].slice(1)]);
              } else {
                phones.push(vowelMap[letter][0][0]);
              }
            }
          }
        }
        if (letter === 'u' || letter === 'i') {
          if (emphaticContext) {
            if ((unambiguousConsonantMap[letter1] !== undefined
              || letter1 === 'l') && letter2 === 'e' && word.length > 7) {
              phones.push(vowelMap[letter][1][1]);
            } else {
              phones.push(vowelMap[letter][1][0]);
            }
          } else {
            if ((unambiguousConsonantMap[letter1] !== undefined
              || letter1 === 'l') && letter2 === 'e' && word.length > 7) {
              phones.push(vowelMap[letter][0][1]);
            } else {
              phones.push(vowelMap[letter][0][0]);
            }
          }
        }
        if (letter === 'a' || letter === 'A' || letter === 'Y') {
          if (letter === 'A' && (letter_1 === 'w' || letter_1 === 'k')
            && letter_2 === 'b') {
            phones.push(['a', vowelMap[letter][0][0]]);
          } else if (letter === 'A'
            && (letter_1 === 'u' || letter_1 === 'i')) {
            // لا شيء
          } else if (letter === 'A' && letter_1 === 'w' && letter1 === 'e') {
            phones.push([vowelMap[letter][0][0], vowelMap[letter][0][1]]);
          } else if ((letter === 'A' || letter === 'Y') && letter1 === 'e') {
            phones.push([vowelMap[letter][emphaticContext ? 1 : 0][0],
            vowelMap['a'][emphaticContext ? 1 : 0]]);
          } else {
            phones.push(vowelMap[letter][emphaticContext ? 1 : 0][0]);
          }
        }
      }
    }
    let possibilities = 1;
    for (const letter of phones) {
      if (Array.isArray(letter)) possibilities *= letter.length;
    }
    for (let i = 0; i < possibilities; i++) {
      pronunciations.push([]);
      let iterations = 1;
      for (const letter of phones) {
        if (Array.isArray(letter)) {
          const curIndex = Math.floor(i / iterations) % letter.length;
          if (letter[curIndex] !== '') {
            pronunciations[pronunciations.length - 1]
              .push(letter[curIndex]);
          }
          iterations *= letter.length;
        } else {
          if (letter !== '') {
            pronunciations[pronunciations.length - 1].push(letter);
          }
        }
      }
    }
    for (const pronunciation of pronunciations) {
      let prevLetter = '';
      const toDelete = [];
      for (let i = 0; i < pronunciation.length; i++) {
        const letter = pronunciation[i];
        if (['aa', 'uu0', 'ii0', 'AA', 'UU0', 'II0'].includes(letter)
          && prevLetter.toLowerCase() === letter.slice(1).toLowerCase()) {
          toDelete.push(i - 1);
          pronunciation[i] = pronunciation[i - 1][0] + pronunciation[i - 1];
        }
        if (['u0', 'i0'].includes(letter)
          && prevLetter.toLowerCase() === letter.toLowerCase()) {
          toDelete.push(i - 1);
          pronunciation[i] = pronunciation[i - 1];
        }
        if ((letter === 'y' || letter === 'w') && prevLetter === letter) {
          pronunciation[i - 1] += pronunciation[i - 1];
          toDelete.push(i);
        }
        prevLetter = letter;
      }
      for (let i = toDelete.length - 1; i >= 0; i--) {
        pronunciation.splice(toDelete[i], 1);
      }
    }
    return pronunciations[0];
  }

  function processUtterance(utterance) {
    utterance = preprocessUtterance(utterance);
    const phonemes = [];
    for (const word of utterance) {
      if (word === '-' || word === 'sil') {
        phonemes.push(['sil']);
        continue;
      }
      const phonemesWord = processWord(word);
      if (punctuation.includes(phonemesWord) && phonemes.length > 0) {
        // بايثون: phonemes[-1] += سلسلة → تمديد القائمة بأحرف السلسلة
        // (JS: مصفوفة += سلسلة تجمع نصيًا — خطأ! نمدد بالأحرف صراحة)
        for (const ch of phonemesWord) {
          phonemes[phonemes.length - 1].push(ch);
        }
      } else if (punctuation.includes(phonemesWord)) {
        // ترقيم أول كلمة: بايثون يضيف السلسلة ثم يكرر أحرفها عند الجمع
        phonemes.push([...phonemesWord]);
      } else {
        phonemes.push(phonemesWord);
      }
    }
    return phonemes.map(p => p.join(' ')).join(' + ');
  }

  // ========================================================================
  // tts_arabic/text/__init__.py — الرموز والتوكنز
  // ========================================================================
  const DOUBLING_TOKEN = '_dbl_';
  const EOS_TOKEN = '_eos_';
  const SEPARATOR_TOKEN = '_+_';
  const vowels = ['aa', 'AA', 'uu0', 'uu1', 'UU0', 'UU1', 'ii0', 'ii1',
    'II0', 'II1', 'a', 'A', 'u0', 'u1', 'U0', 'U1', 'i0', 'i1', 'I0', 'I1'];
  const vowel_map = {
    'aa': 'aa', 'AA': 'aa', 'uu0': 'uu', 'uu1': 'uu', 'UU0': 'uu',
    'UU1': 'uu', 'ii0': 'ii', 'ii1': 'ii', 'II0': 'ii', 'II1': 'ii',
    'a': 'a', 'A': 'a', 'u0': 'u', 'u1': 'u', 'U0': 'u', 'U1': 'u',
    'i0': 'i', 'i1': 'i', 'I0': 'i', 'I1': 'i',
  };
  const symbols = D.symbols;
  const phonToId = new Map(symbols.map((p, i) => [p, i]));

  function tokensToIds(phonemes) {
    return phonemes.map(p => {
      const id = phonToId.get(p);
      if (id === undefined) {
        throw new Error('رمز غير معروف: ' + JSON.stringify(p));
      }
      return id;
    });
  }

  function phonemesToTokens(phonemes) {
    const list = phonemes.split('sil').join('').split('+')
      .join('_+_').split(/\s+/).filter(Boolean);
    for (let i = 0; i < list.length; i++) {
      const phon = list[i];
      if (phon.length === 2 && !vowels.includes(phon)
        && phon[0] === phon[1]) {
        list[i] = phon[0];
        list.splice(i + 1, 0, DOUBLING_TOKEN);
        i++;
      }
      if (vowels.includes(list[i])) list[i] = vowel_map[list[i]];
    }
    list.push(SEPARATOR_TOKEN);
    list.push(EOS_TOKEN);
    return list;
  }

  // ========================================================================
  // eqz_tokens.py — سياسة القاف v1
  // ========================================================================
  function skelVariants(s) {
    const out = new Set([s]);
    if (s.startsWith('Al') && s.length > 3) out.add(s.slice(2));
    for (const c of QAF_CLITICS) {
      if (s.startsWith(c) && s.length > 2) {
        out.add(s.slice(1));
        if (s.slice(1, 3) === 'Al' && s.length > 4) out.add(s.slice(3));
      }
    }
    for (const v of [...out]) {
      if (v.endsWith('h') && v.length > 2) out.add(v.slice(0, -1) + 'p');
    }
    for (const v of [...out]) {
      if (v.endsWith('A') && v.length > 2) out.add(v.slice(0, -1));
    }
    return out;
  }

  function arSkelBuck(wordAr) {
    const w = pyStripChars(wordAr, '.,!?؟؛،:"\'()«»');
    if (!w.includes('ق')) return null;
    const b = arabicToBuckwalter(w);
    let sk = '';
    for (const c of b) if (!QAF_DIAC.has(c)) sk += c;
    if (sk.endsWith('h') && sk.length > 2) sk = sk.slice(0, -1) + 'p';
    return sk || null;
  }

  const QAF_MARKER_RE = /([\u0621-\u063A\u0641-\u064A][\u0621-\u063A\u0641-\u064A\u064B-\u0652]*)\s*\{([^{}]{1,2})\}/g;
  const QAF_MARKER_TAG_RE = /\{([^{}]{1,2})\}/g;
  const PUNCT_SPACE_RE = /\s+([،؛,.!؟:…]+)/g;

  function parseQafMarkers(text) {
    const actions = {};
    let m;
    QAF_MARKER_RE.lastIndex = 0;
    while ((m = QAF_MARKER_RE.exec(text)) !== null) {
      const word = m[1], letters = m[2];
      let act = letters.length === 1 ? MARKER_MAP[letters] : undefined;
      if (act === undefined) {
        for (const ch of letters) {
          act = MARKER_MAP[ch];
          if (act) break;
        }
      }
      if (act) {
        const skel = arSkelBuck(word);
        if (skel) {
          for (const v of skelVariants(skel)) actions[v] = act;
        }
      }
    }
    let clean = text.replace(QAF_MARKER_TAG_RE, ' ');
    clean = pySplit(clean).join(' ');
    clean = clean.replace(PUNCT_SPACE_RE, '$1');
    return [clean, actions];
  }

  function lookupAction(skel, actions) {
    if (!actions || Object.keys(actions).length === 0) return null;
    for (const v of skelVariants(skel)) {
      if (v in actions) return actions[v];
    }
    return null;
  }

  function hamzaWord(w) {
    return w.split('q~').join('<').split('q').join('<');
  }

  function fixQafV1(buck, wordActions = null) {
    if (!buck.includes('q')) return buck;
    const words = buck.split(' ');
    let hit = false;
    for (let i = 0; i < words.length; i++) {
      const w = words[i];
      if (!w.includes('q')) continue;
      let skel = '';
      for (const c of w) {
        if (!QAF_DIAC.has(c) && !QAF_PUNCT.has(c)) skel += c;
      }
      if (QAF_PROGRESSIVE_BAQR.test(skel)) {
        words[i] = hamzaWord(w);
        hit = true;
        continue;
      }
      const act = wordActions ? lookupAction(skel, wordActions) : null;
      if (act === 'q') continue;
      if (act === 'g') {
        words[i] = w.split('q').join('j');
        hit = true;
        continue;
      }
      if (act === 'h') {
        words[i] = hamzaWord(w);
        hit = true;
        continue;
      }
      const variants = skelVariants(skel);
      if ([...variants].some(v => FORCED_Q.has(v))) continue;
      if ([...variants].some(v => FORCED_G.has(v))) {
        words[i] = w.split('q').join('j');
        hit = true;
        continue;
      }
      words[i] = hamzaWord(w);
      hit = true;
    }
    return hit ? words.join(' ') : buck;
  }

  function fixQafMsa(buck, wordActions = null) {
    if (!wordActions) return buck;
    const words = buck.split(' ');
    let hit = false;
    for (let i = 0; i < words.length; i++) {
      const w = words[i];
      if (!w.includes('q')) continue;
      let skel = '';
      for (const c of w) {
        if (!QAF_DIAC.has(c) && !QAF_PUNCT.has(c)) skel += c;
      }
      const act = lookupAction(skel, wordActions);
      if (act === 'q') continue;
      if (act === 'h') {
        words[i] = w.split('q~').join('<').split('q').join('<');
        hit = true;
      } else if (act === 'g') {
        words[i] = w.split('q').join('j');
        hit = true;
      }
    }
    return hit ? words.join(' ') : buck;
  }

  // v1.0.1: توحيد كانوني لترتيب حركة/شدة (شدة أولًا)
  const DIAC_SWAP_RE = /([\u064B-\u0650])(\u0651)/g;
  function normalizeDiacriticOrder(text) {
    return text.replace(DIAC_SWAP_RE, '$2$1');
  }

  function toksMs(text) {
    const [clean, actions] = parseQafMarkers(text);
    const c2 = normalizeDiacriticOrder(clean);
    const buck = fixQafMsa(arabicToBuckwalter(c2), actions);
    return phonemesToTokens(processUtterance(buck));
  }

  function toksEgy(text) {
    const [clean, actions] = parseQafMarkers(text);
    const c2 = normalizeDiacriticOrder(clean);
    const buck = fixQafV1(arabicToBuckwalter(c2), actions);
    const toks = phonemesToTokens(processUtterance(buck));
    return toks.map(t => EGY_SOUND_MAP[t] !== undefined
      ? EGY_SOUND_MAP[t] : t);
  }

  // ========================================================================
  // infer.py — prepare_text_rich + التطبيع النهائي
  // ========================================================================
  function diacriticDensity(text) {
    const letters = text.match(new RegExp(AR_LETTERS.source, 'g')) || [];
    if (!letters.length) return [0.0, 0];
    let nD = 0;
    for (const c of text) {
      if (c >= '\u064B' && c <= '\u0652') nD++;
    }
    return [nD / letters.length, nD];
  }

  // infer.effective_diacritize_mode: كثافة < 0.30 → تشكيل تلقائي
  function effectiveDiacritizeMode(rawText, mode, dialect = 'egy') {
    if (mode === 'egyptian' || mode === 'fusha' || mode === 'manual') {
      return mode;
    }
    const [density] = diacriticDensity(
      keepArabicAndPunct(rawText));
    if (density >= 0.30) return 'manual';
    return dialect === 'msa' ? 'fusha' : 'egyptian';
  }

  const FINAL_KEEP = /[^\u0621-\u063A\u0641-\u064A\u064B-\u0652 .,?!{}]/g;
  function finalClean(text) {
    text = text.split(TATWEEL).join('');
    text = text.replace(FINAL_KEEP, ' ');
    return pySplit(text).join(' ');
  }

  /**
   * prepareTextRich — مطابق infer.prepare_text_rich للوضعين manual وfusha.
   *
   * الوضع egyptian الكامل (catt + قواعد det الصارمة) لم يُنقل للمتصفح —
   * هنا egyptian = catt + restorePunctuation (مثل fusha) — الفرق موثق:
   * قواعد det المصرية تعمل في واجهة الإنتاج بايثون فقط.
   *
   * cattVocalize: دالة (text) => Promise<string> أو null.
   */
  async function prepareTextRich(rawText, diacritizeMode = 'auto',
    dialect = 'egy', cattVocalize = null) {
    const norm = normalizeText(rawText, dialect);
    if (!AR_LETTERS.test(norm)) {
      throw new Error('النص لا يحتوي حروفًا عربية.');
    }

    const wordsTags = splitMarkers(norm);
    const tags = wordsTags.map(([, t]) => t);
    const stripped = wordsTags.map(([w]) => w).join(' ');
    const hasMarkers = tags.some(Boolean);

    const mode = effectiveDiacritizeMode(stripped, diacritizeMode, dialect);
    let voc;
    if (mode === 'egyptian' || mode === 'fusha') {
      if (typeof cattVocalize !== 'function') {
        throw new Error('التشكيل التلقائي مطلوب لكن catt غير متاح');
      }
      voc = pySplit(await cattVocalize(stripped)).join(' ');
      // det المصرية غير منقولة للمتصفح — استرجاع الترقيم فقط (كمسار fusha)
      voc = restorePunctuation(stripped, voc);
    } else {
      voc = keepArabicAndPunct(stripped);
      if (!AR_LETTERS.test(voc)) {
        throw new Error('النص لا يحتوي حروفًا عربية بعد التنظيف.');
      }
    }

    const vocWords = voc.split(' ');
    if (hasMarkers && vocWords.length === wordsTags.length) {
      voc = vocWords.map((w, i) => wordsTags[i][1] ? w + wordsTags[i][1] : w)
        .join(' ');
    }
    const text = finalClean(voc);
    return {
      text, didVocalize: mode === 'egyptian' || mode === 'fusha',
      diacritize: mode, normalized: norm,
      markers: hasMarkers,
    };
  }

  // ========================================================================
  // تقسيم النص الطويل (حدود جمل + سقف توكنز التدريب) — للبث والتوليد
  // ========================================================================
  const SENT_END = /(?<=[.!؟?…])\s+/;

  function nTokensOf(text, dialect) {
    const cleaned = keepArabicAndPunct(text);
    if (!AR_LETTERS.test(cleaned)) return 0;
    const toks = dialect === 'msa' ? toksMs(cleaned) : toksEgy(cleaned);
    return tokensToIds(toks).length;
  }

  function splitIntoChunks(rawText, dialect, maxTokens = TRAIN_MAX_TOKENS) {
    let text = bindMarkersToWords(expandPrefixMarkers(rawText));
    const segments = text.split('\n').map(s => s.trim()).filter(Boolean);
    const pieces = [];
    for (const seg of segments) {
      for (const p of seg.split(new RegExp(SENT_END.source, 'g'))) {
        const clean = p.trim()
          .replace(/^[ \t\r,.،؛;:…!؟?]+/, '')
          .replace(/[ \t\r,.،؛;:…!؟?]+$/, '');
        if (clean) pieces.push(clean);
      }
    }
    const merged = [];
    for (const p of pieces) {
      if (merged.length && nTokensOf(p, dialect) < 6) {
        merged[merged.length - 1] += ' ' + p;
      } else {
        merged.push(p);
      }
    }
    const chunks = [];
    for (const p of merged) {
      if (nTokensOf(p, dialect) <= maxTokens) {
        chunks.push(p);
        continue;
      }
      let sub = p.split(/\s*[,،؛;:]+\s*/).map(s => s.trim())
        .filter(Boolean);
      if (sub.length < 2) {
        const words = p.split(' ');
        sub = [];
        for (let i = 0; i < words.length; i += 12) {
          sub.push(words.slice(i, i + 12).join(' '));
        }
      }
      for (const s of sub) {
        if (nTokensOf(s, dialect) <= maxTokens) {
          chunks.push(s);
        } else {
          const words = s.split(' ');
          const cur = [];
          let curN = 0;
          for (const w of words) {
            const wn = nTokensOf(w, dialect);
            if (cur.length && curN + wn > maxTokens) {
              chunks.push(cur.join(' '));
              cur.length = 0;
              cur.push(w);
              curN = wn;
            } else {
              cur.push(w);
              curN += wn;
            }
          }
          if (cur.length) chunks.push(cur.join(' '));
        }
      }
    }
    const mergeCap = maxTokens >= 20 ? Math.floor(maxTokens * 0.9)
      : maxTokens;
    const merged14 = [];
    for (const c of chunks) {
      if (merged14.length
        && nTokensOf(merged14[merged14.length - 1] + ' ' + c, dialect)
        <= mergeCap) {
        merged14[merged14.length - 1] += ' ' + c;
      } else {
        merged14.push(c);
      }
    }
    return merged14.filter(c => AR_LETTERS.test(keepArabicAndPunct(c)));
  }

  // ========================================================================
  // الواجهة العامة
  // ========================================================================
  globalThis.TextPipe = {
    symbols, EGY_SOUND_MAP, TRAIN_MAX_TOKENS,
    // eqz_text
    normalizeText, numberToArabicWords, convertNumbersIn,
    transliterateLatin, keepArabicAndPunct, restorePunctuation,
    expandPrefixMarkers, bindMarkersToWords, splitMarkers,
    normalizePunctuation,
    // phonetise
    arabicToBuckwalter, processUtterance, processWord,
    phonemesToTokens, tokensToIds,
    // eqz_tokens
    parseQafMarkers, fixQafV1, fixQafMsa, arSkelBuck, skelVariants,
    normalizeDiacriticOrder, toksMs, toksEgy,
    // infer
    prepareTextRich, effectiveDiacritizeMode, diacriticDensity,
    finalClean,
    // تقسيم
    splitIntoChunks, nTokensOf,
    pySplit,
  };
})();
