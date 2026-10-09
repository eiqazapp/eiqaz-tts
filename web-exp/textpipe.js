/**
 * textpipe.js — منفذ JavaScript لخط معالجة النص العربي (Eiqaz/NileTTS)
 * ===========================================================================
 * نقل حرفي (line-by-line) للمسار الإنتاجي في inference/infer.py +
 * inference/webapp.py + tts_arabic/text/phonetise_buckwalter.py — بلا أي
 * تعديل منطقي. الصحة مضمونة باختبار تكافؤ توكن-بموكن مقابل بايثون
 * (web-exp/tests/parity_textpipe.mjs) على النصوص الذهبية.
 *
 * يحتاج قبله: <script src="models/textpipe_data.js"> (globalThis.TEXTPIPE_DATA)
 * يعرّف: globalThis.TextPipe
 *
 * لا يعتمد على ES modules — سكريبت كلاسيكي (توافق أوسع مع متصفحات الهواتف)
 */

(function () {
  'use strict';

  const D = globalThis.TEXTPIPE_DATA;
  if (!D) throw new Error('TEXTPIPE_DATA غير محمّلة — أدرج models/textpipe_data.js قبل textpipe.js');

  // ========================================================================
  // ثوابت من infer.py (قيم من textpipe_data.js — مولّدة من بايثون نفسه)
  // ========================================================================
  const EGY_TOKEN_MAP = D.EGY_TOKEN_MAP;              // {'j':'v','q':'<','^':'t','*':'d'}
  const TRAIN_MAX_TOKENS = D.TRAIN_MAX_TOKENS;
  const QAF_DIAC = new Set(D.QAF_DIAC.split(''));     // 'auiFNK~o'
  const QAF_CLITICS = D.QAF_CLITICS;                 // ('w','f','b','l','k')
  const SUN_LETTERS = new Set(Array.isArray(D.SUN_LETTERS)
    ? D.SUN_LETTERS : D.SUN_LETTERS.split(''));
  const MARKER_MAP = D.QAF_MARKER_MAP;

  const QAF_G_SKELETONS = new Set(D.QAF_G_SKELETONS);
  const QAF_Q_SKELETONS = new Set(D.QAF_Q_SKELETONS);
  const QAF_Q_STUDY_FORMS = D.QAF_Q_STUDY_FORMS;
  const QAF_Q_AFFIX_OK = new Set(D.QAF_Q_AFFIX_OK);
  const QAF_Q_CORPUS_FORMS = D.QAF_Q_CORPUS_FORMS;
  const QAF_Q_TRUST = new Set(D.QAF_Q_TRUST);
  const QAF_Q_TIER1 = new Set(D.QAF_Q_TIER1);
  const QAF_Q_SENTENCE_SKIP = new Set(D.QAF_Q_SENTENCE_SKIP);
  const QAF_Q_DEEP_OK = new Set(D.QAF_Q_DEEP_OK);

  // infer.py: _QAF_PROGRESSIVE_BAQR = ^[wf]?b(?:yt|y|t)?qr>
  const QAF_PROGRESSIVE_BAQR = /^[wf]?b(?:yt|y|t)?qr>/;

  const AR_LETTERS = /[\u0621-\u063A\u0641-\u064A]/;
  const TATWEEL = '\u0640';
  const SAFE_CHAR = /[^\u0621-\u063A\u0641-\u064A\u064B-\u0652 ]/g;

  // ========================================================================
  // أدوات نصية عامة (مكافئات Python)
  // ========================================================================
  function pySplit(text) {           // str.split() البايثونية
    return text.trim().split(/\s+/).filter(Boolean);
  }
  function pyStripChars(s, chars) {  // str.strip(set)
    let a = 0, b = s.length;
    const cs = new Set(chars.split(''));
    while (a < b && cs.has(s[a])) a++;
    while (b > a && cs.has(s[b - 1])) b--;
    return s.slice(a, b);
  }

  // ========================================================================
  // infer.py — التنظيف والكثافة
  // ========================================================================
  function stripTatweel(text) { return text.split(TATWEEL).join(''); }

  function keepArabicOnly(text) {
    text = stripTatweel(text);
    text = text.replace(SAFE_CHAR, ' ');
    return pySplit(text).join(' ');
  }

  function diacriticDensity(text) {
    const letters = text.match(new RegExp(AR_LETTERS.source, 'g')) || [];
    if (!letters.length) return [0.0, 0];
    let nD = 0;
    for (const c of text) {
      if (c >= '\u064B' && c <= '\u0652') nD++;
    }
    return [nD / letters.length, nD];
  }

  // ========================================================================
  // phonetise_buckwalter.py — التحويل باكوالتير + التفقيم
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
    for (const letter of word) res += (AR2BW[letter] !== undefined ? AR2BW[letter] : letter);
    return res;
  }

  const unambiguousConsonantMap = {
    'b': 'b', '*': '*', 'T': 'T', 'm': 'm', 't': 't', 'r': 'r', 'Z': 'Z',
    'n': 'n', '^': '^', 'z': 'z', 'E': 'E', 'h': 'h', 'j': 'j', 's': 's',
    'g': 'g', 'H': 'H', 'q': 'q', 'f': 'f', 'x': 'x', 'S': 'S', '$': '$',
    'd': 'd', 'D': 'D', 'k': 'k', '>': '<', "'": '<', '}': '<', '&': '<',
    '<': '<',
  };
  const ambiguousConsonantMap = { 'l': ['l', ''], 'w': 'w', 'y': 'y', 'p': ['t', ''] };
  const maddaMap = { '|': [['<', 'aa'], ['<', 'AA']] };
  const vowelMap = {
    'A': [['aa', ''], ['AA', '']], 'Y': [['aa', ''], ['AA', '']],
    'w': [['uu0', 'uu1'], ['UU0', 'UU1']],
    'y': [['ii0', 'ii1'], ['II0', 'II1']],
    'a': ['a', 'A'],
    'u': [['u0', 'u1'], ['U0', 'U1']],
    'i': [['i0', 'i1'], ['I0', 'I1']],
  };
  const nunationMap = {
    'F': [['a', 'n'], ['A', 'n']], 'N': [['u1', 'n'], ['U1', 'n']],
    'K': [['i1', 'n'], ['I1', 'n']],
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
    'Allh': ['ll aa h i0', 'll aa h', 'll AA h u0', 'll AA h a', 'll AA h', 'll A'],
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
      if (emphatics.includes(letter1) && !forwardEmphatics.includes(letter1)) {
        emphaticContext = true;
      }

      if (unambiguousConsonantMap[letter] !== undefined) {
        phones.push(unambiguousConsonantMap[letter]);
      }

      if (letter === 'l') {
        if ((!diacritics.includes(letter1) && vowelMap[letter1] === undefined)
          && letter2 === '~') {
          phones.push(ambiguousConsonantMap['l'][1]);
        } else {
          phones.push(ambiguousConsonantMap['l'][0]);
        }
      }

      if (letter === '~' && letter_1 !== 'w' && letter_1 !== 'y' && phones.length > 0) {
        phones[phones.length - 1] += phones[phones.length - 1];
      }

      if (letter === '|') {
        phones.push(maddaMap['|'][emphaticContext ? 1 : 0]);
      }

      if (letter === 'p') {
        if (diacritics.includes(letter1)) phones.push(ambiguousConsonantMap['p'][0]);
        else phones.push(ambiguousConsonantMap['p'][1]);
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
                phones.push([ambiguousConsonantMap[letter], vowelMap[letter][0][0]]);
              } else {
                phones.push(ambiguousConsonantMap[letter]);
              }
            }
          } else if (letter1 === '~') {
            if (letter_1 === 'a'
              || (letter === 'w' && (letter_1 === 'i' || letter_1 === 'y'))
              || (letter === 'y' && (letter_1 === 'w' || letter_1 === 'u'))) {
              phones.push(ambiguousConsonantMap[letter], ambiguousConsonantMap[letter]);
            } else {
              phones.push(vowelMap[letter][0][0], ambiguousConsonantMap[letter]);
            }
          } else {
            if (emphaticContext) {
              if ((consonants.includes(letter_1) || letter_1 === 'u'
                || letter_1 === 'i') && letter1 === 'e') {
                phones.push([vowelMap[letter][1][0], vowelMap[letter][1][0].slice(1)]);
              } else {
                phones.push(vowelMap[letter][1][0]);
              }
            } else {
              if ((consonants.includes(letter_1) || letter_1 === 'u'
                || letter_1 === 'i') && letter1 === 'e') {
                phones.push([vowelMap[letter][0][0], vowelMap[letter][0][0].slice(1)]);
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
          } else if (letter === 'A' && (letter_1 === 'u' || letter_1 === 'i')) {
            // لا شيء (temp = True في بايثون)
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
            pronunciations[pronunciations.length - 1].push(letter[curIndex]);
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
        phonemes[phonemes.length - 1] += phonemesWord;
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
      if (id === undefined) throw new Error(`رمز غير معروف: ${JSON.stringify(p)}`);
      return id;
    });
  }

  function phonemesToTokens(phonemes) {
    const list = phonemes.split('sil').join('').split('+').join('_+_').split(/\s+/).filter(Boolean);
    for (let i = 0; i < list.length; i++) {
      const phon = list[i];
      if (phon.length === 2 && !vowels.includes(phon) && phon[0] === phon[1]) {
        list[i] = phon[0];
        list.splice(i + 1, 0, DOUBLING_TOKEN);
        i++;  // يحاكي سلوك enumerate مع الإدراج — التوكن المدرج بلا فحص
      }
      if (vowels.includes(list[i])) list[i] = vowel_map[list[i]];
    }
    list.push(SEPARATOR_TOKEN);
    list.push(EOS_TOKEN);
    return list;
  }

  // ========================================================================
  // infer.py — طبقة القاف الكاملة
  // ========================================================================
  function isArDiac(ch) { return ch >= '\u064B' && ch <= '\u0652'; }

  function arSkel(wordAr) {
    let w = pyStripChars(wordAr, '.,!?؟؛،:"\'()«»');
    if (!w.includes('ق')) return null;
    const b = arabicToBuckwalter(w);
    let sk = '';
    for (const c of b) if (!QAF_DIAC.has(c)) sk += c;
    if (sk.endsWith('h') && sk.length > 2) sk = sk.slice(0, -1) + 'p';
    return sk || null;
  }

  function qafEnvIsDeep(wordAr) {
    const i = wordAr.indexOf('ق');
    if (i < 0 || i + 1 >= wordAr.length) return false;
    return ['\u0650', '\u064F', '\u064D', '\u064C'].includes(wordAr[i + 1]);
  }

  function forceHamzaWord(wordAr) {
    const i = wordAr.indexOf('ق');
    if (i < 0 || i + 1 >= wordAr.length) return wordAr;
    const nxt = wordAr[i + 1];
    if (['\u0650', '\u064F', '\u064D', '\u064C'].includes(nxt)) {
      return wordAr.slice(0, i + 1) + '\u064E' + wordAr.slice(i + 2);
    }
    return wordAr;
  }

  function splitArPrefix(wordAr) {
    const n = wordAr.length;
    let i = 0;
    while (i < n && isArDiac(wordAr[i])) i++;
    if (i >= n) return ['', wordAr];

    function skipDiac(j) {
      while (j < n && isArDiac(wordAr[j])) j++;
      return j;
    }

    const first = wordAr[i];
    if (first === 'ا') {
      const j = skipDiac(i + 1);
      const k = (j < n && wordAr[j] === 'ل') ? skipDiac(j + 1) : j;
      if (j < n && wordAr[j] === 'ل' && n - k >= 2) {
        return [wordAr.slice(0, k), wordAr.slice(k)];
      }
      return ['', wordAr];
    }
    if ('وفبلك'.includes(first)) {
      const j = skipDiac(i + 1);
      if (j >= n) return ['', wordAr];
      const second = wordAr[j];
      if (second === 'ا') {
        const k = skipDiac(j + 1);
        const m = (k < n && wordAr[k] === 'ل') ? skipDiac(k + 1) : k;
        if (k < n && wordAr[k] === 'ل' && n - m >= 2) {
          return [wordAr.slice(0, m), wordAr.slice(m)];
        }
        return [wordAr.slice(0, j), wordAr.slice(j)];
      }
      if (second === 'ل' && first === 'ل') {
        const k = skipDiac(j + 1);
        if (n - k >= 2) return [wordAr.slice(0, k), wordAr.slice(k)];
        return ['', wordAr];
      }
      if (n - j >= 3) return [wordAr.slice(0, j), wordAr.slice(j)];
    }
    return ['', wordAr];
  }

  function prefixCandidates(wordAr, maxStrip = 3) {
    const out = [['', wordAr]];
    let cur = wordAr, acc = '';
    for (let k = 0; k < maxStrip; k++) {
      const [p, rest] = splitArPrefix(cur);
      if (!p || rest === cur) break;
      acc += p;
      cur = rest;
      out.push([acc, cur]);
    }
    return out;
  }

  function prefixIsDefinite(prefix) {
    return prefix.includes('ال') || prefix.includes('لل');
  }

  function definitizeForm(form) {
    let f = form.replace(/[\u064B-\u064D]+$/, '');
    if (/ا[\u064B]$/.test(form)) f = f.slice(0, -1);
    f = f.replace(/[\u064E\u064F\u0650]$/, '');
    return f;
  }

  function attachDefinite(prefix, form) {
    let lead = '', al = 'ال';
    if (prefix === 'لل') { lead = 'ل'; al = 'ل'; }
    else if (prefix.length >= 2 && prefix[prefix.length - 2] === 'ا'
      && prefix[prefix.length - 1] === 'ل') {
      lead = prefix.slice(0, -2);
    } else if (['و', 'ف', 'ب', 'ل', 'ك'].includes(prefix)) {
      return prefix + form;
    }
    const base = definitizeForm(form);
    if (base.startsWith('ا')) return lead + 'الِ' + base;
    const first = base[0];
    if (SUN_LETTERS.has(first)) {
      return lead + al + first + '\u0651' + base.slice(1);
    }
    return lead + al + '\u0652' + base;
  }

  const MARKER_RE = /([\u0621-\u063A\u0641-\u064A][\u0621-\u063A\u0641-\u064A\u064B-\u0652]*)\s*\{([^{}]{1,2})\}/g;
  const MARKER_TAG_RE = /\{([^{}]{1,2})\}/g;
  const PUNCT_SPACE_RE = /\s+([،؛,.!؟:…]+)/g;

  function parseQafMarkers(text) {
    const actions = {};
    let m;
    MARKER_RE.lastIndex = 0;
    while ((m = MARKER_RE.exec(text)) !== null) {
      const word = m[1], letters = m[2];
      let act = letters.length === 1 ? MARKER_MAP[letters] : undefined;
      if (act === undefined) {
        for (const ch of letters) {
          act = MARKER_MAP[ch];
          if (act) break;
        }
      }
      if (act) {
        const skel = arSkel(word);
        if (skel) {
          for (const v of qafSkelVariants(skel)) actions[v] = act;
        }
      }
    }
    let clean = text.replace(MARKER_TAG_RE, ' ');
    clean = pySplit(clean).join(' ');
    clean = clean.replace(PUNCT_SPACE_RE, '$1');
    return [clean, actions];
  }

  function lookupQafAction(skel, actions) {
    if (!actions || Object.keys(actions).length === 0) return null;
    for (const v of qafSkelVariants(skel)) {
      if (v in actions) return actions[v];
    }
    return null;
  }

  function qafFormOf(skel) {
    if (QAF_Q_STUDY_FORMS[skel] !== undefined) return QAF_Q_STUDY_FORMS[skel];
    return QAF_Q_CORPUS_FORMS[skel] !== undefined ? QAF_Q_CORPUS_FORMS[skel] : null;
  }

  function plantStudyWord(wordAr, qafMode, action, standalone) {
    let best = null;
    for (const [prefix, base] of prefixCandidates(wordAr)) {
      const skel = arSkel(base);
      if (skel && qafFormOf(skel) !== null) {
        best = [prefix, base, skel];
        break;
      }
    }

    if (action === 'q') {
      const wSkel = arSkel(wordAr);
      if (wSkel && [...qafSkelVariants(wSkel)].some(v => QAF_Q_TRUST.has(v))) {
        return [null, wSkel, true];
      }
      if (wSkel && QAF_Q_CORPUS_FORMS[wSkel] !== undefined) {
        for (const [_pfx, _base] of prefixCandidates(wordAr)) {
          const bsk = arSkel(_base);
          if (bsk && QAF_Q_DEEP_OK.has(bsk)) {
            const form = QAF_Q_CORPUS_FORMS[wSkel];
            return [form, arSkel(form) || wSkel, true];
          }
        }
      }
      if (best !== null) {
        const [prefix, _base, skel] = best;
        if (QAF_Q_DEEP_OK.has(skel) && !prefixIsDefinite(prefix)) {
          const form = qafFormOf(skel);
          const planted = prefix + form;
          const plantedSkel = arSkel(planted) || skel;
          return [planted, plantedSkel, true];
        }
      }
      return [null, null, false];
    }

    if (best === null) return [null, null, false];
    const [prefix, _base, skel] = best;
    const envDeep = qafEnvIsDeep(wordAr);
    if (envDeep) {
      return [null, skel, true];
    } else if (qafMode !== 'auto' && qafMode !== 'qaf') {
      return [null, skel, false];
    } else if (!(qafMode === 'auto' ? QAF_Q_TIER1 : QAF_Q_DEEP_OK).has(skel)) {
      return [null, skel, false];
    } else if (qafMode === 'auto'
      && [...qafSkelVariants(skel)].some(v => QAF_G_SKELETONS.has(v))) {
      return [null, skel, false];
    } else if (!standalone && QAF_Q_SENTENCE_SKIP.has(skel)) {
      return [null, skel, false];
    }
    const form = qafFormOf(skel);
    let planted;
    if (prefixIsDefinite(prefix)) planted = attachDefinite(prefix, form);
    else planted = prefix + form;
    const plantedSkel = arSkel(planted) || skel;
    return [planted, plantedSkel, true];
  }

  function applyQafTextLayer(text, qafMode, dialect, actions, standalone) {
    if (dialect !== 'egy' || !text.includes('ق')) {
      return [text, [], new Set(), []];
    }
    const out = [], planted = [], native = new Set(), unDeep = [];
    for (const w of text.split(' ')) {
      if (!w.includes('ق')) { out.push(w); continue; }
      const skelFull = arSkel(w);
      const act = skelFull ? lookupQafAction(skelFull, actions) : null;
      if (act === 'h') {
        let w2 = forceHamzaWord(w);
        if (w2 !== w) { unDeep.push({ was: w, now: w2 }); w2 = w; }
        out.push(w2);
        continue;
      }
      const [plantedW, skel, envDeep] = plantStudyWord(w, qafMode, act, standalone);
      if (plantedW !== null && plantedW !== w) {
        planted.push({ was: w, now: plantedW, skel: skel });
        out.push(plantedW);
        if (skel) native.add(skel);
      } else {
        out.push(w);
        if (envDeep && skel && !(
          act === null && qafMode === 'qaf'
          && !QAF_Q_DEEP_OK.has(skel)
          && !QAF_Q_TRUST.has(skel)
          && !(skel in QAF_Q_CORPUS_FORMS))) {
          native.add(skel);
        }
      }
    }
    return [out.join(' '), planted, native, unDeep];
  }

  function qafSkelVariants(s) {
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

  function fixQaf(buck, mode = 'auto', dialect = 'egy', wordActions = null,
    nativeQSkel = null) {
    if (!buck.includes('q')) return buck;
    if (dialect === 'msa') {
      if (wordActions && Object.keys(wordActions).length > 0) {
        const words = buck.split(' ');
        let hit = false;
        for (let i = 0; i < words.length; i++) {
          const w = words[i];
          if (!w.includes('q')) continue;
          let skel = '';
          for (const c of w) if (!QAF_DIAC.has(c)) skel += c;
          const act = lookupQafAction(skel, wordActions);
          if (act === 'g') { words[i] = w.split('q').join('j'); hit = true; }
          else if (act === 'h') { words[i] = w.split('q').join('<'); hit = true; }
          else if (act === 'q') { words[i] = w.split('q').join('<'); hit = true; }
        }
        if (hit) buck = words.join(' ');
      }
      if (mode === 'hamza') return buck.split('q').join('<');
      if (mode === 'g') return buck.split('q').join('j');
      return buck.split('q').join('<');
    }
    const words = buck.split(' ');
    let hit = false;

    for (let i = 0; i < words.length; i++) {
      const w = words[i];
      if (!w.includes('q')) continue;
      let skel = '';
      for (const c of w) if (!QAF_DIAC.has(c)) skel += c;
      if (QAF_PROGRESSIVE_BAQR.test(skel)) continue;
      const variants = qafSkelVariants(skel);
      const act = wordActions ? lookupQafAction(skel, wordActions) : null;
      if (act === 'g') {
        words[i] = w.split('q').join('j');
        hit = true;
        continue;
      }
      if (act === 'h') continue;
      if (act === 'q') {
        if (nativeQSkel && nativeQSkel.has(skel)) continue;
        words[i] = w.split('q').join('j');
        hit = true;
        continue;
      }
      if (mode === 'g') {
        words[i] = w.split('q').join('j');
        hit = true;
        continue;
      }
      if (mode === 'hamza') continue;
      if (mode === 'qaf'
        && [...variants].some(v => QAF_Q_SKELETONS.has(v))) {
        if (nativeQSkel && nativeQSkel.has(skel)) continue;
        words[i] = w.split('q').join('j');
        hit = true;
        continue;
      }
      if ([...variants].some(v => QAF_G_SKELETONS.has(v))) {
        words[i] = w.split('q').join('j');
        hit = true;
      }
    }
    return hit ? words.join(' ') : buck;
  }

  // ========================================================================
  // المُشكِّل (توكنات) — مسارات infer.py
  // ========================================================================
  function toksMs(text) {
    return phonemesToTokens(processUtterance(arabicToBuckwalter(text)));
  }

  function toksEgy(text, qafMode = 'auto', wordActions = null, nativeQSkel = null) {
    const buck = fixQaf(arabicToBuckwalter(text), qafMode, 'egy',
      wordActions, nativeQSkel);
    const toks = phonemesToTokens(processUtterance(buck));
    return toks.map(t => EGY_TOKEN_MAP[t] !== undefined ? EGY_TOKEN_MAP[t] : t);
  }

  function msaSynthesisTokens(text, qafMode = 'auto', wordActions = null) {
    const buck = fixQaf(arabicToBuckwalter(text), qafMode, 'msa', wordActions);
    return phonemesToTokens(processUtterance(buck));
  }

  // ========================================================================
  // prepare_text_rich (infer.py) — catt يُمرَّر كدالة خارجية
  // ========================================================================
  function prepareTextRich(rawText, vocalizeMode, dialect, qafMode,
    cattVocalize /* (text) => vocalizedText | null */) {
    let text = pySplit(rawText).join(' ');
    const [cleanText, markerActions] = parseQafMarkers(text);
    text = cleanText;

    if (!AR_LETTERS.test(keepArabicOnly(text))) {
      throw new Error('النص لا يحتوي حروفًا عربية.');
    }

    const [density] = diacriticDensity(keepArabicOnly(text));
    const doVocalize = (vocalizeMode === 'always'
      || (vocalizeMode === 'auto' && density < 0.30));

    let voc;
    if (doVocalize) {
      if (typeof cattVocalize !== 'function') {
        throw new Error('التشكيل التلقائي مطلوب لكن catt غير متاح');
      }
      voc = pySplit(cattVocalize(text)).join(' ');
    } else {
      voc = keepArabicOnly(text);
      if (!AR_LETTERS.test(voc)) {
        throw new Error('النص لا يحتوي حروفًا عربية بعد التنظيف.');
      }
    }

    const standalone = pySplit(voc).filter(Boolean).length <= 2;
    const [newText, planted, native, unDeep] = applyQafTextLayer(
      voc, qafMode, dialect, markerActions, standalone);
    return {
      text: newText, didVocalize: doVocalize, qafPlanted: planted,
      qafNative: [...native], qafActions: markerActions,
      qafUnDeep: unDeep,
    };
  }

  // ========================================================================
  // webapp.py — التقسيم التلقائي للنصوص الطويلة
  // ========================================================================
  const SENT_END = /(?<=[.!؟?…])\s+/;
  const SOFT_SPLIT = /\s*[,،؛;:]+\s*/;
  const TAG_BIND_RE = /([\u0621-\u063A\u0641-\u064A\u064B-\u0652])\s+(\{[^{}]{1,2}\})/g;

  function bindMarkersToWords(rawText) {
    return rawText.replace(TAG_BIND_RE, '$1$2');
  }

  function nTokensOf(text, dialect) {
    const cleaned = keepArabicOnly(text);
    if (!AR_LETTERS.test(cleaned)) return 0;
    const toks = dialect === 'msa'
      ? msaSynthesisTokens(cleaned, 'auto')
      : toksEgy(cleaned, 'auto');
    return tokensToIds(toks).length;
  }

  function splitIntoChunks(rawText, dialect, maxTokens = TRAIN_MAX_TOKENS) {
    rawText = bindMarkersToWords(rawText);
    const segments = pySplit(rawText.split('\n').join('\n'))
      .length ? rawText.split('\n').map(s => s.trim()).filter(Boolean) : [];
    const pieces = [];
    for (const seg of segments) {
      for (const p of seg.split(new RegExp(SENT_END.source, 'g'))) {
        const clean = p.trim().replace(/^[ \t\r,.،؛;:…!؟?]+|[ \t\r,.،؛;:…!؟?]+$/g, '');
        if (clean) pieces.push(clean);
      }
    }

    const merged = [];
    for (const p of pieces) {
      if (merged.length && nTokensOf(p, dialect) < 6) {
        merged[merged.length - 1] = merged[merged.length - 1] + ' ' + p;
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
      let sub = p.split(new RegExp(SOFT_SPLIT.source, 'g')).map(s => s.trim()).filter(Boolean);
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

    const mergeCap = maxTokens >= 20 ? Math.floor(maxTokens * 0.9) : maxTokens;
    const merged14 = [];
    for (const c of chunks) {
      if (merged14.length
        && nTokensOf(merged14[merged14.length - 1] + ' ' + c, dialect) <= mergeCap) {
        merged14[merged14.length - 1] = merged14[merged14.length - 1] + ' ' + c;
      } else {
        merged14.push(c);
      }
    }
    return merged14.filter(c => AR_LETTERS.test(keepArabicOnly(c)));
  }

  function effectiveVocalizeMode(rawText, mode) {
    if (mode === 'always' || mode === 'never') return mode;
    const [density] = diacriticDensity(keepArabicOnly(rawText));
    return density < 0.30 ? 'always' : 'never';
  }

  // ========================================================================
  // الواجهة العامة
  // ========================================================================
  globalThis.TextPipe = {
    // بيانات
    symbols, EGY_TOKEN_MAP, TRAIN_MAX_TOKENS,
    // تنظيف
    keepArabicOnly, stripTatweel, diacriticDensity, pySplit,
    // باكوالتير/التفقيم
    arabicToBuckwalter, processUtterance, processWord,
    phonemesToTokens, tokensToIds,
    // القاف
    parseQafMarkers, fixQaf, arSkel, qafSkelVariants,
    applyQafTextLayer, plantStudyWord,
    // مسارات كاملة
    toksMs, toksEgy, msaSynthesisTokens, prepareTextRich,
    // تقسيم
    bindMarkersToWords, splitIntoChunks, nTokensOf,
    effectiveVocalizeMode,
  };
})();
