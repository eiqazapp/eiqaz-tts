/**
 * catt.js — منفذ JavaScript للمُشكِّل catt_eo (تشكيل تلقائي داخل المتصفح)
 * ===========================================================================
 * نقل حرفي لـ tts_arabic/vocalizer/models/catt/:
 *   - network.py::CATTModel.predict (دون جزء onnxruntime — يُمرَّر جلسة)
 *   - tashkeel_tokenizer_mod.py::TashkeelTokenizer
 *   - bw2ar.py (التحويل باكوالتير↔عربي)
 *   - utils.py (strip_tashkeel / strip_tatweel / remove_non_arabic)
 *
 * الصحة مضمونة باختبار تكافؤ (expected_catt.json مولَّد من بايثون)
 * داخل اختبار المتصفح الحقيقي (tests/browser_e2e.py).
 *
 * يعرّف: globalThis.CattTashkeel
 */
(function () {
  'use strict';

  // ========================================================================
  // utils.py
  // ========================================================================
  const FATHATAN = '\u064b', DAMMATAN = '\u064c', KASRATAN = '\u064d';
  const FATHA = '\u064e', DAMMA = '\u064f', KASRA = '\u0650';
  const SHADDA = '\u0651', SUKUN = '\u0652', TATWEEL = '\u0640';
  const HARAKAT_CLASS = '[' + [FATHATAN, DAMMATAN, KASRATAN, FATHA, DAMMA,
    KASRA, SUKUN, SHADDA].join('') + ']';

  function stripTashkeel(text) {
    let t = text.replace(new RegExp(HARAKAT_CLASS, 'g'), '');
    t = t.split('\u064E').join('');
    t = t.split('\u0671').join('');
    return t;
  }
  function stripTatweel(text) {
    return text.split(TATWEEL).join('');
  }
  function removeNonArabic(text) {
    text = stripTashkeel(text);
    text = stripTatweel(text);
    return text.replace(/[^\u0621-\u063A\u0641-\u064A ]/g, ' ')
      .trim().split(/\s+/).filter(Boolean).join(' ');
  }

  // ========================================================================
  // bw2ar.py
  // ========================================================================
  const buck2uni = {
    "'": '\u0621', '|': '\u0622', '>': '\u0623', '&': '\u0624',
    '<': '\u0625', '}': '\u0626', 'A': '\u0627', 'b': '\u0628',
    'p': '\u0629', 't': '\u062A', 'v': '\u062B', 'j': '\u062C',
    'H': '\u062D', 'x': '\u062E', 'd': '\u062F', '*': '\u0630',
    'r': '\u0631', 'z': '\u0632', 's': '\u0633', '$': '\u0634',
    'S': '\u0635', 'D': '\u0636', 'T': '\u0637', 'Z': '\u0638',
    'E': '\u0639', 'g': '\u063A', '_': '\u0640', 'f': '\u0641',
    'q': '\u0642', 'k': '\u0643', 'l': '\u0644', 'm': '\u0645',
    'n': '\u0646', 'h': '\u0647', 'w': '\u0648', 'Y': '\u0649',
    'y': '\u064A', 'F': '\u064B', 'N': '\u064C', 'K': '\u064D',
    'a': '\u064E', 'u': '\u064F', 'i': '\u0650', '~': '\u0651',
    'o': '\u0652', '`': '\u0670', '{': '\u0671',
  };
  const uni2buck = {};
  for (const [k, v] of Object.entries(buck2uni)) uni2buck[v] = k;
  uni2buck['\ufefb'] = 'lA';
  uni2buck['\ufef7'] = 'l>';
  uni2buck['\ufef5'] = 'l|';
  uni2buck['\ufef9'] = 'l<';

  function transliterateWord(inputWord, direction) {
    let out = '';
    for (const char of inputWord) {
      if (direction === 'bw2ar') out += (buck2uni[char] !== undefined ? buck2uni[char] : char);
      else out += (uni2buck[char] !== undefined ? uni2buck[char] : char);
    }
    return out;
  }
  function transliterateText(inputText, direction) {
    let out = '';
    for (const w of inputText.split(' ')) out += transliterateWord(w, direction) + ' ';
    return out.slice(0, -1);
  }

  // ========================================================================
  // TashkeelTokenizer
  // ========================================================================
  class TashkeelTokenizer {
    constructor() {
      this.letters = [' ', '$', '&', "'", '*', '<', '>', 'A', 'D', 'E', 'H',
        'S', 'T', 'Y', 'Z', 'b', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n',
        'p', 'q', 'r', 's', 't', 'v', 'w', 'x', 'y', 'z', '|', '}'];
      this.letters = ['<PAD>', '<BOS>', '<EOS>'].concat(this.letters, ['<MASK>']);

      this.noTashkeelTag = '<NT>';
      this.tashkeelList = ['<NT>', '<SD>', '<SDD>', '<SF>', '<SFF>', '<SK>',
        '<SKK>', 'F', 'K', 'N', 'a', 'i', 'o', 'u', '~'];
      this.tashkeelList = ['<PAD>', '<BOS>', '<EOS>'].concat(this.tashkeelList);

      this.tashkeelMap = {};
      this.tashkeelList.forEach((c, i) => { this.tashkeelMap[c] = i; });
      this.lettersMap = {};
      this.letters.forEach((c, i) => { this.lettersMap[c] = i; });

      this.inverseTags = {
        '~a': '<SF>', '~u': '<SD>', '~i': '<SK>', '~F': '<SFF>',
        '~N': '<SDD>', '~K': '<SKK>',
      };
      this.tags = {};
      for (const [k, v] of Object.entries(this.inverseTags)) this.tags[v] = k;
      this.shaddahLast = ['a~', 'u~', 'i~', 'F~', 'N~', 'K~'];
      this.shaddahFirst = ['~a', '~u', '~i', '~F', '~N', '~K'];
      this.tahkeelChars = ['F', 'N', 'K', 'a', 'u', 'i', '~', 'o'];
    }

    cleanText(text) {
      text = text.split(TATWEEL).join('');
      text = text.split('ٱ').join('ا');
      text = text.replace(
        /[^\u0621-\u063A\u0640-\u0652\u0670\u0671\ufefb\ufef7\ufef5\ufef9 ]/g,
        ' ');
      return text.trim().split(/\s+/).filter(Boolean).join(' ');
    }

    unifyShaddahPosition(textWithTashkeel) {
      for (let i = 0; i < this.shaddahFirst.length; i++) {
        textWithTashkeel = textWithTashkeel.split(this.shaddahLast[i])
          .join(this.shaddahFirst[i]);
      }
      return textWithTashkeel;
    }

    splitTashkeelFromText(textWithTashkeel) {
      textWithTashkeel = this.cleanText(textWithTashkeel);
      textWithTashkeel = transliterateText(textWithTashkeel, 'ar2bw');
      textWithTashkeel = textWithTashkeel.split('`').join('');
      textWithTashkeel = this.unifyShaddahPosition(textWithTashkeel);

      for (const ch of this.tahkeelChars) {
        textWithTashkeel = textWithTashkeel.split(ch + ch).join(ch);
      }

      const pairs = [];
      for (let i = 0; i < textWithTashkeel.length; i++) {
        const cur = textWithTashkeel[i];
        const nxt = textWithTashkeel[i + 1];
        if (i < textWithTashkeel.length - 1
          && this.tashkeelMap[cur] === undefined
          && this.tashkeelMap[nxt] !== undefined) {
          if (nxt === '~') {
            const nn = textWithTashkeel[i + 2];
            if (i + 2 < textWithTashkeel.length
              && ('~' + nn) in this.inverseTags) {
              pairs.push([cur, this.inverseTags['~' + nn]]);
            } else {
              pairs.push([cur, '~']);
            }
          } else {
            pairs.push([cur, nxt]);
          }
        } else if (this.tashkeelMap[cur] === undefined) {
          pairs.push([cur, this.noTashkeelTag]);
        }
      }
      return [['<BOS>', '<BOS>']].concat(pairs, [['<EOS>', '<EOS>']]);
    }

    combineTashkeelWithText(pairs) {
      const combined = [];
      for (const [letter, tashkeel] of pairs) {
        combined.push(letter);
        if (this.tags[tashkeel] !== undefined) {
          combined.push(this.tags[tashkeel]);
        } else if (tashkeel !== this.noTashkeelTag) {
          combined.push(tashkeel);
        }
      }
      return combined.join('');
    }

    encode(textWithTashkeel) {
      const pairs = this.splitTashkeelFromText(textWithTashkeel);
      const inputIds = pairs.map(p => {
        const id = this.lettersMap[p[0]];
        if (id === undefined) throw new Error(`حرف غير معروف للمشكل: ${p[0]}`);
        return id;
      });
      const targetIds = pairs.map(p => this.tashkeelMap[p[1]]);
      return [inputIds, targetIds];
    }

    filterTashkeel(tashkeel) {
      return tashkeel.map((t, i) => {
        if (i !== 0 && t === '<BOS>') return this.noTashkeelTag;
        if (i !== tashkeel.length - 1 && t === '<EOS>') {
          return this.noTashkeelTag;
        }
        return t;
      });
    }

    decode(inputIds, targetIds) {
      const arTexts = [];
      for (let j = 0; j < inputIds.length; j++) {
        let letters = inputIds[j].map(i => this.letters[i]);
        let tashkeel = targetIds[j].map(i => this.tashkeelList[i]);
        letters = letters.filter(x => x !== '<BOS>' && x !== '<EOS>'
          && x !== '<PAD>');
        tashkeel = this.filterTashkeel(tashkeel);
        tashkeel = tashkeel.filter(x => x !== '<BOS>' && x !== '<EOS>'
          && x !== '<PAD>');
        const n = Math.min(letters.length, tashkeel.length);
        const pairs = [];
        for (let i = 0; i < n; i++) pairs.push([letters[i], tashkeel[i]]);
        const bwText = this.combineTashkeelWithText(pairs);
        arTexts.push(transliterateText(bwText, 'bw2ar'));
      }
      return arTexts;
    }
  }

  // ========================================================================
  // CATTModel.predict — الجلسة تُمرَّر من الخارج (onnxruntime-web)
  // ========================================================================
  /**
   * predict(ortSession, text) — واجهة غير متزامنة.
   * ortSession: جلسة onnxruntime-web على catt_eo.onnx
   */
  async function predict(ortSession, text) {
    text = removeNonArabic(text);
    const ort = globalThis.ort;
    const tokenizer = predict._tokenizer
      ? predict._tokenizer : (predict._tokenizer = new TashkeelTokenizer());

    let [inputIds] = tokenizer.encode(text);
    inputIds = inputIds.slice(1, -1);        // [:, 1:-1] — نزع BOS/EOS

    const results = await ortSession.run({
      'in_token_ids': new ort.Tensor('int64',
        BigInt64Array.from(inputIds.map(x => BigInt(x))),
        [1, inputIds.length]),
    }, ['out_token_ids']);
    const logits = results.out_token_ids;   // [1, T, 18] float32
    const dims = logits.dims;
    const data = logits.data;                // Float32Array
    const nTags = dims[dims.length - 1];
    const yPred = new Int32Array(dims[1]);
    for (let t = 0; t < dims[1]; t++) {
      let best = 0, bestV = -Infinity;
      const off = t * nTags;
      for (let k = 0; k < nTags; k++) {
        if (data[off + k] > bestV) { bestV = data[off + k]; best = k; }
      }
      yPred[t] = best;
    }

    // المسافة تبقى بلا تشكيل (lettersMap[' '] == inputIds → <NT>)
    const spaceId = tokenizer.lettersMap[' '];
    const ntId = tokenizer.tashkeelMap[tokenizer.noTashkeelTag];
    for (let i = 0; i < inputIds.length; i++) {
      if (inputIds[i] === spaceId) yPred[i] = ntId;
    }

    const texts = tokenizer.decode([inputIds], [Array.from(yPred)]);
    return texts[0];
  }

  globalThis.CattTashkeel = {
    TashkeelTokenizer, predict, removeNonArabic,
    stripTashkeel, stripTatweel,
    transliterateText, transliterateWord,
  };
})();
