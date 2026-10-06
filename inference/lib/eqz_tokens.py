#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eqz_tokens.py — Eiqaz v1 tokenizer policy  (Eiqaz TTS training project v1)
=============================================================================

سياسة القاف ثلاثية الاتجاه للنموذج الجديد (تدريب من الصفر) — مبنية على
قاموس النطق egyptian_q_default_lexicon_v1.json (مهمة الاكتشاف 2026-10-04؛
المصنفات مطابقة لتوجيه المستخدم 2026-10-04):

  q_default      (20 عائلة)  -> توكن 'q'  (قاف أصيلة — ستُدرَّب)
  g_default      (6 عائلة)  -> توكن 'v'  (جيم مصرية [g])
  hamza_default (صريحة+الذيل) + context_dependent + uncertain + غير المعجم
                                -> توكن '<' (همزة قاهرية — الافتراضي)
  context_dependent/uncertain: لا تُفرض أبدًا («لا تحول الحالات السياقية
  إلى قاعدة مطلقة») — تسقط للافتراضي همزة.

علامات التحكم الصريح (تفوق كل القوائم والأوضاع دائمًا — نظام Eiqaz محفوظ):
  كلمة{ق} / كلمة {ق} / كلمة{q}  -> توكن 'q'  (قاف أصيلة)
  كلمة{ج} / كلمة {ج} / كلمة{گ}  -> توكن 'v'  (جيم مصرية [g])
  كلمة{ء} / كلمة {ء} / كلمة{أ}  -> توكن '<' (همزة)

المسارات:
  toks_egy(text) : مصري — قرارات القاف على مستوى الكلمة في باكوالتير ثم
                   EGY_SOUND_MAP (بلا 'q' — الفرق الجوهري عن NileTTS
                   القديم حيث q->'<' حتمًا: 0/1,590,209 توكن q مدرَّب).
  toks_ms(text)  : فصحى — بلا خريطة؛ q تبقى 'q' (مدربة من MSA)، ج='j'
                   [dʒ]؛ العلامات: {ق}->'q'، {ء}->'<'، {ج}->'v'.
  tokens_to_ids  : من tts_arabic.text كما هي.

الأصل الهندسي: دوال tts_arabic (nipponjo) + منطق fix_qaf/_qaf_skel_
variants من inference/infer.py (PATCH 6..14) منسوخة ومعدلة للسياسة
الجديدة. مشروع NileTTS 4h Train القديم لم يُمَس.

الهياكل أدناه مولدة آليًا (scripts/eqz_build_tokens.py) من القاموس:
كلمة كل عائلة مجردة + هياكل corpus الفعلية كما وردت في المسح الشامل
(21,854 نصًا) — لا تعارض بين forced_q و forced_g (فُحص exact+variants).
v1.0 — 2026-10-04.
"""
import re

FORCED_Q_SKELETONS = frozenset({
    'AlAnqsAm', 'AlHqyqp', 'AlHqyqy', 'AlHqyqyp',
    'Al^qAfAt', 'Al^qAfp', 'Al^qAfy', 'Al^qAfyp',
    'AlmnTq', 'AlmnTqy', 'AlmnTqyp', 'Almstqym',
    'Almstqymp', 'AlqAbD', 'AlqAnwn', 'AlqAnwny',
    'AlqAnwnyp', 'AlqSS', 'AlqSSy', 'AlqSp',
    'AlqT', 'AlqTAE', 'AlqTE', 'AlqTEp',
    'Alqdws', 'AlqhAr', 'Alqr|n', 'Alqsmp',
    'AlqwAnyn', 'Alqywm', 'Altqryb', 'AnqsAm',
    'Hqyqp', 'Hqyqy', 'Hqyqyp', '^qAfAt',
    '^qAfp', '^qAftnA', '^qAfy', '^qAfyp',
    'bAlHqyqp', 'bAlmnTq', 'bAlqSS', 'bAlqSp',
    'bAlqTE', 'bHqyqp', 'bmnTq', 'bmnTqyp',
    'bqTEp', 'bqwAnynh', 'btqrybA', 'btqwY',
    'fAlqSp', 'fqT', 'kqTE', 'l^qAfp',
    'llqSp', 'llqTEp', 'lqSS', 'lqSp',
    'lqTAE', 'mnTq', 'mnTqy', 'mnTqyp',
    'mstqym', 'mstqymp', 'qAnwn', 'qAnwny',
    'qAnwnyA', 'qAnwnyp', 'qSS', 'qSp',
    'qT', 'qTAE', 'qTE', 'qTEp',
    'qTEtyn', 'qTb', 'qTk', 'qry$',
    'qr|n', 'qsmp', 'qwAnyn', 'qwAnynh',
    'qwAnynhA', 'tqryb', 'tqrybA', 'tqryby',
    'tqwY', 'wAlHqyqp', 'wAl^qAfp', 'wAlmnTqy',
    'wHqyqy', 'w^qAfp', 'wmnTqy', 'wmnTqyp',
    'wqAbD', 'wqSp',
})

FORCED_G_SKELETONS = frozenset({
    '>rqAm', 'Al>rqAm', 'AlArqAm', 'AlmqAl',
    'AlmqAlAt', 'AlmqAm', 'Alrqm', 'bAl>rqAm',
    'bAlrqm', 'brqm', 'krqm', 'l>rqAm',
    'lrqm', 'mqAl', 'mqAlAt', 'mqAm',
    'qnTAr', 'qr$', 'qr$yn', 'qyrAT',
    'rqm', 'wrqm',
})

Q_FAMILY_WORDS = ['القرآن', 'فقط', 'قانون', 'قطعة', 'قصة', 'قطب', 'حقيقة', 'تقريب', 'قسمة', 'منطق', 'ثقافة', 'تقوى', 'القدوس', 'القهّار', 'القيّوم', 'القابض', 'قريش', 'مستقيم', 'انقسام', 'قطاع']
G_FAMILY_WORDS = ['رقم', 'قرش', 'مقال', 'قيراط', 'قنطار', 'مقام']

# حارس «بقرأ» التقدمية (PATCH 6b — مستقل دائمًا): بقرأ/بتقرأ/بيقرأ/بيتقرأ
# (و/بقرأ) = همزة دائمًا مهما احتوت أي قائمة مستقبلًا.
_QAF_PROGRESSIVE_BAQR = re.compile(r'^[wf]?b(?:yt|y|t)?qr>')

_QAF_DIAC = frozenset('auiFNK~o')
# ترقيم باكوالتير الملتصق بالكلمات (زائد الشرطة) — يُستبعد من الهيكل:
# الخطأ المُثبَت (2026-10-05): 'حَقِيقَة{ق}.' → هيكل 'Hqyqp.' لا يطابق
# 'Hqyqp' فتسقط العلامة والقاموس معًا إلى الهمزة الافتراضية. (ة='p' حرف
# تبقى — لا تُمس.)
_QAF_PUNCT = frozenset('.,!?;:"\'،؛…()\u2014-')
_QAF_CLITICS = ('w', 'f', 'b', 'l', 'k')

_QAF_MARKER_MAP = {'ق': 'q', 'q': 'q', 'ء': 'h', 'أ': 'h', 'h': 'h',
                   'ج': 'g', 'گ': 'g', 'g': 'g'}
_QAF_MARKER_RE = re.compile(
    r'([\u0621-\u063A\u0641-\u064A][\u0621-\u063A\u0641-\u064A\u064B-\u0652]*)'
    r'\s*\{([^{}]{1,2})\}')
_QAF_MARKER_TAG_RE = re.compile(r'\{([^{}]{1,2})\}')
_PUNCT_SPACE_RE = re.compile(r'\s+([،؛,.!؟:…]+)')

# توحيد كانوني لترتيب الحركات/التنوين مع الشدة (v1.0.1):
# corpus التشكيل يكتب «حركة ثم شدة» في 93.6% من النصوص (61,791 موضعًا
# مصريًا + 16,819 MSA — قياس 2026-10-04) بينما محول باكوالتير يتوقع
# «شدة ثم حركة». الرسمان تسلسل واحد بالنطق — التبديل كانوني صرف.
_DIAC_SWAP_RE = re.compile('([ً-ِ])(ّ)')


def normalize_diacritic_order(text):
    """(حركة/تنوين + شدة) → (شدة + حركة) — ترتيب كانوني بلا تغيير نطق."""
    return _DIAC_SWAP_RE.sub(r'}', text)

# خريطة الأصوات المصرية — الفرق الجوهري عن NileTTS القديم:
# 'q' ليست في الخريطة! كل قرارات القاف تتم على مستوى الكلمة في باكوالتير
# (fix_qaf_v1)؛ ما يبقى 'q' يمر خامًا كتوكن q مدرَّب.
# (القديمة: EGY_TOKEN_MAP={'j':'v','q':'<','^':'t','*':'d'} → صفر توكن q)
EGY_SOUND_MAP = {'j': 'v', '^': 't', '*': 'd'}

_TTS_ARABIC_IMPORTED = False
_arabic_to_buckwalter = None
_buckwalter_to_phonemes = None
_phonemes_to_tokens = None
_tokens_to_ids = None


def _ensure_tts_arabic(paths=None):
    """استيراد كسول لواجهات tts_arabic (مع مسارات بحث اختيارية)."""
    global _TTS_ARABIC_IMPORTED, _arabic_to_buckwalter, _buckwalter_to_phonemes
    global _phonemes_to_tokens, _tokens_to_ids
    if _TTS_ARABIC_IMPORTED:
        return
    if paths:
        for p in paths:
            if p not in sys.path:
                sys.path.insert(0, p)
    from tts_arabic.text import (arabic_to_buckwalter,
                                 buckwalter_to_phonemes,
                                 phonemes_to_tokens, tokens_to_ids)
    _arabic_to_buckwalter = arabic_to_buckwalter
    _buckwalter_to_phonemes = buckwalter_to_phonemes
    _phonemes_to_tokens = phonemes_to_tokens
    _tokens_to_ids = tokens_to_ids
    _TTS_ARABIC_IMPORTED = True


import sys  # noqa: E402  (يُستورد أعلاه ضمنيًا في الاستخدام العادي)


def _skel_variants(s):
    """صيغ المطابقة المحتملة لهيكل الكلمة: كما هو، أو بعد نزع «ال»،
    أو سابقة اتصال (و/ف/ب/ل/ك) [+ «ال»]، مع ه↔ة ونزع ألف تنوين النصب
    (نسخة من infer.py _qaf_skel_variants — PATCH 11/12)."""
    out = {s}
    if s.startswith('Al') and len(s) > 3:
        out.add(s[2:])
    for c in _QAF_CLITICS:
        if s.startswith(c) and len(s) > 2:
            out.add(s[1:])
            if s[1:3] == 'Al' and len(s) > 4:
                out.add(s[3:])
    out |= {v[:-1] + 'p' for v in tuple(out)
            if v.endswith('h') and len(v) > 2}
    out |= {v[:-1] for v in tuple(out) if v.endswith('A') and len(v) > 2}
    return out


def _ar_skel_buck(word_ar):
    """هيكل باكوالتير مجرد لكلمة عربية واحدة (مع ه→ة النهائية المصرية).
    يعيد None إن لم تحوِ قافًا أو فشل التحويل — مطابق _ar_skel في infer.py
    (المفتاح بنفس تمثيل هياكل باكوالتير المستخدم في fix_qaf_v1)."""
    _ensure_tts_arabic()
    w = word_ar.strip(".,!?؟؛،:\"'()«»")
    if 'ق' not in w:
        return None
    try:
        b = _arabic_to_buckwalter(w)
    except Exception:                            # noqa: BLE001
        return None
    sk = ''.join(c for c in b if c not in _QAF_DIAC)
    if sk.endswith('h') and len(sk) > 2:          # قطعه → قطة (رسم مصري)
        sk = sk[:-1] + 'p'
    return sk or None


def parse_qaf_markers(text):
    """تحليل علامات النص {ق}/{ج}/{ء}.

    يعيد (نص_نظيف_بلا_علامات, {هيكل: فعل}) — الفعل ∈ {'q','h','g'}.
    العلامة تسند للكلمة السابقة (مباشرة أو بفاصل أبيض) وتسري على كل صيغ
    الهيكل (بال/سوابق). الوسم يُستبدل بمسافة ثم توحد الفراغات — لا لصق
    كلمات أبدًا (PATCH 12)."""
    actions = {}
    for m in _QAF_MARKER_RE.finditer(text):
        word, letters = m.group(1), m.group(2)
        act = _QAF_MARKER_MAP.get(letters) if len(letters) == 1 else None
        if act is None:
            for ch in letters:
                act = _QAF_MARKER_MAP.get(ch)
                if act:
                    break
        if act:
            skel = _ar_skel_buck(word)
            if skel:
                for v in _skel_variants(skel):
                    actions[v] = act
    clean = _QAF_MARKER_TAG_RE.sub(' ', text)
    clean = ' '.join(clean.split())
    clean = _PUNCT_SPACE_RE.sub(r'\1', clean)
    return clean, actions


def _lookup_action(skel, actions):
    if not actions:
        return None
    for v in _skel_variants(skel):
        if v in actions:
            return actions[v]
    return None


def _hamza_word(w):
    """قاف → همزة مع الجيمين المُشدَّدة: '<~' (همزة+شدة) تُكسر المحول
    ('i0i0') — فتُستبدل q~ كاملة بهمزة واحدة (إسقاط الشدة: ~148 موضعًا
    من 2.1M توكن — موثق). q العادية → '<' كما هي."""
    return w.replace('q~', '<').replace('q', '<')


def fix_qaf_v1(buck, word_actions=None):
    """سياسة القاف المصرية v1 — على مستوى الكلمة في نص باكوالتير.

    الأولوية: علامة النص > forced_q > forced_g > افتراضي همزة '<'.
    حارس بقرأ التقدمية يعمل دائمًا (همزة).
    الإخراج: باكوالتير تعدلت فيه قاف الكلمات المتعلمة فقط:
      q يبقى q (عائلات q_default / علامة {ق})
      q -> j (عائلات g_default / علامة {ج}) — يصبح 'v' بعد EGY_SOUND_MAP
      q -> '< (افتراضي/علامة {ء})"""
    if 'q' not in buck:
        return buck
    words = buck.split(' ')
    hit = False
    for i, w in enumerate(words):
        if 'q' not in w:
            continue
        skel = ''.join(
            c for c in w if c not in _QAF_DIAC and c not in _QAF_PUNCT)
        if _QAF_PROGRESSIVE_BAQR.match(skel):
            words[i] = _hamza_word(w)
            hit = True
            continue
        act = _lookup_action(skel, word_actions) if word_actions else None
        if act == 'q':
            continue                       # قاف أصيلة خام — توكن 'q'
        if act == 'g':
            words[i] = w.replace('q', 'j')  # جيم -> 'v'
            hit = True
            continue
        if act == 'h':
            words[i] = _hamza_word(w)      # همزة صريحة
            hit = True
            continue
        variants = _skel_variants(skel)
        if any(v in FORCED_Q_SKELETONS for v in variants):
            continue                       # عائلة q_default — خام 'q'
        if any(v in FORCED_G_SKELETONS for v in variants):
            words[i] = w.replace('q', 'j')  # عائلة g_default — 'v'
            hit = True
            continue
        words[i] = _hamza_word(w)          # الافتراضي — همزة قاهرية
        hit = True
    return ' '.join(words) if hit else buck


def fix_qaf_msa(buck, word_actions=None):
    """علامات النص على مسار الفصحى: {ق} -> خام q، {ء} -> '<'،
    {ج} -> جيم مصرية 'v' (تعريف المستخدم للعلامة). بلا علامة: قاف
    الفصحى خام 'q' (مدربة الآن — لا حاجة لأي تقريب)."""
    if not word_actions:
        return buck
    words = buck.split(' ')
    hit = False
    for i, w in enumerate(words):
        if 'q' not in w:
            continue
        skel = ''.join(c for c in w if c not in _QAF_DIAC
                       and c not in _QAF_PUNCT)
        act = _lookup_action(skel, word_actions)
        if act == 'q':
            continue
        if act == 'h':
            words[i] = w.replace('q~', '<').replace('q', '<')
            hit = True
        elif act == 'g':
            words[i] = w.replace('q', 'j')
            hit = True
    return ' '.join(words) if hit else buck


def get_tokenizers(tts_paths=None):
    """يعيد (toks_ms, toks_egy, tokens_to_ids, parse_qaf_markers).

    toks_egy/text: يقبل النص المصري (مشكولًا) وقد يحتوي علامات {ق}/{ج}/{ء}.
    """
    _ensure_tts_arabic(tts_paths)

    def toks_ms(text):
        clean, actions = parse_qaf_markers(text)
        clean = normalize_diacritic_order(clean)
        buck = fix_qaf_msa(_arabic_to_buckwalter(clean), actions)
        return _phonemes_to_tokens(_buckwalter_to_phonemes(buck))

    def toks_egy(text):
        clean, actions = parse_qaf_markers(text)
        clean = normalize_diacritic_order(clean)
        buck = fix_qaf_v1(_arabic_to_buckwalter(clean), actions)
        toks = _phonemes_to_tokens(_buckwalter_to_phonemes(buck))
        return [EGY_SOUND_MAP.get(t, t) for t in toks]

    return toks_ms, toks_egy, _tokens_to_ids, parse_qaf_markers


def token_counts(toks):
    """إحصاء رموز القاف الثلاثة لقائمة توكنات (للتدقيق)."""
    return {
        'q': sum(1 for t in toks if t == 'q'),
        'v': sum(1 for t in toks if t == 'v'),
        'hamza_from_q': None,   # لا يمكن عده من التوكنات وحدها — انظر الفحص
    }


if __name__ == '__main__':
    # فحص ذاتي سريع بلا tts_arabic: الهياكل فقط
    print('FORCED_Q:', len(FORCED_Q_SKELETONS), 'skeletons')
    print('FORCED_G:', len(FORCED_G_SKELETONS), 'skeletons')
    assert not (FORCED_Q_SKELETONS & FORCED_G_SKELETONS)
    print('self-check OK')
