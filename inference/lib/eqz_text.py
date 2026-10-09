#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eqz_text.py — التطبيع الموحد للنص قبل التشكيل والترميز (Eiqaz v1)
=============================================================================
مصدر الحقيقة الوحيد لمسار ما قبل النموذج — تستخدمه الواجهة (webapp.py)
وواجهة السطر (infer.py CLI) معًا، فلا يمكن أن يتباعدا (توحيد preprocessing).

المراحل (بنفس الترتيب الذي طلبه التصميم):
    digits → spoken Arabic words → punctuation normalization
    → Egyptian/Fusha diacritization → G2P (eqz_tokens) → model

القواعد مبنية حصريًا على أدلة المشروع القائمة:
  * تحويل الترقيم ، → , و ؟ → ? — القاعدة المعتمدة الوحيدة
    (ph2_punct_convert.py §أ-1/§ل-1/§ن-8 من دليل التشكيل v2.1). الرموز
    الأخرى تبقى كما هي — وG2P يُسقطها تمامًا كما رآها النموذج في التدريب.
  * التوكنز الحقيقية للترقيم في جدول الرموز: . , ? ! فقط (ids 5-8).
  * أرقام التدريب كانت تختفي صامتة (572 نصًا فيها أرقام، صفر نطق) —
    هنا تُحوَّل إلى كلمات منطوقة قبل أي مرحلة، بأشكال مصرية موثقة من
    corpus التدريب نفسه (واحد 403 / اتنين 53 / مية 38 / تلاتة 33 /
    خمسة 32 / عشرين 29 / خمسين 13 / خمستاشر 5 / في المية 24 / نقطة 477).
  * النصوص المختلطة: corpus التدريب صفر كلمات لاتينية، فالكلمة اللاتينية
    تُنقحر حرفيًا إلى عربية (meeting → ميتينج) لتبقى منطوقة لا مختفية.

لا يعدل هذا الملف الأوزان ولا الـcheckpoint ولا tokenizer التدريب.
"""
import re

# ---------------------------------------------------------------------------
# ثوابت الترقيم — القاعدة المعتمدة فقط (ph2_punct_convert.py)
# ---------------------------------------------------------------------------
AR_COMMA = '\u060C'      # ،
AR_QMARK = '\u061F'      # ؟
AR_SEMI = '\u061B'       # ؛
REAL_PUNCT_TOKENS = '. , ? !'      # التوكنز المدربة (symbols.py ids 5-8)
_PUNCT_CHARS = '. , ? ! ؛ : ; " \' ( ) « » … — - ٪ %'.replace(' ', '')
_ATTACH_PUNCT = '.,?!' + AR_SEMI + ':'        # تلحق بالكلمة السابقة

_EASTERN_DIGITS = str.maketrans('٠١٢٣٤٥٦٧٨٩', '0123456789')

# ---------------------------------------------------------------------------
# علامات القاف {ق}/{ج}/{ء} — ربطها بالكلمة قبل أي معالجة (PATCH 12)
# + توسيع علامة البادئة «{ق}انون» → «قانون{ق}» (إدراج الحرف + وسم الكلمة)
# ---------------------------------------------------------------------------
_TAG_BIND_RE = re.compile(
    r'([\u0621-\u063A\u0641-\u064A\u064B-\u0652])\s+(\{[^{}]{1,2}\})')
_TAG_ANY_RE = re.compile(r'\{[^{}]{1,2}\}')
_PREFIX_MARKER_RE = re.compile(
    r'\{([\u0642\u062C\u0621\u0623\u06AFqhg])\}'
    r'([\u0621-\u064A][\u0621-\u064A\u064B-\u0652]*)')
_PREFIX_INSERT = {'\u0642': '\u0642', '\u062C': '\u062C', '\u06AF': '\u062C',
                  '\u0621': '\u0623', '\u0623': '\u0623', 'q': '\u0642',
                  'g': '\u062C', 'h': '\u0623'}


def expand_prefix_markers(text):
    """«{ق}انون» → «قانون{ق}» — علامة البادئة تُوسَّع إلى إدراج الحرف نفسه
    في أول الكلمة + وسم كلمة عادي (يُحلل عند الترميز بـ eqz_tokens).
    الكلمة التي تبدأ بالحرف أصلًا لا يُكرَّر فيها («{ق}قانون» → «قانون{ق}»)."""
    def _sub(m):
        marker, word = m.group(1), m.group(2)
        insert = _PREFIX_INSERT.get(marker, marker)
        if word.startswith(insert):
            return word + '{' + marker + '}'
        return insert + word + '{' + marker + '}'
    return _PREFIX_MARKER_RE.sub(_sub, text)


def bind_markers_to_words(text):
    """لصق الوسم المنفصل بكلمته («كلمة {ق}» → «كلمة{ق}») — PATCH 12."""
    return _TAG_BIND_RE.sub(r'\1\2', text)


def _split_markers(text):
    """يفك الوسوم عن الكلمات. يعيد (كلمة، وسم) لكل كلمة — العدد محفوظ."""
    out = []
    for w in text.split():
        m = None
        for mm in _TAG_ANY_RE.finditer(w):
            if _is_marker(mm.group(0)):
                m = mm.group(0)
        out.append((w.replace(m, '') if m else w, m or ''))
    return out


def _is_marker(tag):
    inner = tag[1:-1]
    return any(c in 'قءأجگqhg' for c in inner) and len(inner) <= 2


def _rejoin_markers(words_with_tags):
    return ' '.join(w + t if t else w for w, t in words_with_tags)


# ---------------------------------------------------------------------------
# الأرقام → كلمات عربية منطوقة (أشكال corpus التدريب المصري)
# ---------------------------------------------------------------------------
EGY_UNITS = {1: 'واحد', 2: 'اتنين', 3: 'تلاتة', 4: 'أربعة', 5: 'خمسة',
             6: 'ستة', 7: 'سبعة', 8: 'تمانية', 9: 'تسعة'}
EGY_TEENS = {11: 'حداشر', 12: 'اتناشر', 13: 'تلتاشر', 14: 'اربعتاشر',
             15: 'خمستاشر', 16: 'ستاشر', 17: 'سبعتاشر', 18: 'تمنتاشر',
             19: 'تسعتاشر'}
EGY_TENS = {2: 'عشرين', 3: 'تلاتين', 4: 'أربعين', 5: 'خمسين', 6: 'ستين',
            7: 'سبعين', 8: 'تمانين', 9: 'تسعين'}
EGY_HUNDREDS = {1: 'مية', 2: 'ميتين', 3: 'تلاتمية', 4: 'اربعمية',
                5: 'خمسمية', 6: 'ستمية', 7: 'سبعمية', 8: 'تمانمية',
                9: 'تسعمية'}
EGY_THOUSAND = {1: 'ألف', 2: 'ألفين'}
EGY_MILLION = {1: 'مليون', 2: 'مليونين'}
EGY_MILLIARD = {1: 'مليار', 2: 'مليارين'}

MSA_UNITS = {1: 'واحد', 2: 'اثنين', 3: 'ثلاثة', 4: 'أربعة', 5: 'خمسة',
             6: 'ستة', 7: 'سبعة', 8: 'ثمانية', 9: 'تسعة'}
MSA_TEENS = {11: 'أحد عشر', 12: 'اثنا عشر', 13: 'ثلاثة عشر',
             14: 'أربعة عشر', 15: 'خمسة عشر', 16: 'ستة عشر',
             17: 'سبعة عشر', 18: 'ثمانية عشر', 19: 'تسعة عشر'}
MSA_TENS = {2: 'عشرون', 3: 'ثلاثون', 4: 'أربعون', 5: 'خمسون', 6: 'ستون',
            7: 'سبعون', 8: 'ثمانون', 9: 'تسعون'}
MSA_HUNDREDS = {1: 'مائة', 2: 'مئتان', 3: 'ثلاثمائة', 4: 'أربعمائة',
                5: 'خمسمائة', 6: 'ستمائة', 7: 'سبعمائة', 8: 'ثمانمائة',
                9: 'تسعمائة'}
MSA_THOUSAND = {1: 'ألف', 2: 'ألفان'}
MSA_MILLION = {1: 'مليون', 2: 'مليونان'}
MSA_MILLIARD = {1: 'مليار', 2: 'ملياران'}
MSA_PLURALS = {('th', 3): 'آلاف', ('th', 5): 'ألف', ('mn', 3): 'ملايين',
               ('mn', 5): 'مليون', ('md', 3): 'مليارات', ('md', 5): 'مليار'}
EGY_PLURALS = {('th', 3): 'آلاف', ('th', 5): 'ألف', ('mn', 3): 'ملايين',
               ('mn', 5): 'مليون', ('md', 3): 'مليارات', ('md', 5): 'مليار'}

_DIGIT_WORDS_EGY = {0: 'صفر', **EGY_UNITS}
_DIGIT_WORDS_MSA = {0: 'صفر', **MSA_UNITS}


def _under_100(n, egy):
    """0-99 → كلمات. الصيغة المصرية أو الفصحى."""
    if n == 0:
        return ''
    if n < 10:
        return (EGY_UNITS if egy else MSA_UNITS)[n]
    if 11 <= n <= 19:
        return (EGY_TEENS if egy else MSA_TEENS)[n]
    if n == 10:
        return 'عشرة'
    tens, unit = divmod(n, 10)
    t = (EGY_TENS if egy else MSA_TENS)[tens]
    u = (EGY_UNITS if egy else MSA_UNITS)[unit] if unit else ''
    return f'{u} و{t}' if u else t


def _under_1000(n, egy):
    """0-999 → كلمات."""
    out = []
    h, rest = divmod(n, 100)
    if h:
        out.append((EGY_HUNDREDS if egy else MSA_HUNDREDS)[h])
    if rest:
        out.append(_under_100(rest, egy))
    return ' و'.join(out)


def _big_scale(n, one, two, plurals, egy):
    """مساعدة النطاقات الكبيرة (آلاف/ملايين/مليارات).

    [إصلاح معزول موثق 2026-10-09 — اكتشفه منفذ JS للمتصفح]: كان
    plurals[(3,)] بمفتاح صفّي (tuple) بينما المتصل يمرر قاموسًا بمفاتيح
    int {3:…, 5:…} — KeyError يسقط أي طلب يحوي عددًا ≥ مليون (لم يظهر
    في الاختبارات الذاتية لأنها لا تتجاوز 12345). الصواب البنيوي:
    plurals[3]. لا يمس أي نطق — نفس الصيغ الجمعية المقصودة."""
    out = []
    if n == 1:
        out.append(one)
    elif n == 2:
        out.append(two)
    elif 3 <= n <= 10:
        out.append(f'{_under_100(n, egy)} {plurals[3]}')
    else:
        out.append(f'{_under_1000(n, egy)} {plurals[5]}')
    return ' '.join(out)


def number_to_arabic_words(n, egy=True):
    """عدد صحيح ≥ 0 → كلمات عربية منطوقة (مصري أو فصحى)."""
    if n == 0:
        return 'صفر'
    if egy:
        th_map, mn_map, md_map = EGY_THOUSAND, EGY_MILLION, EGY_MILLIARD
        pl = EGY_PLURALS
    else:
        th_map, mn_map, md_map = MSA_THOUSAND, MSA_MILLION, MSA_MILLIARD
        pl = MSA_PLURALS
    parts = []
    milliard, rest = divmod(n, 10 ** 9)
    million, rest = divmod(rest, 10 ** 6)
    thousand, rest = divmod(rest, 10 ** 3)
    if milliard:
        parts.append(_big_scale(milliard, md_map[1], md_map[2],
                                {3: pl[('md', 3)], 5: pl[('md', 5)]}, egy))
    if million:
        parts.append(_big_scale(million, mn_map[1], mn_map[2],
                                {3: pl[('mn', 3)], 5: pl[('mn', 5)]}, egy))
    if thousand:
        if thousand == 1:
            parts.append(th_map[1])
        elif thousand == 2:
            parts.append(th_map[2])
        elif 3 <= thousand <= 10:
            parts.append(f'{_under_100(thousand, egy)} {pl[("th", 3)]}')
        else:
            parts.append(f'{_under_1000(thousand, egy)} {pl[("th", 5)]}')
    if rest:
        parts.append(_under_1000(rest, egy))
    return ' و'.join(parts)


def _digits_to_words(s, egy):
    """سلسلة أرقام مفردة (IP/إصدار) → كل رقم باسمه: 192 → واحد تسعة اتنين."""
    dw = _DIGIT_WORDS_EGY if egy else _DIGIT_WORDS_MSA
    return ' '.join(dw[int(c)] for c in s)


# ---------------------------------------------------------------------------
# الكشف عن السياقات الرقمية
# ---------------------------------------------------------------------------
# أرقام بفواصل غربية للتلاف (1,000 / 12,500,300)
_RE_WESTERN_GRP = re.compile(r'\b(\d{1,3}(?:,\d{3})+)\b')
# رقم + نسبة (5% / 80% / 5.5% / ٪)
_RE_PERCENT = re.compile(r'(\d+(?:\.\d+)?)\s*[%٪]')
# رقم عشري (3.5) — بلا نسبة (تعالج قبلها)
_RE_DECIMAL = re.compile(r'(?<![\d.])(\d+)\.(\d+)(?![\d.])')
# مجاميع رقمية متعددة النقاط (192.168.1.1 / 2.5.1)
_RE_MULTIDOT = re.compile(r'\b(\d+(?:\.\d+){2,})\b')
# رقم صحيح متبقي
_RE_INT = re.compile(r'\d+')


def _read_decimal(int_part, frac_part, egy):
    """3.5 → «تلاتة ونص» (خمسة عشرية فقط) — وإلا فاصل."""
    if frac_part.rstrip('0') == '5':
        base = _under_1000(int(int_part), egy) if int_part else ''
        return (base + ' ونص').strip()
    int_w = _under_1000(int(int_part), egy) if int_part else 'صفر'
    frac_w = _digits_to_words(frac_part, egy)
    return f'{int_w} فاصل {frac_w}'


def _convert_numbers_in(text, egy):
    """كل الأرقام في النص → كلمات منطوقة (لا يختفي أي رقم)."""
    # 0) التلاف الغربية: 1,000 → 1000 (فقط مجموعات من 3 بالضبط)
    def _ungrp(m):
        return m.group(1).replace(',', '')
    text = _RE_WESTERN_GRP.sub(_ungrp, text)

    # 1) نسب مئوية أولاً: 80% → «تمانين في المية»
    def _pct(m):
        num = m.group(1)
        if '.' in num:
            words = _read_decimal(*num.split('.'), egy)
        else:
            words = _under_1000(int(num), egy) or 'صفر'
        return f'{words} في المية'
    text = _RE_PERCENT.sub(_pct, text)

    # 2) متعدد النقاط (IP/إصدار): 192.168.1.1 → أرقام مفردة بنقطة
    def _mdot(m):
        groups = m.group(1).split('.')
        parts = [_digits_to_words(g, egy) for g in groups]
        return ' نقطة '.join(parts)
    text = _RE_MULTIDOT.sub(_mdot, text)

    # 3) عشري مفرد: 3.5 / 5.4
    def _dec(m):
        return ' ' + _read_decimal(m.group(1), m.group(2), egy) + ' '
    text = _RE_DECIMAL.sub(_dec, text)

    # 4) أعداد صحيحة متبقية
    def _int(m):
        return ' ' + number_to_arabic_words(int(m.group(0)), egy) + ' '
    text = _RE_INT.sub(_int, text)

    return ' '.join(text.split())


# ---------------------------------------------------------------------------
# نقحرة لاتينية → عربية (نصوص مختلطة — meeting → ميتينج)
# ---------------------------------------------------------------------------
_LATIN_DIGRAPHS = {
    'sh': 'ش', 'ch': 'تش', 'th': 'ث', 'gh': 'غ', 'ph': 'ف',
    'kh': 'خ', 'oo': 'و', 'ee': 'ي', 'ck': 'ك', 'qu': 'كو',
}
_LATIN_SINGLE = {
    'a': 'ا', 'b': 'ب', 'c': 'ك', 'd': 'د', 'e': 'ي', 'f': 'ف', 'g': 'ج',
    'h': 'ه', 'i': 'ي', 'j': 'ج', 'k': 'ك', 'l': 'ل', 'm': 'م', 'n': 'ن',
    'o': 'و', 'p': 'ب', 'q': 'ك', 'r': 'ر', 's': 'س', 't': 'ت', 'u': 'و',
    'v': 'ف', 'w': 'و', 'x': 'كس', 'y': 'ي', 'z': 'ز',
}
# استثناءات شائعة للنقحرة (كلمات مختلطة متكررة في الاستخدام المصري)
_LATIN_EXCEPTIONS = {
    'ok': 'أوكي', 'okay': 'أوكي', 'wifi': 'واي فاي', 'wi-fi': 'واي فاي',
    'hello': 'ألو', 'hi': 'هاي', 'app': 'آب', 'email': 'إيميل',
    'iphone': 'آيفون', 'android': 'أندرويد', 'windows': 'ويندوز',
    'google': 'جوجل', 'youtube': 'يوتيوب', 'facebook': 'فيسبوك',
    'deadline': 'ديدلاين', 'meeting': 'ميتينج', 'report': 'ريبورت',
    'update': 'أبديت', 'download': 'دونلود', 'upload': 'أبلود',
    'link': 'لينك', 'post': 'بوست', 'story': 'ستوري',
}
_LATIN_WORD = re.compile(r"[A-Za-z]+")
_HAS_LATIN = re.compile(r'[A-Za-z]')


def _translit_word(w):
    wl = w.lower()
    if wl in _LATIN_EXCEPTIONS:
        return _LATIN_EXCEPTIONS[wl]
    out = []
    i = 0
    while i < len(wl):
        two = wl[i:i + 2]
        if two in _LATIN_DIGRAPHS:
            out.append(_LATIN_DIGRAPHS[two])
            i += 2
            continue
        out.append(_LATIN_SINGLE.get(wl[i], ''))
        i += 1
    return ''.join(out)


def transliterate_latin(text):
    """الكلمات اللاتينية → نقحرة عربية. العربية لا تُمس."""
    if not _HAS_LATIN.search(text):
        return text
    return _LATIN_WORD.sub(lambda m: _translit_word(m.group(0)), text)


# ---------------------------------------------------------------------------
# التطبيع الكامل
# ---------------------------------------------------------------------------
_RE_PUNCT_RUN = re.compile(r'([.,!?؛:])\1+')
_RE_STANDALONE_PUNCT = re.compile(r'(?:^|\s)([.,!?؛:،؟]+)(?=\s|$)')


def _attach_standalone_punct(text):
    """الترقيم المنفصل يلحق بالكلمة السابقة (يحفظ عدد الكلمات لمحاذاة det)."""
    out = []
    for w in text.split():
        if w and all(c in '.,!?؛:' for c in w):
            if out:
                out[-1] = out[-1] + w
            else:
                out.append(w)
        else:
            out.append(w)
    return ' '.join(out)


def normalize_punctuation(text):
    """، → , و ؟ → ? (القاعدة المعتمدة فقط) + رصّ التكرار."""
    text = text.replace(AR_COMMA, ',').replace(AR_QMARK, '?')
    text = _RE_PUNCT_RUN.sub(r'\1', text)          # !!! → ! و ؟؟ → ؟
    return text


def normalize_text(raw_text, dialect='egy', egy=None):
    """التطبيع الموحد: أرقام → كلمات، ترقيم، نقحرة لاتينية.

    dialect: 'egy' (أشكال مصرية) أو 'msa' (أشكال فصحى).
    يعيد النص المطبع — بلا تشكيل وبلا حذف لأي محتوى منطوق."""
    if egy is None:
        egy = dialect != 'msa'
    text = raw_text.translate(_EASTERN_DIGITS)
    text = expand_prefix_markers(text)
    text = bind_markers_to_words(text)
    text = normalize_punctuation(text)
    text = _convert_numbers_in(text, egy)
    text = _attach_standalone_punct(text)   # بعد الأرقام: «عشرة .» → «عشرة.»
    text = transliterate_latin(text)
    return ' '.join(text.split())


# ---------------------------------------------------------------------------
# الإبقاء على العربية + الترقيم المدرب فقط (بديل keep_arabic_only)
# ---------------------------------------------------------------------------
_AR_AND_PUNCT = re.compile(
    r'[^\u0621-\u063A\u0641-\u064A\u064B-\u0652 .,?!]')


def keep_arabic_and_punct(text):
    """إبقاء الحروف العربية والحركات والمسافات والترقيم المدرب (. , ? !).

    هذا هو سلوك التدريب الصافي: catt_eo يطبع غير العربي قبل التشكيل ثم
    يعيده det بالمحاذاة، وG2P يُسقط كل ما ليس في جدول الرموز — فتصل
    النموذج الحروف والحركات و . , ? ! فقط، بلا حذف للترقيم المدرب."""
    text = text.replace('\u0640', '')               # التطويل
    text = _AR_AND_PUNCT.sub(' ', text)
    return ' '.join(text.split())


# توافق قديم — الاسم القديم كان يحذف الترقيم (خلل التدريب/الاستدلال)
def keep_arabic_only(text):
    """توافق قديم — الآن يُبقي الترقيم المدرب (., ? !) أيضًا."""
    return keep_arabic_and_punct(text)


# ---------------------------------------------------------------------------
# استرجاع الترقيم فوق مخرج catt (مسار الفصحى — بلا قواعد det المصرية)
# ---------------------------------------------------------------------------
_AR_LETTER_CH = lambda c: '\u0621' <= c <= '\u064A' and c != '\u0640'
_LEAD_PUNCT = '.,!?؛:"()«»\u060C\u061F\u061B-…'
_TAIL_PUNCT = '.,!?؛:"()«»\u060C\u061F\u061B-…%٪'


def restore_punctuation(raw_norm, catt_out):
    """إعادة الترقيم المرفق بالكلمات إلى مخرج catt (الذي قصّه remove_non_arabic).

    المحاذاة موضعية: نفس عدد الكلمات (مضمون بعد normalize_text — النقحرة
    والأرقام كلمات عربية، والترقيم ملحق بالكلمات). تُضاف حواف الترقيم
    الخام حول كلمة catt المشكولة، وتُحذف الكلمات غير القابلة للمحاذاة."""
    raw_ws, catt_ws = raw_norm.split(), catt_out.split()
    if len(raw_ws) != len(catt_ws):
        return catt_out                     # تعذر المحاذاة — مخرج catt كما هو
    out = []
    for r, c in zip(raw_ws, catt_ws):
        i, j = 0, len(r)
        while i < j and r[i] in _LEAD_PUNCT:
            i += 1
        while j > i and r[j - 1] in _TAIL_PUNCT:
            j -= 1
        out.append(r[:i] + c + r[j:])
    return ' '.join(out)


if __name__ == '__main__':
    # فحص ذاتي سريع بلا اعتماديات
    tests = [
        ('عندي 5 كتب', 'عندي خمسة كتب'),
        ('عمره 10 سنين', 'عمره عشرة سنين'),
        ('عام 2026', 'عام ألفين وستة وعشرين'),
        ('250 جنيه', 'ميتين وخمسين جنيه'),
        ('12345', 'اتناشر ألف وتلاتمية وخمسة وأربعين'),
        ('٥ كتب', 'خمسة كتب'),
        ('١٠ سنين', 'عشرة سنين'),
        ('٢٥٠', 'ميتين وخمسين'),
        ('٢٠٢٦', 'ألفين وستة وعشرين'),
        ('الساعة 10.', 'الساعة عشرة.'),
        ('استهلكت 80% من الباقة', 'استهلكت تمانين في المية من الباقة'),
        ('3.5', 'تلاتة ونص'),
        ('5.4', 'خمسة فاصل أربعة'),
        ('192.168.1.1', 'واحد تسعة اتنين نقطة واحد ستة تمانية نقطة واحد نقطة واحد'),
        ('1,000 جنيه', 'ألف جنيه'),
        ('meeting بكرة', 'ميتينج بكرة'),
        ('أنا رايح أعمل meeting بكرة', 'أنا رايح أعمل ميتينج بكرة'),
        ('{ق}انون', 'قانون{ق}'),
        ('{ق}رآن', 'قرآن{ق}'),
        ('{ق}قانون', 'قانون{ق}'),
        ('هل أنت جاهز؟', 'هل أنت جاهز?'),
        ('ممتاز! نبدأ الآن.', 'ممتاز! نبدأ الآن.'),
        ('دلوقتي هنبدأ الدرس، يا عمر.', 'دلوقتي هنبدأ الدرس, يا عمر.'),
    ]
    ok = 0
    for raw, want in tests:
        got = normalize_text(raw)
        mark = 'PASS' if got == want else 'FAIL'
        ok += got == want
        print(f'[{mark}] {raw!r} -> {got!r}' + ('' if got == want
                                                else f'  (want {want!r})'))
    print(f'{ok}/{len(tests)} PASS')
