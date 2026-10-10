# -*- coding: utf-8 -*-
"""det_tashkeel.py — الطبقة الحتمية (بلا LLM) لتصحيح مخرجات catt_eo.

======================================================================
 ⚠️ تحذير تحديد المواقع — اقرأه قبل أي استخدام أو واجهة
======================================================================
هذه الطبقة هي **تحسين جزئي** لمخرجات catt_eo، وليست بديلًا كاملًا عن
التشكيل بالذكاء الاصطناعي (tashkeel-ai). ما تصلحه:
  * الترقيم (، → , و ؟ → ?) والأرقام (استرجاع حرف الخام)
  * نهايات الكلمات: نزع الإعراب، السكون الوقفي، تنظيف ة و نواحي المد
  * التنوين: يُسمح فقط على القائمة المغلقة §ب (190 هيكلًا)، والباقي يُحذف
  * القوائم المغلقة المجمدة: التنوين (ي-1)، الدينية Tier A/C (§ك)،
    الشائعة (§ز/§ح)، العبارات المجمدة (ي-1ج)
  * كسرة ال التعريف والبادئات (§د) + شدة الشمسية الغائبة
  * هو/هي (§ج-5)
ما لا تصلحه (يبقى كما أخرجه catt_eo):
  * **الحركات الداخلية** للكلمات (مثل بِيَقُولُ مقابل بِيْقُولْ) —
    تتطلب فهمًا سياقيًا خارج قدرة القواعد الميكانيكية
  * بادئات و/ب المنفصلة عن ال، والهمزات، واختيار المبني
قواعد القاف [g]/[ʔ] (قرار 2026-09-30، PATCH 6b) تعمل في نواة G2P
داخل infer.py — خارج نطاق هذه الطبقة (القاف حرف واحد في الإملاء).

الملصق الإلزامي في أي واجهة/تقرير:
  AR: «تصحيح جزئي: أرقام وترقيم وقوائم مغلقة ونهايات فقط»
  EN: "Partial fix: numbers, punctuation, closed lists, endings only"

المصدر: دليل التشكيل المصري v2.1 (المعتمد 2026-09-30) + منطق
applyPostFixes المُعتمد من محرك v5.3 (ph3_engine_v53.mjs).
البيانات: lib/det_data/det_lists.json (det-v1.2).
"""

import json
import os
import re

# ---------------------------------------------------------------- ثوابت --
FATHA, DAMMA, KASRA, SUKUN, SHADDA = (
    '\u064E', '\u064F', '\u0650', '\u0652', '\u0651')
FATHATAN, DAMMATAN, KASRATAN = '\u064B', '\u064C', '\u064D'
TANWEEN = {FATHATAN, DAMMATAN, KASRATAN}
VOWELS = {FATHA, DAMMA, KASRA}
DIACS = set('\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652'
            '\u0653\u0654\u0655\u0670')
LONG_VOWELS = set('اوىيآ')
SUN_LETTERS = set('تثدذرزسشصضطظلن')
ALEF_FOLD = str.maketrans({'\u0623': '\u0627', '\u0625': '\u0627',
                           '\u0622': '\u0627'})
TATWEEL = '\u0640'
AR_LETTER = lambda c: '\u0621' <= c <= '\u064A' and c != TATWEEL
DIGIT_AR = {chr(0x0660 + i): str(i) for i in range(10)}
DIGIT_LA = {str(i): chr(0x0660 + i) for i in range(10)}

# ملصقات الواجهة (إلزامية عند أي عرض للمستخدم)
LABEL_AR = ('تصحيح جزئي: أرقام وترقيم وقوائم مغلقة ونهايات فقط — '
            'الحركات الداخلية تبقى من catt_eo إلا قائمة مغلقة موثقة من '
            'corpus التشكيل (det-v1.3: 22 كلمة عالية التردد + هو)')
LABEL_EN = ('Partial fix: numbers, punctuation, closed lists, endings '
            'only — internal vowels remain from catt_eo except a '
            'documented closed set from the tashkeel corpus (det-v1.3: '
            '22 high-frequency words + hua)')

# استثناءات كسرة ال التعريف (منقول حرفيًا من ph3_engine_v53.mjs + استثناء
# الدليل §د «آلآن إن ورد» + حارس لام المحمولة: ال التعريف الحقيقية لامها
# ساكنة/عارية/مشددة — أما لام عليها فتحة/ضمة/كسرة فجذر (مثل اَلَمْ) لا يُلمس)
AL_EXCLUDE = {'ال', 'الف', 'الفين', 'الله', 'الآن'}
WAL_EXCLUDE = {'والله', 'والد', 'والدة', 'والدي', 'والدك', 'والدنا'}
FAL_EXCLUDE = {'فالح', 'فالحة', 'فالحين'}
BAL_EXCLUDE = {'بال', 'بالي', 'بالك', 'باله', 'بالنا', 'بالم',
               'بالمي', 'بالكم', 'بالهم'}
# أسماء موصولة لامها هي المُشدَّدة (اَلَّذِي لا اَلذِّي) — تُستثنى من
# قاعدة الشمسية منعًا لإفسادها
RELATIVE_PRONOUNS = {'الذي', 'التي', 'الذين', 'اللتان', 'اللاتي',
                     'اللذان', 'اللائي'}

STRIP_D = re.compile('[' + ''.join(DIACS) + ']')


def strip_d(s):
    return STRIP_D.sub('', s)


def fold(s):
    return s.translate(ALEF_FOLD)


def fskel(w):
    """هيكل مطوي الهمزات — مفتاح البحث في كل الخرائط."""
    return fold(strip_d(w))


def _compat(a, b):
    """توافق حرفين للمحاذاة (طي همزة / ة↔ه / أرقام بالقيمة)."""
    if a == b:
        return True
    fa, fb = fold(a), fold(b)
    if fa == fb and (a != TATWEEL or b != TATWEEL):
        return True
    if {a, b} == {'\u0629', '\u0647'}:          # ة ↔ ه
        return True
    if DIGIT_AR.get(a) == b or DIGIT_LA.get(a) == b:
        return True
    return False


# ------------------------------------------------------------ تفكيك كلمة --

class Word:
    """كلمة = بادئة غير حرفية + وحدات (حرف، حركاته) + لاحقة غير حرفية."""

    __slots__ = ('prefix', 'units', 'suffix', 'raw')

    def __init__(self, prefix, units, suffix):
        self.prefix = prefix
        self.units = units            # [(letter, diacs_str)]
        self.suffix = suffix

    @classmethod
    def parse(cls, w):
        i = 0
        while i < len(w) and not AR_LETTER(w[i]):
            i += 1
        j = len(w)
        # الحركات الطرفية جزء من الكلمة (سكون/تنوين النهاية) — لا تُقصّ؛
        # اللاحقة المقطوعة ترقيم فقط (لا حركة ولا حرف)
        while j > i and not AR_LETTER(w[j - 1]) and w[j - 1] not in DIACS:
            j -= 1
        core, prefix, suffix = w[i:j], w[:i], w[j:]
        units, cur, letter_seen = [], None, False
        for c in core:
            if AR_LETTER(c):
                if cur is not None:
                    units.append(cur)
                cur = [c, '']
                letter_seen = True
            elif c in DIACS:
                if not letter_seen:          # حركة قبل أول حرف → بادئة
                    prefix += c
                else:
                    cur[1] += c
            else:                            # حرف غريب داخل النواة
                return None
        if cur is not None:
            units.append(cur)
        return cls(prefix, [(l, d) for l, d in units], suffix)

    def letters(self):
        return [l for l, _ in self.units]

    def skeleton(self):
        return ''.join(self.letters())

    def key(self):
        """مفتاح البحث المطوي (مع تجريد سابقة و/ف)."""
        k = fold(self.skeleton())
        if k and k[0] in 'وف' and len(k) > 2:
            return k[1:], True
        return k, False

    def render(self):
        out = [self.prefix]
        for l, d in self.units:
            out.append(l + d)
        out.append(self.suffix)
        return ''.join(out)

    def plant(self, target_units):
        """زرع حركات الشكل الهدف على حروف هذه الكلمة (الخام)."""
        if len(target_units) != len(self.units):
            return False
        for (rl, _), (tl, td) in zip(self.units, target_units):
            if not _compat(rl, tl):
                return False
        self.units = [(rl, td) for (rl, _), (_, td)
                       in zip(self.units, target_units)]
        return True


def _parse_target(form):
    """تفكيك شكل مجمد (من القوائم) إلى وحدات (حرف، حركات)."""
    w = Word.parse(form)
    return w.units if w else None


# ------------------------------------------------------------ الوحدة --

class DetTashkeel:
    """الطبقة الحتمية — انظر توثيق الوحدة (تحسين جزئي، ليس بديلًا كاملًا)."""

    def __init__(self, data_path=None):
        if data_path is None:
            data_path = os.path.join(os.path.dirname(os.path.abspath(
                __file__)), 'det_data', 'det_lists.json')
        with open(data_path, encoding='utf-8') as f:
            doc = json.load(f)
        self.version = doc['version']
        self.closed_set = set(doc['closed_set'])
        self.tanween_map = doc['tanween_map']
        self.common_map = doc['common_map']
        self.tier_c_map = doc['tier_c_map']
        self.frozen_phrases = doc['frozen_phrases']
        self.tier_a = doc['tier_a']
        self.rabbena = doc['rabbena']
        # عبارات مرتبة بالأطول أولًا (أولوية المطابقة)
        self.phrases = sorted(
            [{'pat': p['pat'], 'rep': p['rep'], 'src': src}
             for src, lst in (('tier_a', self.tier_a),
                              ('rabbena', self.rabbena),
                              ('frozen', self.frozen_phrases))
             for p in lst],
            key=lambda p: -len(p['pat']))
        self._tgt_cache = {}

    # ---------- أدوات ----------

    def _target(self, form):
        if form not in self._tgt_cache:
            self._tgt_cache[form] = _parse_target(form)
        return self._tgt_cache[form]

    @staticmethod
    def _convert_punct(t):
        return t.replace('\u060C', ',').replace('\u061F', '?')

    # ---------- المحاذاة حرفيًا (raw ↔ catt) ----------

    def _align(self, cw, rw, fixes, flags):
        """يزرع حركات catt على حروف raw (مع نزع ألف الإعراب الزائدة)."""
        c_units, r_letters = cw.units, rw.letters()
        if len(c_units) == len(r_letters):
            if all(_compat(c[0], r) for c, r in zip(c_units, r_letters)):
                rw.units = [(r, d) for r, (_, d) in
                            zip(r_letters, c_units)]
                if any(c[0] != r for (c, _), r in zip(c_units, r_letters)):
                    fixes['letter_restore'] += 1
                return True
            return False
        # ألف إعراب زائدة: ناجح + ً + ا (والخام بلا ألف)
        if (len(c_units) == len(r_letters) + 1
                and c_units[-1][0] == '\u0627'
                and all(_compat(c[0], r)
                        for c, r in zip(c_units[:-1], r_letters))):
            new_units = [(r, d) for r, (_, d) in
                         zip(r_letters, c_units[:-1])]
            rw.units = new_units
            fixes['i3rab_alef_strip'] += 1
            return True
        return False

    # ---------- التنوين ----------

    def _tanween_sweep(self, w, fixes):
        """يحذف/يستبدل التنوين خارج القائمة المغلقة أو المحظور منه،
        ويرتب فتح القائمة المغلقة بعد الألف (§ب)."""
        hit = [(i, d) for i, (l, ds) in enumerate(w.units)
               for d in ds if d in TANWEEN]
        if not hit:
            return
        kf = fold(w.skeleton())
        ks = (kf[1:] if kf and kf[0] in 'وف' and len(kf) > 2 else None)
        in_closed = kf in self.closed_set or (ks and ks in self.closed_set)
        n = len(w.units)
        for i, d in hit:
            l, ds = w.units[i]
            if in_closed and d == FATHATAN:
                continue                      # يُعالج أدناه
            # خارج القائمة أو محظور (ضم/كسر) ← الحركة القصيرة المكافئة
            ds = ''.join(x for x in ds if x not in TANWEEN)
            if not (i == n - 1 and (l in LONG_VOWELS or l == '\u0629')):
                ds += {FATHATAN: FATHA, DAMMATAN: DAMMA,
                       KASRATAN: KASRA}[d]
            w.units[i] = (l, ds)
            fixes['tanween_banned' if in_closed else 'tanween_deleted'] += 1
        if not in_closed:
            return
        fpos = [i for i, d in hit if d == FATHATAN]
        if not fpos:
            return
        i = fpos[0]
        l, ds = w.units[i]
        if i == n - 1:                           # على الحرف الأخير (اً)
            fixes['tanween_kept'] += 1
        elif i == n - 2 and w.units[n - 1][0] == '\u0627':
            # نقل التنوين إلى ما بعد الألف: جِدًا → جِدَاً
            w.units[i] = (l, ds.replace(FATHATAN, FATHA))
            la, lda = w.units[n - 1]
            w.units[n - 1] = (la, lda + FATHATAN)
            fixes['tanween_reordered'] += 1
        else:                                    # موضع شاذ ← فتحة فقط
            w.units[i] = (l, ds.replace(FATHATAN, FATHA))
            fixes['tanween_deleted'] += 1

    # ---------- نهايات §ج ----------

    def _finals(self, w, fixes):
        n = len(w.units)
        if n == 0:
            return
        l, ds = w.units[-1]
        if any(d in TANWEEN for d in ds):       # تنوين محفوظ (§ب) — لا يُمس
            return
        if l == '\u0629':                       # ة — بلا حركة نهائية
            if ds:
                w.units[-1] = (l, '')
                fixes['taa_stripped'] += 1
            return
        if l in LONG_VOWELS:
            # ا/ى/آ: بلا حركة نهائية إطلاقًا (§ج-3)
            if l in '\u0627\u0649\u0622':
                if ds:
                    w.units[-1] = (l, '')
                    fixes['final_stripped'] += 1
                return
            # و/ي: تُنزع حركات الإعراب فقط؛ السكون المرسوم يُحفظ
            # (أَيْ/دِيْ/فِيْ بأسلوب المخرجات المعتمدة)
            rest = ''.join(x for x in ds if x not in VOWELS)
            if rest != ds:
                w.units[-1] = (l, rest)
                fixes['final_stripped'] += 1
            return
        if SHADDA in ds:                        # شدة طرفية → بلا حركة
            rest = ''.join(x for x in ds
                           if x not in VOWELS and x not in TANWEEN)
            if rest != ds:
                w.units[-1] = (l, rest)
                fixes['final_stripped'] += 1
            return
        if any(x in VOWELS for x in ds):        # إعراب → سكون
            w.units[-1] = (l, SUKUN)
            fixes['final_sukun'] += 1
            return
        if SUKUN in ds:
            return
        w.units[-1] = (l, SUKUN)                # عارية → سكون مرسوم
        fixes['final_sukun_added'] += 1

    # ---------- هو / هي (§ج-5) ----------

    def _hua_hiya(self, w, next_w, fixes):
        k = fold(w.skeleton())
        if k == 'و':                            # §هـ: واو العطف = وِ دائمًا
            w.units = [('و', KASRA)]
            fixes['wa_prefix'] += 1
            return True
        if k == 'هي':
            w.units = [('ه', KASRA), ('ي', FATHA)]
            fixes['hiya'] += 1
            return True
        if k == 'هو':
            # هُوَ دائمًا — دليل corpus التشكيل (معيار المشروع الذي
            # درّب النموذج عليه): 2,286/2,336 موضعًا هُوَ (97.9%) مقابل
            # 25 هُوْ فقط (1.1% — ضجيج محرك). قاعدة §ج-5 القديمة كانت
            # تُخرج هُوْ الافتراضية وتخالف ممارسة corpus نفسها (انحدار
            # موثق في قياس det_eval50). فرع الإدغام الشرطي أُزيل: لم
            # يسنده أي توزيع فعلي في corpus.
            w.units = [('ه', DAMMA), ('و', FATHA)]
            fixes['hua'] += 1
            return True
        return False

    # ---------- الخرائط الأحادية (زرع مباشر أو بسلب سابقة) ----------

    def _plant_word(self, w, fixes):
        """يجرّب tier_c ← tanween ← common (بالأولوية). سلب سابقة و/ف/ب
        مسموح لـ tier_c/tanween (§ب «تجريد و/ف» + §ك و/ف+ال)؛ common
        بمفتاح كامل فقط (حماية من جذور مثل ولم)."""
        kf = fold(w.skeleton())
        has_pfx = bool(kf) and kf[0] in 'وفب' and len(kf) > 3
        kb = kf[1:] if has_pfx else None
        for mp, tag, allow_pfx in (
                (self.tier_c_map, 'tier_c', True),
                (self.tanween_map, 'tanween', True),
                (self.common_map, 'common', False)):
            hit = None
            for k in ((kf, kb) if allow_pfx else (kf,)):
                if k and k in mp:
                    hit = k
                    break
            if hit is None:
                continue
            tgt = self._target(mp[hit])
            if tgt is None:
                return False
            if len(tgt) == len(w.units) and w.plant(tgt):
                fixes['planted_' + tag] += 1
                return True
            if allow_pfx and has_pfx and len(tgt) == len(w.units) - 1:
                rest = w.units[1:]
                if all(_compat(u[0], t[0]) for u, t in zip(rest, tgt)):
                    pv = FATHA if w.units[0][0] == 'ف' else KASRA
                    w.units = ([(w.units[0][0], pv)] +
                               [(u[0], t[1]) for u, t in zip(rest, tgt)])
                    fixes['planted_' + tag] += 1
                    fixes['planted_pfx'] += 1
                    return True
            return False
        return False

    # ---------- ال التعريف (§د) + الشمسية ----------

    def _al_fix(self, w, fixes):
        n = len(w.units)
        if n < 3:
            return
        # أسماء موصولة لامها المُشدّدة والفها مفتوحة (اَلَّذِي لا اِلذِي) —
        # محمية بالكامل من كسرة ال ومن الشمسية معًا
        if fold(w.skeleton()) in RELATIVE_PRONOUNS:
            return
        # النمط: [و/ف/ب]? ا ل — بلا حركة على لام (حارس الجذور مثل اَلَمْ)
        i = 0
        if w.units[0][0] in 'وفب' and n >= 4:
            i = 1
        if w.units[i][0] != '\u0627' or w.units[i + 1][0] != 'ل':
            return
        pfx = w.units[0][0] if i == 1 else ''
        alef_d, lam_d = w.units[i][1], w.units[i + 1][1]
        if any(x in VOWELS for x in lam_d):
            return                              # لام محمولة → جذر لا تعريف
        if i == 1 and any(x in VOWELS and x != FATHA
                          for x in w.units[0][1]):
            return                              # وِال الصحيحة تُترك
        if any(x in VOWELS and x != FATHA for x in alef_d):
            return                              # اِل موجودة أصلًا
        skel = w.skeleton()
        excl = ({'و': WAL_EXCLUDE, 'ف': FAL_EXCLUDE,
                 'ب': BAL_EXCLUDE}[pfx] if pfx else AL_EXCLUDE)
        if skel not in excl:
            if i == 1:                          # وال/بال/فال → وِال/بِال/فَال
                w.units[0] = (pfx, FATHA if pfx == 'ف' else KASRA)
                w.units[1] = ('\u0627', '')
            else:                               # ال/اَل → اِل
                w.units[0] = ('\u0627', KASRA)
            fixes['al_kasra'] += 1
        # لام قمرية عارية → سكون مرسوم (الشمسية تبقى عارية — مُدمجة)
        if (not lam_d and n >= i + 2
                and w.units[i + 2][0] not in SUN_LETTERS):
            w.units[i + 1] = ('\u0644', SUKUN)
            fixes['qamari_lam_sukun'] += 1
        # شدة الشمسية الغائبة (بعد لام ساكنة/عارية)
        if fold(skel) not in RELATIVE_PRONOUNS and n >= i + 3:
            l2, d2 = w.units[i + 2]
            if (l2 in SUN_LETTERS and SHADDA not in d2
                    and SHADDA not in lam_d):
                base = ''.join(x for x in d2 if x not in VOWELS)
                vowels = ''.join(x for x in d2 if x in VOWELS)
                w.units[i + 2] = (l2, base + SHADDA + vowels)
                fixes['sun_shadda'] += 1

    # ---------- المعالجة الرئيسية ----------

    def process(self, raw, catt_out, safety_fn=None):
        fixes = {k: 0 for k in (
            'letter_restore', 'i3rab_alef_strip', 'tanween_deleted',
            'tanween_kept', 'tanween_reordered', 'tanween_banned',
            'taa_stripped', 'final_stripped', 'final_sukun',
            'final_sukun_added', 'al_kasra', 'sun_shadda', 'hua', 'hiya',
            'planted_tanween', 'planted_common', 'planted_tier_c',
            'planted_tier_a', 'planted_rabbena', 'planted_frozen',
            'planted_pfx', 'qamari_lam_sukun', 'wa_prefix',
            'skeleton_mismatch', 'safety_reverted')}
        flags = []
        raw_c, catt_c = self._convert_punct(raw), self._convert_punct(catt_out)
        raw_ws = raw_c.split()
        catt_ws = catt_c.split()
        if len(raw_ws) != len(catt_ws):
            # تعذر المحاذاة الكلمية — إخراج catt كما هو (ترقيم فقط)
            flags.append(f'word_count_mismatch: raw={len(raw_ws)} '
                         f'catt={len(catt_ws)}')
            return {'out': catt_c, 'label_ar': LABEL_AR,
                    'label_en': LABEL_EN, 'fixes': fixes, 'flags': flags,
                    'checks': {'skeleton_ok': False, 'density': None,
                               'n_words': len(catt_ws)},
                    'version': self.version}
        words, planted = [], []
        for rw_s, cw_s in zip(raw_ws, catt_ws):
            rw, cw = Word.parse(rw_s), Word.parse(cw_s)
            if rw is None or cw is None or not rw.units or not cw.units:
                words.append(cw_s)              # كلمة غير قابلة للتفكيك
                planted.append(True)            # تُترك كما هي
                flags.append(f'unparseable: {rw_s!r}')
                continue
            if not self._align(cw, rw, fixes, flags):
                words.append(cw)                # تعذر محاذاة → شكل catt
                planted.append(True)
                fixes['skeleton_mismatch'] += 1
                flags.append(f'skeleton_mismatch: {rw_s!r} → {cw_s!r}')
                continue
            words.append(rw)
            planted.append(False)
        # --- 1) العبارات المجمدة (الأطول أولًا) ---
        i = 0
        while i < len(words):
            matched = False
            if isinstance(words[i], Word) and not planted[i]:
                for ph in self.phrases:
                    n = len(ph['pat'])
                    if i + n > len(words):
                        continue
                    span = words[i:i + n]
                    if any(not isinstance(x, Word) or planted[j]
                           for j, x in enumerate(span, start=i)):
                        continue
                    keys = [fold(x.skeleton()) for x in span]
                    if keys != ph['pat']:
                        continue
                    ok = True
                    for x, rep_w in zip(span, ph['rep']):
                        tgt = self._target(rep_w)
                        if tgt is None or not x.plant(tgt):
                            ok = False
                            break
                    if ok:
                        for j in range(i, i + n):
                            planted[j] = True
                        fixes['planted_' + ph['src']] += n
                        i += n
                        matched = True
                        break
            if not matched:
                i += 1
        # --- 2) الخرائط الأحادية (دينية ← تنوين ← شائعة) ---
        for idx, (w, pl) in enumerate(zip(words, planted)):
            if not isinstance(w, Word) or pl:
                continue
            if self._plant_word(w, fixes):
                planted[idx] = True
        # --- 3) هو/هي ثم التنوين ثم النهايات ---
        for idx, (w, pl) in enumerate(zip(words, planted)):
            if not isinstance(w, Word) or pl:
                continue
            nxt = None
            for j in range(idx + 1, len(words)):
                if isinstance(words[j], Word):
                    nxt = words[j]
                    break
            if self._hua_hiya(w, nxt, fixes):
                planted[idx] = True
                continue
            self._tanween_sweep(w, fixes)
            self._finals(w, fixes)
        # --- 4) ال التعريف (بعد الزرع؛ الأشكال المزروعة صحيحة أصلًا) ---
        for w, pl in zip(words, planted):
            if isinstance(w, Word) and not pl:
                self._al_fix(w, fixes)
        # --- 5) الأمان الصوتي: ارتجاع كلمات catt للمصطدمة ---
        if safety_fn is not None:
            for idx, w in enumerate(words):
                if not isinstance(w, Word):
                    continue
                hits = safety_fn(w.render())
                if hits:
                    words[idx] = catt_ws[idx]
                    planted[idx] = True
                    fixes['safety_reverted'] += 1
                    flags.append(f'safety: {catt_ws[idx]!r} → {hits}')
        # --- 6) التركيب والفحوص ---
        out = ' '.join(x if isinstance(x, str) else x.render()
                       for x in words)
        skeleton_ok = strip_d(out) == strip_d(raw_c)
        if not skeleton_ok:
            flags.append('skeleton_check_failed')
        ar_words = [x for x in words if isinstance(x, Word)]
        n_d = sum(sum(1 for c in x.render() if c in DIACS)
                  for x in ar_words)
        density = (n_d / len(ar_words)) if ar_words else 0.0
        return {'out': out, 'label_ar': LABEL_AR, 'label_en': LABEL_EN,
                'fixes': fixes, 'flags': flags,
                'checks': {'skeleton_ok': skeleton_ok,
                           'density': round(density, 2),
                           'n_words': len(ar_words)},
                'version': self.version}


_DEFAULT = None


def get_default():
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = DetTashkeel()
    return _DEFAULT


def vocalize_partial(raw, catt_out, safety_fn=None):
    """واجهة مختصرة — تحسين جزئي (انظر LABEL_AR/LABEL_EN)."""
    return get_default().process(raw, catt_out, safety_fn=safety_fn)
