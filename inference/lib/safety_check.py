# -*- coding: utf-8 -*-
"""safety_check — فحص الأمان الصوتي لكلمات النص المشكَّل (نسخة الحزمة).

أصل الوحدة: tashkeel-pipeline/scripts/ph1_safety.py (القاعدة §ل-5 في دليل
التشكيل v2.1) — أُعيدت هندستها هنا لتعمل داخل حزمة الاستدلال المستقلة بلا
أي اعتماد خارجي: دوال الفحص نفسها، وقاموس البناء يُبنى بمحول G2P الخاص
بالحزمة نفسها (toks_egy من infer.py) بدل step0_counter — فتظل المقارنة
متسقة (نفس الدالة تبني القاموس وتفحص الكلمات).

الغرض: عند تفعيل «طبقة التصحيح الجزئي» (det_tashkeel)، كل كلمة مخرَجة
تُفحص لاصطدامها الصوتي (مستوى الهيكل أو مستوى تسلسل الفونيمات) بكلمات
غير لائقة — وأي اصطدام يُرجِّع الكلمة إلى مخرج catt الأصلي.

Tier 1 — كلمات بذيئة غير ملتبسة (القائمة في هذا الملف ولا تُطبع في
          التقارير). أي إصابة => ارتجاع الكلمة.
Tier 2 — متماثلات صوتية محرجة حسب السياق (طبية/قانونية قد تحرج خارج
          سياقها). الإصابة => ارتجاع الكلمة أيضًا في الحزمة (الخيار
          الأكثر أمانًا للمستخدم النهائي).
"""
import re

TASH = re.compile(r'[\u064B-\u0652\u0653-\u0655\u0670]')
PUNCT = '.,?!:;"()«»\u060C\u061F\u061B\u201C\u201D\u2026-'

# ---- Tier 1: unambiguous vulgar words (vocalized to fix pronunciation) ----
TIER1 = [
    'كَسّ', 'كُسّ', 'كَسْ', 'خَرَا', 'زَبّ', 'زُبْ', 'زَبْر', 'يَنِيكُ',
    'يَنِيكْ', 'نَاكَ', 'نِيكْ', 'قَحْبَة', 'شَرْمُوطَة', 'خُولْ',
    'مَتْنَاكْ', 'مِتْنَاكَة', 'عَرَّصْ', 'فَشْخْ', 'طَيْزَة', 'اِسْتَ',
    'اِسْتِ', 'إِسْتَ', 'قُوَّادْ', 'كَوَادَة',
]

# ---- Tier 2: context-dependent embarrassing homophones ----
TIER2 = [
    'فَرْجْ',      # legitimate: relief/opening — vulgar: female genital
    'دُبْرْ',      # legitimate: rear/Qur'anic (الليل إذا دبر) — vulgar: butt
    'عِيرْ',       # legitimate classical: caravan/whip — vulgar: penis
    'مَصَّ',       # legitimate verb: to suck (medicine) — crude in isolation
    'الشَّرَجْ',   # medical: rectum (anatomy) — embarrassing out of context
]


def make_safety_fn(toks_egy):
    """يبني دالة check_vocalized_word(vw) المستقلة، بقاموس مبنٍ من نفس
    toks_egy المُمرَّرة (اتساق البناء والفحص — نفس مبدأ ph1_safety)."""
    def phkey(word):
        """مفتاح تسلسل الفونيمات لكلمة (مشكولة) عبر G2P الحزمة."""
        try:
            toks = toks_egy(word)
        except Exception:
            return None
        return tuple(t for t in toks if t not in ('_+_', '_eos_'))

    lex1 = {w: phkey(w) for w in TIER1}
    lex2 = {w: phkey(w) for w in TIER2}
    lexall = {**lex1, **lex2}
    sk1 = {TASH.sub('', w) for w in TIER1}
    sk2 = {TASH.sub('', w) for w in TIER2}

    def check_vocalized_word(vw):
        """يعيد قائمة (tier, trigger-word, match-type) لأي إصابة في كلمة."""
        if not vw:
            return []
        s = TASH.sub('', vw).strip(PUNCT)
        hits = []
        if s in sk1:
            hits.append((1, s, 'spelling'))
        elif s in sk2:
            hits.append((2, s, 'spelling'))
        k = phkey(vw)
        if k:
            for lw, lk in lexall.items():
                if lk == k and TASH.sub('', lw) != s:
                    hits.append((1 if lw in lex1 else 2, TASH.sub('', lw),
                                 'phoneme'))
                    break
        return hits

    return check_vocalized_word
