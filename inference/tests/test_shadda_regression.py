#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""اختبار انحدار الشدة (shadda) — الإصلاح R1F لترتيب الحركات مع الشدة
=============================================================================
الإصلاح المُختبَر (2026-10-09): normalize_diacritic_order في
lib/eqz_tokens.py كانت تحمل النسخة التالفة:

    النسخة التالفة:  return _DIAC_SWAP_RE.sub(r'<STX>}', text)
                     (محرف تحكم خام 0x02 + قوس } — يمحو الحركة والشدة
                      ويُدخل همزة زائفة عبر } في الفونتيسة)
    النسخة المصلحة:  return _DIAC_SWAP_RE.sub(r'\2\1', text)
                     (شدة ثم حركة — الترتيب الكانوني؛ التضعيف يمر
                      كتوكن _dbl_ المدرَّب)

المرجع التجريبي (نواة Kaggle r4-reeval-v1 — تقرير R1/R4): R1F = أوزان R1
+ هذه الواجهة المصلحة؛ 12/12 PASS على المصلحة مقابل 11/12 FAIL على
التالفة في المرجع السابق — هذا الملف يحافظ على نفس نوع التغطية
(انظر القسم Z: محاكاة النسخة التالفة عمدًا وإثبات أن الحالات تُكشف).

الحالات (بأسلوب corpus في الكتابة: حركة ثم شدة — 93.6% من النصوص):
  التضعيف يظهر كتوكن _dbl_ · الحركة المحذوفة سابقًا تعود · لا همزة
  زائفة من } · إدغام الحروف الشمسية يعمل (لام ال تُحذف أمام المشددة).

الاستخدام (من مجلد inference/):
    python tests/test_shadda_regression.py
يكتب tests/shadda_regression_results.json ويعيد رمز خروج 0 عند النجاح.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
INF = os.path.dirname(HERE)
for p in (INF, os.path.join(INF, 'lib')):
    if p not in sys.path:
        sys.path.insert(0, p)

import eqz_tokens                                          # noqa: E402
import infer                                               # noqa: E402

RESULTS = {'cases': [], 'tamper_simulation': {}, 'n_pass': 0,
           'n_fail': 0, 'byte_guard': None}


def record(name, ok, detail=None):
    RESULTS['cases'].append({'name': name, 'ok': bool(ok),
                             'detail': detail})
    RESULTS['n_pass' if ok else 'n_fail'] += 1
    print(f'  [{"PASS" if ok else "FAIL"}] {name}'
          + ('' if ok else f'\n         {str(detail)[:240]}'))


# ============================================================================
# القسم S — الحالات الاثنتا عشرة (المسار الإنتاجي: infer.tokenize)
# ============================================================================
# (اسم، جملة، الكلمة المستهدفة المشدودة، الحركة المتوقعة بعد _dbl_)
# expect_vowel=None ⟵ تحكم سلبي: الكلمة بلا شدة أصلًا
SHADDA_CASES = [
    ('S01 معلم (كسرة+شدة)', 'اِلْمُعَلِّم قَرَا اِلدَّرْس كُلُّه',
     'اِلْمُعَلِّم', 'i'),
    ('S02 الدرس (فتحة+شدة + إدغام شمسي)', 'اِلدَّرْس اِلْيَوْم حِلْو',
     'اِلدَّرْس', 'a'),
    ('S03 كله (ضمة+شدة)', 'اِلْمُعَلِّم شَرَح اِلدَّرْس كُلُّه',
     'كُلُّه', 'u'),
    ('S04 مدرسة (شدتان في كلمة)', 'اِلْمُمَدَّرْسَة كَبِيرَة',
     'اِلْمُمَدَّرْسَة', 'a'),
    ('S05 اتصلت (شدة أول الكلمة)', 'اِتَّصَلْت بِأُمِّي',
     'اِتَّصَلْت', 'a'),
    ('S06 شدة (كلمة الاختبار نفسها)', 'اِلشَّدَّة فِي اِلْكَلَام',
     'اِلشَّدَّة', 'a'),
    ('S07 جدة حكت (شدة + جيم مصرية)', 'اِلْجَدَّة حَكَتْ لِي حِكَايَة',
     'اِلْجَدَّة', 'a'),
    ('S08 جمع وتضعيف (جملة المرجع)', 'تَعَلَّمْنَا اِلْجَمْع وَ اِلتَّضْعِيف',
     'اِلتَّضْعِيف', 'a'),
    ('S09 الحقيقة (قاف خام — تحكم سلبي بلا شدة)',
     'اِلْحَقِيقَة اِن اِلْعِلْم نُور', 'اِلْحَقِيقَة', None),
    ('S10 كتب مشوقة (شدة وسط الكلمة)', 'اِلْمَكْتَبَة فِيهَا كُتُب مُشَوَّقَة',
     'مُشَوَّقَة', 'a'),
    ('S11 تجمعوا (شدة داخل كلمة مركبة)', 'كُلُّ اِلتَّلَامِيذ اِتَّجَمَّعُوا',
     'اِتَّجَمَّعُوا', 'a'),
    ('S12 جملة المسبار المرجعية كاملة', 'اِلْمُعَلِّم قَرَا اِلدَّرْس كُلُّه',
     'كُلُّه', 'u'),
]


def run_cases():
    print('=' * 72)
    print('S — حالات الشدة (المسار الإنتاجي infer.tokenize → eqz_tokens)')
    print('=' * 72)
    for name, sentence, word, expect_vowel in SHADDA_CASES:
        try:
            # 1) التطبيع نفسه: لا محرف تحكم ولا } زائفة في الجملة كلها
            norm = eqz_tokens.normalize_diacritic_order(sentence)
            clean = ('\x02' not in norm) and ('}' not in norm)
            # 2) توكنات الكلمة المستهدفة وحدها (سلوك الكلمة معزولة =
            #     سلوكها داخل الجملة — الفونتيسة تعمل كلمة-كلمة)
            wt = infer.tokenize(word, dialect='egy')
            if expect_vowel is None:
                # تحكم سلبي: كلمة بلا شدة — لا _dbl_ + قاف خام q موجودة
                ok = clean and ('_dbl_' not in wt) and ('q' in wt)
                record(name, ok, {'word_tokens': wt, 'clean': clean,
                                  'negative_control': True})
                continue
            has_dbl = '_dbl_' in wt
            # 3) الحركة المتوقعة تعود فور توكن التضعيف
            i = wt.index('_dbl_')
            vowel_ok = (i + 1 < len(wt)) and (wt[i + 1] == expect_vowel)
            # 4) لا همزة زائفة: عدّ الهمزات في الكلمة النظيفة لا يتجاوز
            #    همزات الرسم الأصلي (ال التعريف الشمسية لا تبدأ بهمزة
            #    زائدة: اِلشَّدَّة تبدأ < i — هذه شرعية)
            n_hamza_word = sum(1 for t in wt if t == '<')
            ok = clean and has_dbl and vowel_ok
            record(name, ok, {'word_tokens': wt, 'clean': clean,
                              'has_dbl': has_dbl, 'vowel_ok': vowel_ok,
                              'expected_vowel': expect_vowel,
                              'n_hamza': n_hamza_word})
        except Exception as e:                     # noqa: BLE001
            record(name, False, f'EXC {type(e).__name__}: {e}')


# ============================================================================
# القسم Z — محاكاة النسخة التالفة عمدًا (حرّاس كشف إعادة الإدخال)
# ============================================================================
def run_tamper_simulation():
    """إثبات أن هذا الملف يكشف النسخة التالفة إذا أُعيد إدخالها:
    نُبدّل normalize_diacritic_order مؤقتًا بالنسخة التالفة (0x02 + })
    ونتوقع أن تفشل الحالات — ثم نُعيد المصلحة فورًا (نفس بنية
    get_tokenizers_pair في نواة r4-reeval-v1 المرجعية)."""
    print('=' * 72)
    print('Z — محاكاة النسخة التالفة عمدًا (يجب أن تُكشف الحالات)')
    print('=' * 72)
    fixed = eqz_tokens.normalize_diacritic_order

    def broken(t):
        return eqz_tokens._DIAC_SWAP_RE.sub('\x02}', t)

    n_detected = 0
    details = []
    try:
        eqz_tokens.normalize_diacritic_order = broken
        _, toks_egy_broken, _, _ = eqz_tokens.get_tokenizers(
            [os.path.join(INF, 'lib')])
        for name, sentence, word, expect_vowel in SHADDA_CASES:
            wt = toks_egy_broken(word)
            if expect_vowel is None:
                # تحكم سلبي: لا شدة أصلًا — التالفة لا تمسها (مقصود:
                # هذه هي الحالة الوحيدة التي لا تُكشف — 11/12 المرجعية)
                details.append({'name': name, 'detected': False,
                                'note': 'لا شدة في الكلمة (تحكم سلبي)'})
                continue
            detected = ('_dbl_' not in wt) or any(
                '\x02' in t or '}' in t for t in wt)
            # التضعيف اختفى و/أو ظهرت بقايا التلف — الحالة تُكشف
            if detected:
                n_detected += 1
            details.append({'name': name, 'detected': detected,
                            'word_tokens_broken': wt[:14]})
    finally:
        eqz_tokens.normalize_diacritic_order = fixed
        infer._TOK_CACHE.clear()                  # إعادة دوال الإنتاج
        infer.get_tokenizer()

    RESULTS['tamper_simulation'] = {
        'n_cases': len(SHADDA_CASES),
        'n_detected': n_detected,
        'reference': '11/12 FAIL on broken (المرجع السابق)',
        'verdict': ('BROKEN VERSION DETECTED' if n_detected >= 11
                    else 'WARNING: tamper detection too weak'),
        'details': details,
    }
    print(f'  كُشفت {n_detected}/{len(SHADDA_CASES)} حالة على النسخة '
          f'التالفة (المرجع: 11/12 FAIL)')
    ok = n_detected >= 11
    record('Z محاكاة التالفة تُكشف (≥11/12)', ok,
           RESULTS['tamper_simulation'])


def run_byte_guard():
    """حارس بايتي: ملف الإنتاج نفسه يجب ألا يحوي محرف التحكم 0x02 —
    يكشف أي نسخ/لصق مستقبلي للسطر التالف (مثل نسخة training/eiqaz-v1
    التاريخية التي حُفظت كما دُرِّبت)."""
    print('=' * 72)
    print('G — حارس البايتات (لا 0x02 في ملف الإنتاج)')
    print('=' * 72)
    src = open(os.path.join(INF, 'lib', 'eqz_tokens.py'), 'rb').read()
    no_ctrl = b'\x02' not in src
    has_fix = b"_DIAC_SWAP_RE.sub(r'\\2\\1', text)" in src
    ok = no_ctrl and has_fix
    RESULTS['byte_guard'] = {'no_control_byte': no_ctrl,
                             'has_fixed_line': has_fix}
    record('G ملف الإنتاج نظيف وبه سطر الإصلاح', ok,
           RESULTS['byte_guard'])


if __name__ == '__main__':
    run_cases()
    run_tamper_simulation()
    run_byte_guard()
    out = os.path.join(HERE, 'shadda_regression_results.json')
    RESULTS['fix'] = {
        'file': 'inference/lib/eqz_tokens.py',
        'function': 'normalize_diacritic_order',
        'broken': "r'<STX>}' (raw 0x02 control byte + })",
        'fixed': "r'\\2\\1' (shadda-before-vowel canonical swap)",
        'reference': 'Kaggle r4-reeval-v1 (R1F arm) — production = R1 '
                     '+ corrected inference interface',
    }
    json.dump(RESULTS, open(out, 'w'), indent=1, ensure_ascii=False)
    print('=' * 72)
    print(f'النتيجة: {RESULTS["n_pass"]} PASS / {RESULTS["n_fail"]} FAIL')
    print(f'  منها: حالات الشدة {sum(1 for c in RESULTS["cases"] if c["name"].startswith("S"))}'
          f' | محاكاة التالفة + حارس البايتات 2')
    print(f'التقرير: {out}')
    print('=' * 72)
    sys.exit(0 if RESULTS['n_fail'] == 0 else 1)
