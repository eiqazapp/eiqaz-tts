#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""اختبارات سريعة مستقلة لمنطق auto — لا تحمل checkpoint ولا تشغّل TTS.

تغطي فئات القرار (بلا محرك catt): التمييز بين العاري/الجزئي/المكتمل،
الأوضاع الصريحة، النصوص القصيرة/المختلطة/المرقمة/الفارغة، فصل الكثافة
عن التغطية، بوابة الكلمات، والثبات عند التكرار.
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
INF = os.path.dirname(HERE)
for path in (INF, os.path.join(INF, 'lib')):
    if path not in sys.path:
        sys.path.insert(0, path)

import infer  # noqa: E402


class AutoDecisionTests(unittest.TestCase):
    """فئات 1-13 من مواصفة الاختبار (مستوى القرار)."""

    # 1) غير مشكول + egy
    def test_unvocalized_egyptian_defaults_to_egyptian(self):
        self.assertEqual(
            infer.effective_diacritize_mode('هو بيكتب الدرس', 'auto', 'egy'),
            'egyptian')

    # 2) غير مشكول + msa
    def test_unvocalized_msa_defaults_to_fusha(self):
        self.assertEqual(
            infer.effective_diacritize_mode('هذا درس مفيد', 'auto', 'msa'),
            'fusha')

    # 3/5) جزئي فوق عتبة الكثافة القديمة — ليس manual
    def test_partial_text_above_old_density_threshold_is_not_manual(self):
        text = 'بِيَكْتِب الدرس'
        density, n_marks = infer.diacritic_density(text)
        self.assertGreaterEqual(density, 0.30, (density, n_marks))
        coverage, marked, total = infer.diacritic_coverage(text)
        self.assertLess(coverage, 0.72, (coverage, marked, total))
        self.assertEqual(
            infer.effective_diacritize_mode(text, 'auto', 'egy'), 'egyptian')

    # 4) كلمة مشكولة وسط كلمات غير مشكولة
    def test_text_with_marks_in_only_one_word_is_not_manual(self):
        self.assertEqual(
            infer.effective_diacritize_mode('هُوَ بيكتب الدرس', 'auto', 'egy'),
            'egyptian')

    # 6) نص مكتمل
    def test_fully_vocalized_text_is_preserved_in_auto(self):
        text = 'بِيِكْتِبُ الدَّرْسُ'
        self.assertTrue(infer._looks_fully_diacritized(text))
        self.assertEqual(
            infer.effective_diacritize_mode(text, 'auto', 'egy'), 'manual')

    # 7) نص مختلط عربي/إنجليزي
    def test_mixed_arabic_latin_goes_to_vocalizer(self):
        self.assertEqual(
            infer.effective_diacritize_mode('الدرس مهم meeting', 'auto',
                                            'egy'),
            'egyptian')

    # 8) أرقام وترقيم
    def test_numbers_and_punctuation_go_to_vocalizer(self):
        self.assertEqual(
            infer.effective_diacritize_mode('عندي 5 كتب. صح؟', 'auto',
                                            'egy'),
            'egyptian')

    # 9) نص قصير جدًا
    def test_single_unvocalized_word_is_vocalized(self):
        self.assertEqual(
            infer.effective_diacritize_mode('بيكتب', 'auto', 'egy'),
            'egyptian')

    def test_single_fully_marked_word_is_manual(self):
        self.assertEqual(
            infer.effective_diacritize_mode('بِيِكْتِبْ', 'auto', 'egy'),
            'manual')

    # 10) فارغ/مسافات/بلا عربي — رفض صريح في prepare_text_rich
    # (الأرقام لا تُرفض: التطبيع يحوّلها كلمات عربية منطوقة — سلوك مقصود)
    def test_empty_and_non_arabic_raise(self):
        for bad in ('', '   ', '!!!'):
            with self.assertRaises(SystemExit, msg=repr(bad)):
                infer.prepare_text_rich(bad, 'auto', 'egy')

    def test_digits_become_arabic_and_are_not_rejected(self):
        res = infer.prepare_text_rich('123', 'auto', 'egy')
        self.assertTrue(res['numbers'])
        self.assertTrue(infer._AR_LETTERS.search(res['text']))

    def test_whitespace_only_direct_mode_call_is_defined(self):
        # الاستدعاء المباشر بلا نص عربي: قرار محدد لا انهيار
        self.assertEqual(
            infer.effective_diacritize_mode('   ', 'auto', 'egy'),
            'egyptian')

    # 11) الأوضاع الصريحة — دلالتها لا تتغير
    def test_explicit_modes_keep_their_meaning(self):
        partial = 'بِيَكْتِب الدرس'
        self.assertEqual(
            infer.effective_diacritize_mode(partial, 'manual', 'egy'),
            'manual')
        self.assertEqual(
            infer.effective_diacritize_mode(partial, 'egyptian', 'egy'),
            'egyptian')
        self.assertEqual(
            infer.effective_diacritize_mode(partial, 'fusha', 'msa'),
            'fusha')

    # 12-أ) فصل الكثافة عن التغطية: شدة ترفع الكثافة ولا تغطي الحروف
    def test_shadda_density_does_not_fake_full_coverage(self):
        text = 'الشَّمْس الدرس'      # كثافة 0.30، تغطية 0.20
        density, _ = infer.diacritic_density(text)
        coverage, marked, total = infer.diacritic_coverage(text)
        self.assertGreaterEqual(density, 0.30)
        self.assertLess(coverage, 0.72)
        self.assertEqual(
            infer.effective_diacritize_mode(text, 'auto', 'egy'), 'egyptian')

    # 12-ب) بوابة الكلمات: تغطية حروف عالية مع كلمات عارية → ليست مكتملة
    def test_high_coverage_with_bare_words_is_not_manual(self):
        text = 'بِيِكْتُبُ بِيِكْتُبُ بِيِكْتُبُ بِيِكْتُبُ بِيِكْتُبُ ' \
               'الدرس هنا'
        coverage, _, _ = infer.diacritic_coverage(text)
        self.assertGreaterEqual(coverage, 0.72)      # بوابة الحروف تجتاز
        self.assertFalse(infer._looks_fully_diacritized(text))  # الكلمات لا
        self.assertEqual(
            infer.effective_diacritize_mode(text, 'auto', 'egy'), 'egyptian')

    # 13) ثبات القرار عند التكرار
    def test_decision_is_stable_across_calls(self):
        text = 'بِيَكْتِب الدرس مهم'
        results = {infer.effective_diacritize_mode(text, 'auto', 'egy')
                   for _ in range(3)}
        self.assertEqual(len(results), 1)
        self.assertEqual(results.pop(), 'egyptian')

    # تشكيل جزئي في أول/وسط/آخر الكلمة
    def test_partial_marks_at_any_position_are_not_manual(self):
        for text in ('بَيكتب الدرس', 'بيكْتب الدرس', 'بيكتَب الدرس'):
            self.assertEqual(
                infer.effective_diacritize_mode(text, 'auto', 'egy'),
                'egyptian', msg=text)


class MergeDecisionTests(unittest.TestCase):
    """قرارات merge_preserved_marks عند فشل محاذاة الكلمات (بلا catt)."""

    def test_word_count_mismatch_keeps_well_marked_original(self):
        original = 'أَنَا رَايِح أَعْمَلْ'          # تغطية ≥ 0.5
        vocalized = 'أَنَا رَايِح'                  # كلمة سقطت
        out = infer.merge_preserved_marks(original, vocalized)
        self.assertEqual(out, 'أَنَا رَايِح أَعْمَلْ')

    def test_word_count_mismatch_accepts_vocalizer_for_bare_text(self):
        original = 'بيكتب الدرس'                    # تغطية 0
        vocalized = 'مخرج مختلف تمامًا'
        out = infer.merge_preserved_marks(original, vocalized)
        self.assertEqual(out, vocalized)

    def test_no_original_marks_returns_vocalized_unchanged(self):
        out = infer.merge_preserved_marks('بيكتب الدرس', 'بِيِكْتِبْ اِلدَّرْسْ')
        self.assertEqual(out, 'بِيِكْتِبْ اِلدَّرْسْ')


class MergeFallbackFlagTests(unittest.TestCase):
    """تمييز إعادة النص الأصلي الناقص عند فشل المحاذاة (تدقيق 2026-10-10).

    المطلوب: النص شبه المشكول (تغطية >= 0.5) الذي تعذر محاذاة كلماته
    يُعاد كما هو — لكن بعلم صريح يميزه من النص المكتمل، لا كتمًا.
    """

    def test_kept_original_fallback_is_explicit(self):
        out, info = infer.merge_preserved_marks(
            'أَنَا رَايِح أَعْمَلْ', 'أَنَا رَايِح', return_info=True)
        self.assertEqual(out, 'أَنَا رَايِح أَعْمَلْ')
        self.assertEqual(info['fallback'], 'kept_original')
        self.assertIsNotNone(info['coverage'])
        self.assertGreaterEqual(info['coverage'], 0.5)

    def test_accepted_vocalizer_fallback_is_explicit(self):
        out, info = infer.merge_preserved_marks(
            'بيكتب الدرس', 'بِيِكْتِبْ', return_info=True)
        self.assertEqual(out, 'بِيِكْتِبْ')
        self.assertEqual(info['fallback'], 'accepted_vocalizer')
        self.assertLess(info['coverage'], 0.5)

    def test_normal_merge_has_no_fallback(self):
        out, info = infer.merge_preserved_marks(
            'بِيَكْتِب الدرس', 'بِيِكْتِبْ اِلدَّرْسْ', return_info=True)
        self.assertEqual(out, 'بِيَكْتِبْ اِلدَّرْسْ')
        self.assertIsNone(info['fallback'])
        self.assertEqual(info['preserved'], 4)

    def test_default_signature_unchanged(self):
        # توافق كامل مع الواجهة القائمة (بلا return_info يعيد النص فقط)
        out = infer.merge_preserved_marks(
            'أَنَا رَايِح أَعْمَلْ', 'أَنَا رَايِح')
        self.assertEqual(out, 'أَنَا رَايِح أَعْمَلْ')

    def test_pipeline_exposes_merge_fallback_key(self):
        # المسار الكامل: المفتاح موجود دائمًا، وFalse في الدمج الطبيعي
        res = infer.prepare_text_rich('بِيَكْتِب الدرس', 'auto', 'egy')
        self.assertIn('merge_fallback', res)
        self.assertIn('merge_info', res['stages'])
        self.assertFalse(res['merge_fallback'])
        self.assertIsNone(res['stages']['merge_info']['fallback'])

    def test_pipeline_manual_mode_has_no_fallback_flag(self):
        res = infer.prepare_text_rich('بِيَكْتُبُ الدَّرْسُ', 'auto', 'egy')
        self.assertFalse(res['merge_fallback'])
        self.assertEqual(res['diacritize'], 'manual')


if __name__ == '__main__':
    unittest.main(verbosity=2)
