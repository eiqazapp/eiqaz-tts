#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""اختبارات سريعة مستقلة لمنطق auto — لا تحمل checkpoint ولا تشغّل TTS."""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
INF = os.path.dirname(os.path.dirname(HERE))
for path in (INF, os.path.join(INF, 'lib')):
    if path not in sys.path:
        sys.path.insert(0, path)

import infer  # noqa: E402


class AutoDiacritizeModeTests(unittest.TestCase):
    def test_unvocalized_egyptian_defaults_to_egyptian(self):
        self.assertEqual(
            infer.effective_diacritize_mode('هو بيكتب الدرس', 'auto', 'egy'),
            'egyptian')

    def test_unvocalized_msa_defaults_to_fusha(self):
        self.assertEqual(
            infer.effective_diacritize_mode('هذا درس مفيد', 'auto', 'msa'),
            'fusha')

    def test_partial_text_above_old_density_threshold_is_not_manual(self):
        text = 'بِيَكْتِب الدرس'
        density, n_marks = infer.diacritic_density(text)
        self.assertGreaterEqual(density, 0.30, (density, n_marks))
        coverage, marked, total = infer.diacritic_coverage(text)
        self.assertLess(coverage, 0.72, (coverage, marked, total))
        self.assertEqual(
            infer.effective_diacritize_mode(text, 'auto', 'egy'), 'egyptian')

    def test_fully_vocalized_text_is_preserved_in_auto(self):
        text = 'بِيِكْتِبُ الدَّرْسُ'
        self.assertTrue(infer._looks_fully_diacritized(text))
        self.assertEqual(
            infer.effective_diacritize_mode(text, 'auto', 'egy'), 'manual')

    def test_explicit_modes_keep_their_meaning(self):
        partial = 'بِيَكْتِب الدرس'
        self.assertEqual(infer.effective_diacritize_mode(partial, 'manual', 'egy'),
                         'manual')
        self.assertEqual(infer.effective_diacritize_mode(partial, 'egyptian', 'egy'),
                         'egyptian')
        self.assertEqual(infer.effective_diacritize_mode(partial, 'fusha', 'msa'),
                         'fusha')

    def test_text_with_marks_in_only_one_word_is_not_manual(self):
        self.assertEqual(
            infer.effective_diacritize_mode('هُوَ بيكتب الدرس', 'auto', 'egy'),
            'egyptian')


if __name__ == '__main__':
    unittest.main(verbosity=2)
