#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""اختبارات انحدار الحركات الداخلية المصرية — عبر محرك catt_eo الفعلي.

تغطي الفئة 14 من مواصفة الاختبار: النطق المصري للكلمات المستهدفة
بالقائمة المغلقة الموثقة من corpus، وسلوك الدمج (حفظ حركات المستخدم)،
وسلامة العلامات (لا تكرار ولا تناقض)، وثبات المسار، ووصول الحركات
إلى التوكنات. لا يُحمَّل checkpoint الصوتي ولا يُولَّد WAV — catt_eo
مضمّن في lib/tts_arabic/data (onnxruntime فقط).

يُتخطى بأمان (skip) إذا غاب onnxruntime أو أوزان catt — القرارات
البرمجية مغطاة في test_diacritize_auto.py بلا محرك.
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
INF = os.path.dirname(HERE)
for path in (INF, os.path.join(INF, 'lib')):
    if path not in sys.path:
        sys.path.insert(0, path)

try:
    import onnxruntime          # noqa: F401
    _ORT = True
except ImportError:
    _ORT = False

import infer  # noqa: E402

CATT_OK = _ORT and os.path.exists(os.path.join(
    INF, 'lib', 'tts_arabic', 'data'))

VOWELS = set('\u064E\u064F\u0650')          # فتحة/ضمة/كسرة
TANWEEN = set('\u064B\u064C\u064D')
SUKUN = '\u0652'
SHADDA = '\u0651'
AR_LETTER = lambda c: '\u0621' <= c <= '\u064A' and c != '\u0640'


def letter_units(word):
    units, i = [], 0
    chars = list(word)
    while i < len(chars):
        ch = chars[i]
        if AR_LETTER(ch):
            marks = []
            j = i + 1
            while j < len(chars) and '\u064B' <= chars[j] <= '\u0652':
                marks.append(chars[j])
                j += 1
            units.append((ch, ''.join(marks)))
            i = j
        else:
            i += 1
    return units


def marks_are_clean(word):
    """لا تكرار ولا تناقض: حركة واحدة على الأكثر + شدة/سكون واحدة."""
    for _, marks in letter_units(word):
        if sum(m in VOWELS for m in marks) > 1:
            return False
        if sum(m in TANWEEN for m in marks) > 1:
            return False
        if marks.count(SUKUN) > 1:
            return False
        if marks.count(SHADDA) > 1:
            return False
        if marks.count(SUKUN) and sum(m in VOWELS for m in marks):
            return False          # سكون وحركة معًا على حرف واحد
    return True


def unit_marks(word, letter_index):
    return letter_units(word)[letter_index][1]


@unittest.skipUnless(CATT_OK, 'onnxruntime أو أوزان catt_eo غير متاحة')
class EgyptianInternalVowelsTests(unittest.TestCase):
    """الفئة 14: نطق الكلمات المستهدفة وفق معيار corpus الموثق."""

    MAP_WORDS = [
        # (كلمة خام, موضع الحرف الداخلي المفصلي, الحركة المتوقعة)
        # كل توقع = الشكل المهيمن المقيس في corpus (det-v1.3)
        ('بيقول', 1, '\u0650'),      # بِيِقُولْ — ياء بكسرة لا فتحة
        ('بيكتب', 1, '\u0650'),      # بِيِكْتِبْ
        ('دلوقتي', 0, '\u0650'),     # دِلْوَقْتِي
        ('بكرة', 0, '\u064F'),       # بُكْرَة
        ('هنا', 0, '\u0650'),        # هِنَا (لا هُنَا الفصيحة)
        ('يكون', 0, '\u0650'),       # يِكُونْ
        ('تكون', 0, '\u0650'),       # تِكُونْ
        ('قوي', 1, '\u0650'),        # قَوِي — الواو كسرة (واو ساكنة+ياء)
        ('طيب', 1, SHADDA),          # طَيِّبْ
        ('لسه', 1, SHADDA),          # لِسَّهْ
        ('كمان', 0, '\u064E'),       # كَمَانْ (لا كُمَانْ)
        ('نفس', 0, '\u064E'),        # نَفْسْ (لا نِفِسْ)
        ('كويس', 2, SHADDA),         # كُوَيِّسْ
        ('لازم', 2, '\u0650'),       # لَازِمْ (كسرة الزاي)
        ('ليه', 1, ''),              # لِيهْ (ياء عارية بلا فتحة)
        ('يلا', 1, SHADDA),          # يَلَّا
        ('رايح', 2, '\u0650'),       # رَايِحْ (ياء ساكنة→مد خطأ catt)
        ('اكتر', 0, '\u064E'),       # اَكْتَرْ (فتح الهمزة لا كسرها)
    ]

    def test_map_words_get_corpus_internal_vowels(self):
        for word, pos, want in self.MAP_WORDS:
            with self.subTest(word=word):
                res = infer.prepare_text_rich(word, 'egyptian', 'egy')
                out_word = res['text'].split()[0]
                units = letter_units(out_word)
                self.assertEqual(len(units), len(letter_units(word)),
                                 f'هيكل تغير: {word} → {out_word}')
                got = units[pos][1]
                self.assertIn(want, got,
                              f'{word}: حرف {pos} حركته {got!r} '
                              f'والمتوقع فيها {want!r}')

    def test_full_map_forms_match_skeleton_and_vowels(self):
        """الشكل الكامل (هيكل + كل الحركات الداخلية) لكل مدخل مع سابقة."""
        res = infer.prepare_text_rich('هو بيقول كده', 'egyptian', 'egy')
        self.assertIn('بِيِقُولْ', res['text'])
        self.assertTrue(res['text'].startswith('هُوَ'))   # قاعدة هو المصححة

    def test_hua_rule_follows_corpus(self):
        res = infer.prepare_text_rich('هو عارف المشكلة', 'egyptian', 'egy')
        self.assertTrue(res['text'].startswith('هُوَ'),
                        res['text'])


@unittest.skipUnless(CATT_OK, 'onnxruntime أو أوزان catt_eo غير متاحة')
class MergePreservationTests(unittest.TestCase):
    """المبدأ د: حفظ حركات المستخدم عند استكمال النص المشكول جزئيًا."""

    def test_task_example_keeps_user_marks(self):
        """مثال المهمة: بِيَكْتِب الدرس — حركات المستخدم لا تُمس."""
        res = infer.prepare_text_rich('بِيَكْتِب الدرس', 'auto', 'egy')
        self.assertEqual(res['diacritize'], 'egyptian')
        w = res['text'].split()[0]
        # فتحة الياء وسكون الكاف وكسرة التاء = إرادة المستخدم الأصلية
        self.assertEqual(unit_marks(w, 1), '\u064E')   # يَ
        self.assertEqual(unit_marks(w, 2), SUKUN)      # كْ
        self.assertEqual(unit_marks(w, 3), '\u0650')   # تِ

    def test_fully_marked_word_survives_letter_by_letter(self):
        res = infer.prepare_text_rich('دِلْوَقْتِي هنبدأ الدرس', 'auto', 'egy')
        self.assertEqual(res['text'].split()[0], 'دِلْوَقْتِي')

    def test_huwa_user_mark_wins(self):
        res = infer.prepare_text_rich('هُوَ بيكتب الدرس', 'auto', 'egy')
        self.assertEqual(res['text'].split()[0], 'هُوَ')

    def test_unmarked_word_in_partial_text_gets_corpus_form(self):
        res = infer.prepare_text_rich('هُوَ بيكتب الدرس', 'auto', 'egy')
        self.assertEqual(res['text'].split()[1], 'بِيِكْتِبْ')

    def test_fully_vocalized_text_keeps_every_user_mark(self):
        text = 'الْمُعَلِّم بِيِقُول لِلطَّالِب'
        res = infer.prepare_text_rich(text, 'auto', 'egy')
        out_words = res['text'].split()
        in_words = text.split()
        self.assertEqual(len(out_words), len(in_words))
        for orig, out in zip(in_words, out_words):
            for (ol, od), (vl, vd) in zip(letter_units(orig),
                                          letter_units(out)):
                if od:
                    self.assertEqual(od, vd,
                                     f'حركة مستخدم ضاعت: {orig} → {out}')

    def test_no_duplicate_or_conflicting_marks_anywhere(self):
        samples = [
            'بِيَكْتِب الدرس', 'دِلْوَقْتِي هنبدأ الدرس', 'هُوَ بيكتب',
            'هو بيقول كده كتير', 'لازم تكون هنا كمان', 'عندي 5 كتب، صح؟',
            'بكرة الساعة 10 هنبدأ التجربة.',
        ]
        for s in samples:
            res = infer.prepare_text_rich(s, 'auto', 'egy')
            for w in res['text'].split():
                self.assertTrue(marks_are_clean(w),
                                f'علامات فاسدة في {w!r} (من {s!r})')

    def test_pipeline_is_deterministic(self):
        s = 'دلوقتي هنبدأ الدرس يا عمر'
        r1 = infer.prepare_text_rich(s, 'auto', 'egy')
        r2 = infer.prepare_text_rich(s, 'auto', 'egy')
        self.assertEqual(r1['text'], r2['text'])
        self.assertEqual(r1['stages'].get('after_catt'),
                         r2['stages'].get('after_catt'))


@unittest.skipUnless(CATT_OK, 'onnxruntime أو أوزان catt_eo غير متاحة')
class ExplicitModeSemanticsTests(unittest.TestCase):
    """الأوضاع الصريحة عبر المسار الكامل — دلالاتها لم تتغير."""

    def test_manual_leaves_text_as_is(self):
        res = infer.prepare_text_rich('بِيَكْتُبُ الدَّرْسِ', 'manual', 'egy')
        self.assertEqual(res['text'], 'بِيَكْتُبُ الدَّرْسِ')
        self.assertFalse(res['did_vocalize'])

    def test_fusha_keeps_msa_article_not_egyptian_kasra(self):
        res = infer.prepare_text_rich('الدرس هيبدأ', 'fusha', 'egy')
        # الفصحى: ألف ال بلا كسرة (كسرة ال مصرية من قواعد det فقط)
        self.assertNotIn('ا' + '\u0650' + 'ل', res['text'])
        self.assertTrue(res['did_vocalize'])

    def test_egyptian_applies_article_kasra(self):
        res = infer.prepare_text_rich('الدرس هيبدأ', 'egyptian', 'egy')
        self.assertIn('ا' + '\u0650' + 'ل', res['text'])


@unittest.skipUnless(CATT_OK, 'onnxruntime أو أوزان catt_eo غير متاحة')
class NumbersPunctuationMixedTests(unittest.TestCase):
    """الأرقام والترقيم واللاتيني عبر المسار الكامل."""

    def test_numbers_become_spoken_arabic(self):
        res = infer.prepare_text_rich('عندي 5 كتب مهمة', 'auto', 'egy')
        self.assertIn('خَمْسَة', res['text'])
        self.assertTrue(res['numbers'])

    def test_arabic_punctuation_normalized_and_kept(self):
        res = infer.prepare_text_rich('ممتاز، صح؟', 'auto', 'egy')
        self.assertIn(',', res['text'])
        self.assertIn('?', res['text'])
        self.assertNotIn('\u060C', res['text'])      # ،
        self.assertNotIn('\u061F', res['text'])      # ؟

    def test_latin_transliterated_not_dropped(self):
        res = infer.prepare_text_rich('meeting في القاهرة', 'auto', 'egy')
        self.assertTrue(res['translit'])
        self.assertIn('ميتينج', res['normalized'])
        self.assertTrue(infer._AR_LETTERS.search(res['text']))


@unittest.skipUnless(CATT_OK, 'onnxruntime أو أوزان catt_eo غير متاحة')
class TokensReachTests(unittest.TestCase):
    """المرحلة 8: الحركات المختلفة تصل توكنات مختلفة (G2P الفعلي)."""

    def test_egyptian_vs_catt_vowels_give_different_tokens(self):
        _, toks_egy, _ = infer.get_tokenizer()
        pairs = [
            ('بِيِكْتِبْ', 'بِيَكْتُبْ'),
            ('هُوَ', 'هُوْ'),
            ('هِنَا', 'هُنَا'),
            ('يِكُونْ', 'يكُونْ'),
        ]
        for egy, msa_style in pairs:
            with self.subTest(pair=(egy, msa_style)):
                self.assertNotEqual(list(toks_egy(egy)),
                                    list(toks_egy(msa_style)))

    def test_prepared_text_tokenizes_fully(self):
        res = infer.prepare_text_rich('هو بيقول كده', 'auto', 'egy')
        toks = infer.tokenize(res['text'], 'egy')
        self.assertGreater(len(toks), 6)
        self.assertTrue(all(t for t in toks))


if __name__ == '__main__':
    unittest.main(verbosity=2)
