# -*- coding: utf-8 -*-
"""بناء work/tier_a_map.json + work/ph1_letter_fixes.json (دليل v2.1 §ك).

تحقق ذاتي إلزامي لكل مدخل: نزع الحركات من الصيغة المجمدة == الخام حرفيًا
(أحرفًا ومسافات) — وإلا يُرفض البناء كليًا.
"""
import json
import re

STRIP = re.compile(r'[\u064B-\u0652\u0653-\u0655\u0670]')

TIER_A = [
    # (raw words, frozen form, corpus count, note)
    ('إن شاء الله', 'إِنْ شَاءَ اللهُ', 15, ''),
    ('ان شاء الله', 'اِنْ شَاءَ اللهُ', 4, 'ألف عارية بكسرة همزة وصل (§و)'),
    ('ما شاء الله', 'مَا شَاءَ اللهُ', 2, ''),
    ('بإذن الله', 'بِإِذْنِ اللهِ', 8, ''),
    ('والله', 'وَاللهِ', 11, 'قسم'),
    ('سبحان الله', 'سُبْحَانَ اللهِ', 4, ''),
    ('الحمد لله', 'اَلْحَمْدُ لِلَّهِ', 3, 'تجويد: فتحة الألف — يُفرض بعد إصلاح ال التعريف'),
    ('بسم الله', 'بِسْمِ اللهِ', 1, ''),
    ('لا حول ولا قوة إلا بالله', 'لَا حَوْلَ وَلَا قُوَّةَ إِلَّا بِاللهِ', 1,
     'الاستثناء الوحيد: ة بفتحة داخل صيغة مجمدة'),
    ('لله', 'لِلَّهِ', 5, ''),
    ('بالله', 'بِاللهِ', 2, ''),
    ('يا رب', 'يَا رَبِّ', 2, ''),
    ('الله ينور', 'اَللَّهُ يِنَوِّرْ', 5, 'فعل مصري داخل الصيغة'),
    ('الله يرحمها', 'اَللَّهُ يِرْحَمْهَا', 1, 'فعل مصري داخل الصيغة'),
    ('لا قدر الله', 'لَا قَدَرَ اللهُ', 13, 'معتمدة 2026-09-30'),
    ('لقدر الله', 'لَقَدَرَ اللهُ', 10, 'معتمدة 2026-09-30'),
    ('لو قدر الله', 'لَوْ قَدَرَ اللهُ', 1, 'معتمدة 2026-09-30'),
    ('جزاك الله خيرا', 'جَزَاكَ اللهُ خَيْرًا', 0, 'احتياطية — صفر في المتن؛ تنوين خيرا في القائمة البيضاء'),
]

RABBENA = [
    ('ربنا', 'رَبِّنَا', 'القاعدة العامة لكل ورود ربنا — §ك-4'),
    ('بربنا', 'بِرَبِّنَا', 'قسم/تأكيد عامي — §ك-4'),
]


def check(raw, out):
    a = STRIP.sub('', raw)
    b = STRIP.sub('', out)
    assert a == b, f'letter mismatch:\n  raw  = {raw!r}\n  out  = {out!r}\n  skel(raw)={a!r}\n  skel(out)={b!r}'
    # word counts must also align
    assert len(raw.split()) == len(out.split()), f'word count mismatch: {raw!r}'


def main():
    entries = []
    for raw, out, n, note in TIER_A:
        check(raw, out)
        entries.append({'raw': raw, 'out': out, 'n_corpus': n, 'note': note})
    rab = []
    for raw, out, note in RABBENA:
        check(raw, out)
        rab.append({'raw': raw, 'out': out, 'note': note})

    doc = {
        'version': 'v2.1 — 2026-09-30',
        'comment': ('خريطة Tier A + فرض رَبِّنَا (دليل v2.1 §ك). تُفرض آليًا بعد '
                    'التشكيل وإصلاح ال التعريف. المطابقة على هيكل الكلمات المجرد '
                    '(بلا حركات) مع حفظ اللواحق غير الحرفية (ترقيم/أقواس). كل '
                    'مدخل تحقق أن نزع الحركات منه == خامّه حرفيًا.'),
        'entries': entries,
        'rabbena': rab,
        'total_corpus_positions': sum(e['n_corpus'] for e in entries),
    }
    json.dump(doc, open('/home/z/my-project/work/tier_a_map.json', 'w',
                        encoding='utf-8'), ensure_ascii=False, indent=1)

    fixes = {
        'version': '2026-09-30',
        'comment': ('تصحيحا تفريغ معتمدان من المستخدم (دليل v2.1 §ك-5). المحرك '
                    'يبدّل نص الوحدة قبل التشكيل ويوثّق التبديل؛ المرجع الصرفي '
                    'للفحص = النص المصحَّح. الاستثناءان الوحيدان في الدليل.'),
        'fixes': [
            {
                'id': '1_00768_01',
                'original': 'بالزبط عالم كامل من الظواهر الاجتماعية اللي بتحصل جوه صلاة الحديد دي',
                'corrected': 'بالزبط عالم كامل من الظواهر الاجتماعية اللي بتحصل جوه صالة الحديد دي',
                'word_before': 'صلاة', 'word_after': 'صالة',
                'reason': 'خطأ تفريغ: المراد صالة الحديد (الجيم) — نطقها «صلاة» يقلب المعنى',
                'approved': '2026-09-30',
            },
            {
                'id': '1_08382_00',
                'original': 'محتوى بسيط بس بيحل مشكلة صغيرة للجمهور بتاعك. رب على تعليق بصدق و اهتمام.',
                'corrected': 'محتوى بسيط بس بيحل مشكلة صغيرة للجمهور بتاعك. رد على تعليق بصدق و اهتمام.',
                'word_before': 'رب', 'word_after': 'رد',
                'reason': 'خطأ تفريغ: المراد رد على تعليق',
                'approved': '2026-09-30',
            },
        ],
    }
    json.dump(fixes, open('/home/z/my-project/work/ph1_letter_fixes.json', 'w',
                          encoding='utf-8'), ensure_ascii=False, indent=1)

    print(f'tier_a_map: {len(entries)} entries, '
          f'{doc["total_corpus_positions"]} corpus positions '
          f'(expected 88 incl. 0-reserve)')
    print(f'rabbena: {len(rab)} entries')
    print('letter_fixes: 2 entries')
    print('ALL LETTER-FAITHFULNESS CHECKS PASSED')


if __name__ == '__main__':
    main()
