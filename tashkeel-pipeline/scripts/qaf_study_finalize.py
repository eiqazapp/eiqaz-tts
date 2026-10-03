# -*- coding: utf-8 -*-
"""qaf_study_finalize.py — تجميع معجم الأشكال المدروسة النهائي (PATCH 9)
==========================================================================
المدخلات: نتائج الحصاد (qaf_study_forms.json من qaf_study_build.py) + تحقق
catt الإضافي (قطع/يقسم/تقسيم/قياس...) + قرار التنقيح اليدوي.

قرارات التنقيح الموثقة:
1. تجريد بوادي الالتصاق (و/ف/ب/ل/ك/ال) من الأشكال المحصودة — الإنتاج يعيد
   إلحاقها عند الزرع (مع قاعدة الشمسية/الشدة).
2. حذف الملتبسات الدلالية (الهيكل نفسه لكلمات مختلفة):
   qsmt قسمت (حصادها قُسِمَتْ المبني للمجهول ≠ قصد «ماضي القسمة»)
   tqys تقيس (حصادها بِتَقِيُّسِ = تقييس كلمة أخرى)
   qym قيم (حصادها الْقَيِّمُ = القيِّم كلمة أخرى)
   qrb قرّب (حصادها قُرْبٌ اسم ≠ قرّب فعل)
3. استبعاد قوي qwy (homograph «أوي» العامية — قرار موثق في EXCLUDED).
4. الطبقات:
   verified = قطعة/قطع (تأكيد سمعي من المستخدم 2026-10-01)
   tier1    = أسماء غير ملتبسة بأفعال + شكل corpus مكسور → زرع في auto+qaf
   tier2    = ملتبسة بفعل (قطع/قسم/يقسم) أو عامية-غالبة (قصة/قوة/قيمة)
              أو بحثية → زرع في qaf + علامة {ق} فقط
5. قاعدة الجملة: الملتبسة بفعل لا تُزرع داخل جملة متعددة الكلمات إلا بعلامة
   {ق} صريحة (catt داخل الجملة يعطي القراءة الصحيحة سياقيًا: قَطَعَ فعلًا).
"""
import json
import os
import re

REPO = '/home/z/my-project/work/github_repo/eiqaz-tts'
HARVEST = os.path.join(REPO, 'validation', 'qaf_q_study_forms.json')
OUT = os.path.join(REPO, 'validation', 'qaf_q_study_forms.json')   # نفس الملف — نسخة نهائية

TASH = re.compile(r'[\u064B-\u0652]')
SUN_LETTERS = set('تثدذرزسشصضطظلن')

# (skel, word_ar, study_form_bare, tier, provenance)
FINAL = [
    # ---- verified (تأكيد المستخدم السمعي 2026-10-01) ----
    ('qTEp', 'قطعة', 'قِطْعَةً', 'verified',
     'corpus ×36 بالشكل نفسه داخل الجمل + تأكيد المستخدم'),
    ('qTE', 'قطع', 'قِطَعٍ', 'verified',
     'تجربة المستخدم (خام_مدروس) + corpus الْقِطَعِ/قِطَعِ ×4 — ملتبس بالفعل '
     'قَطَعَ لذا لا يُزرع داخل جملة إلا بعلامة {ق}'),
    ('qTEtyn', 'قطعتين', 'قِطْعَتَيْنِ', 'tier1', 'corpus ×2'),
    ('qTAE', 'قطاع', 'قِطَاعُ', 'tier1', 'corpus وَالْقِطَاعُ'),
    ('qTAEAt', 'قطاعات', 'قِطَاعَاتِ', 'tier1', 'corpus قِطَاعَاتِ'),
    ('qsmp', 'قسمة', 'قِسْمَةَ', 'tier1', 'corpus الْقِسْمَةَ + تجربة قِسْمَةً'),
    ('qsmyn', 'قسمين', 'قِسْمَيْنِ', 'tier1', 'carrier ×2 + الْقِسْمَيْنِ'),
    ('qyAs', 'قياس', 'قِيَاسُ', 'tier1', 'corpus قِيَاسِ + carrier قِيَاسُ'),
    ('yqys', 'يقيس', 'يَقِيسُ', 'tier1', 'corpus وَيَقِيسُ'),
    ('mnTq', 'منطق', 'مَنْطِقِ', 'tier1', 'corpus بِالْمَنْطِقِ'),
    ('mstqym', 'مستقيم', 'مُسْتَقِيمُ', 'tier1', 'corpus الْمُسْتَقِيمُ'),
    ('AnqsAm', 'انقسام', 'انْقِسَامُ', 'tier1', 'corpus الِانْقِسَامُ'),
    ('Hqwq', 'حقوق', 'حُقُوقُ', 'tier1', 'corpus الْحُقُوقُ + تجربة حُقُوقًا'),
    ('twqyt', 'توقيت', 'تَوْقِيتٌ', 'tier1', 'corpus تَوْقِيتٌ'),
    ('qTb', 'قطب', 'قُطْبٌ', 'tier1', 'carrier قُطْبٌ/قُطْبٍ'),
    ('qTbyn', 'قطبين', 'قُطْبَيْنِ', 'tier1', 'carrier ×2'),
    ('qTr', 'قطر', 'قُطْرَ', 'tier1', 'corpus الْقُطْرَ + تجربة قُطْرًا'),
    ('qdrAt', 'قدرات', 'قُدُرَاتِ', 'tier1', 'corpus الْقُدُرَاتِ ×2'),
    # ---- tier2: ملتبسة بفعل أو عامية-غالبة أو بحثية ----
    ('qsm', 'قسم', 'قِسْمٌ', 'tier2',
     'corpus قِسْمٌ — ملتبس بالفعل قَسَمَ → جملةً بالعلامة فقط'),
    ('qSp', 'قصة', 'قِصَّةِ', 'tier2',
     'research talkinarabic + corpus الْقِصَّةِ — عامية غالبة (hits=200)'),
    ('qwp', 'قوة', 'قُوَّةٍ', 'tier2', 'corpus بِقُوَّةٍ — عامية غالبة (hits=175)'),
    ('qymp', 'قيمة', 'قِيمَةُ', 'tier2', 'corpus الْقِيمَةُ — عامية غالبة (hits=492)'),
    ('qdrp', 'قدرة', 'قُدْرَةٍ', 'tier2', 'corpus قُدْرَةٍ (hits=145)'),
    ('qlwb', 'قلوب', 'قُلُوبُ', 'tier2', 'corpus الْقُلُوبُ — تشريح عامي'),
    ('qwY', 'قوى', 'قُوَى', 'tier2', 'corpus الْقُوَى ×2'),
    ('qyAm', 'قيام', 'قِيَامُ', 'tier2', 'research ديني + corpus الْقِيَامُ'),
    ('tEqyd', 'تعقيد', 'تَعْقِيدِ', 'tier2', 'research talkinarabic + corpus التَّعْقِيدِ'),
    ('$qyq', 'شقيق', 'شَقِيقٌ', 'tier2', 'research سجل فصيح + carrier'),
    ('trqym', 'ترقيم', 'تَرْقِيمٌ', 'tier2',
     'carrier — B∩Q: في auto قرار B [g] ساري؛ الزرع في qaf فقط'),
]

DROPPED = {
    'qsmt': 'قسمت — الحصاد قُسِمَتْ (مبني للمجهول) ≠ قصد القائمة (ماضي القسمة)',
    'tqys': 'تقيس — الحصاد بِتَقِيُّسِ = تقييس كلمة مختلفة (التباس الشدة)',
    'qym': 'قيم — الحصاد الْقَيِّمُ = القيِّم كلمة مختلفة',
    'qrb': 'قرّب — الحصاد قُرْبٌ (اسم) ≠ قرّب (فعل) المقصود',
    'tqsym': 'تقسيم — تَقْسِيمُ قافها ساكنة (مدقق البيئي) → غير قابلة للزرع؛ '
             'تبقى على [k] في وضع qaf',
    'yqsm': 'يقسم — يَقْسِمُ قافها ساكنة (يَفْعِلُ) → غير قابلة للزرع؛ '
            'تبقى على [k] في وضع qaf',
}


def definitize(form):
    """قِطْعَةً → قِطْعَة | قُرْآنًا → قُرْآن (لإعادة إلحاق «ال»)."""
    f = re.sub(r'[\u064B-\u064D]+$', '', form)
    if re.search(r'ا[\u064B]$', form):
        f = f[:-1]
    f = re.sub(r'[\u064E\u064F\u0650]$', '', f)
    return f


def main():
    harvest = json.load(open(HARVEST, encoding='utf-8'))
    entries = {}
    for skel, word, form, tier, prov in FINAL:
        env, _ = qaf_env_of(form)
        assert 'ق' in form, form
        assert env == 'deep', f'{skel} {form} env={env} — يجب deep للزرع!'
        entries[skel] = {
            'word_ar': word, 'study_form': form, 'tier': tier,
            'env': env, 'definite_form': definitize(form),
            'provenance': prov,
        }
    meta = {
        'name': 'QAF_Q_STUDY_FORMS — معجم الأشكال المدروسة للقاف الأصيلة',
        'version': 'study-v2.0-final',
        'date': '2026-10-01',
        'mechanism': (
            'زرع الشكل المطابق للتدريب: التدريب شُكِّل بـcatt على جُمل corpus '
            'نفسه، فمخرج catt داخل الجملة هو توزيع التدريب حرفيًا؛ المنفردة '
            'تسقط للفتحة (OOD) فتُنطق همزة. الزرع يستدعي النطق المتعلَّم. '
            'مؤكد سمعيًا من المستخدم: قطعة/قِطْعَةً → قاف أصيلة.'),
        'env_rule': (
            'deep = كسرة/ضمة/تنوين كسر-ضم على القاف (قِ/قُ) → الشكل قابل '
            'للزرع؛ plain = فتحة/سكون (قَ/قْ) → لا زرع (سلوك أصيل أو [k] في '
            'وضع qaf). قاعدة مستنتجة من الأدلة المؤكدة، قابلة للتنقيح بنتائج '
            'qaf_experiment.py على جهاز المستخدم.'),
        'tiers': {
            'verified': 'زرع في auto+qaf داخل الجملة والمنفرد (تأكيد سمعي)',
            'tier1': 'زرع في auto+qaf (أسماء غير ملتبسة، شكل corpus مكسور)',
            'tier2': 'زرع في qaf + علامة {ق} فقط (ملتبسة بفعل/عامية غالبة/بحثية)',
        },
        'sentence_rule': (
            'داخل جملة متعددة الكلمات: لا تُزرع الهياكل الملتبسة بأفعال (qTE/'
            'qsm/yqsm) إلا بعلامة {ق} — catt يعطي القراءة السياقية الصحيحة '
            '(قَطَعَ فعلًا). المنفرد (≤2 كلمات): تُزرع كل الطبقات.'),
        'sources': [
            'corpus نيلتس عبر catt داخل جمل حقيقية (qaf_study_build.py)',
            'جمل حاملة بأسلوب المعلم + تحقق catt مباشر',
            'بحث الويب 2026-10-01: talkinarabic.com (قائمة مصرية لكلمات القاف) '
            '+ Wikipedia Egyptian Arabic phonology + r/learn_arabic',
            'تجربة المستخدم qaf_experiment.py (تأكيد قطعة/قِطْعَةً)',
        ],
        'n_entries': len(entries),
        'n_verified': sum(1 for e in entries.values() if e['tier'] == 'verified'),
        'n_tier1': sum(1 for e in entries.values() if e['tier'] == 'tier1'),
        'n_tier2': sum(1 for e in entries.values() if e['tier'] == 'tier2'),
        'dropped': DROPPED,
    }
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump({'meta': meta, 'entries': entries}, f,
                  ensure_ascii=False, indent=1)

    print(f'المعجم النهائي: {len(entries)} هيكلًا '
          f'(verified={meta["n_verified"]}, tier1={meta["n_tier1"]}, '
          f'tier2={meta["n_tier2"]})')
    print(f'محذوفات موثقة: {len(DROPPED)} — استبعاد قوي (homograph)')
    print(f'حُفظ: {OUT}')
    print()
    print('# ---- مقتطف التضمين في infer.py ----')
    print('QAF_Q_STUDY_FORMS = {')
    for skel, e in sorted(entries.items()):
        print(f"    '{skel}': '{e['study_form']}',  # {e['word_ar']}")
    print('}')
    print()
    print('QAF_Q_VERIFIED = frozenset({')
    ver = [s for s, _, _, t, _ in FINAL if t == 'verified']
    print('    ' + ', '.join(f"'{s}'" for s in ver) + ',')
    print('})')
    print('QAF_Q_TIER1 = frozenset({')
    t1 = [s for s, _, _, t, _ in FINAL if t in ('tier1', 'verified')]
    print('    ' + ', '.join(f"'{s}'" for s in t1) + ',')
    print('})')
    print('QAF_Q_SENTENCE_SKIP = frozenset({')
    print("    'qTE', 'qsm',   # ملتبسة بأفعال: قَطَعَ/قَسَمَ")
    print('})')


def qaf_env_of(form):
    idx = form.find('ق')
    after = form[idx + 1:idx + 2]
    m = {'\u0650': 'deep', '\u064F': 'deep', '\u064D': 'deep',
         '\u064C': 'deep', '\u064E': 'plain_fatha', '\u0652': 'plain_sukun'}
    return m.get(after, 'plain_bare'), after


if __name__ == '__main__':
    main()
