#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""qaf_study_upgrade.py — PATCH 10: ترقية طبقات معجم الأشكال المدروسة
بناءً على نتائج qaf_experiment.py run1 (تصنيف آلي على جهاز المستخدم).

المدخلات:
  --run      ملف نتائج التجربة (experiment_run1.json)
  --forms    ملف المعجم الحالي (validation/qaf_q_study_forms.json)
  --out      ملف المعجم الجديد (افتراضيًا: يكتب فوق --forms)

القرارات (من الأدلة — run1, speaker=0):
  PROMOTE → verified : qSp (قصة) qTb (قطب) qTbyn (قطبين)   [قياس آلي q?]
  DEMOTE  → tier2    : qTr (قطر) yqys (يقيس)               [الزرع ← همزة]
  FORM UPDATE         : qTr: قُطْرَ → قُطْرًا                  [التنوين = الشكل العميق المقيس]
  الكل الآخر          : كما هو + تسجيل القياس الآلي machine_run1

نقطة المعايرة الحاسمة (توثيق): qTEp (قطعة) قاست 'g' بينما أذن المستخدم
أكدتها 'ق أصيلة' — المصنّف يقيس الجهر/التوقيت لا مكان النطق (المؤخَّر)،
فحكم 'g' على الأشكال المزروعة قد يبقى مدركًا كقاف أصيلة (مجرور مؤخَّر).
لذلك لا نُنزل أي هيكل من auto بناءً على 'g' وحده — فقط 'ء' (الزرع يُنتج
الهمزة فعليًا = ضرر مؤكد) يُنزل. 'q?' (صامت + VOT≥30ms من توكن '<')
هو أقوى إشارة عمق آلية → ترقية.
"""
import argparse
import copy
import json
import sys

RUN_META = {
    'name': 'run1',
    'date': '2026-10-02',
    'speaker': 0,
    'n_words': 24,
    'n_files': 120,
    'n_study_all': 29,
    'source': 'validation/qaf_experiment_run1.json (نسخة من مخرجات جهاز المستخدم)',
}

PROMOTE_VERIFIED = {
    'qSp': 'قياس آلي q? (VOT=38ms, جهر=0.21) + بحث talkinarabic (قصة من كلمات القاف المحتفظة)',
    'qTb': 'قياس آلي q? (VOT=42ms, جهر=0.22)',
    'qTbyn': 'قياس آلي q? (VOT=38ms, جهر=0.13)',
}
DEMOTE_TIER2 = {
    'qTr': ('الزرع بصيغة قُطْرَ قاس همزة مرتين (study_all + افتراضي في words) '
            'بينما قُطْرًا (بالتنوين) قاس q? عميقًا (VOT=46ms) — تنزيل من auto '
            'وتحديث الصيغة إلى العميقة المقيسة'),
    'yqys': 'الزرع بصيغة يَقِيسُ قاس همزة (study_all) — تنزيل من auto؛ بيئتها عميقة طبيعًا فلا تُزرع عمليًا',
}
FORM_UPDATES = {
    'qTr': ('قُطْرًا', 'قُطْرَ',
            'قُطْرًا (بالتنوين) = الشكل الوحيد الذي قاس q? عميقًا في run1 '
            '(words/خام_مدروس VOT=46ms)؛ قُطْرَ بلا تنوين قاس ء مرتين'),
}

CALIBRATION_NOTE = (
    "معايرة run1: qTEp (قطعة) قاست 'g' (جهر=0.29, VOT=8ms) بينما أذن المستخدم "
    "أكدتها قافًا أصيلة فصحى — المصنّف numpy يقيس جهر الإغلاق+VOT ولا يقيس مكان "
    "النطق، فمجرور مؤخَّر [ɢ] يُقاس 'g' ويُسمع أصيلًا. لذلك: 'g' على الأشكال "
    "المزروعة لا يكفي للتنزيل من auto (البقاء)، 'ء' = زرع ضار (تنزيل)، "
    "'q?' = أقوى إشارة عمق (ترقية). ملاحظة ثانية: كاف_k الحقيقي [k] يُقاس 'q?' "
    "أيضًا (صامت+VOT طويل) — المصنّف لا يفرز [k] عن [q]."
)

KEEP_VERIFIED = {
    'qTEp': "بقاء verified: تأكيد سمعي من المستخدم (أصيلة) رغم قياس 'g' — نقطة المعايرة",
    'qTE': "بقاء verified: تأكيد سمعي من المستخدم",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='/home/z/my-project/work/qaf_results/experiment_run1.json')
    ap.add_argument('--forms', default='/home/z/my-project/work/github_repo/eiqaz-tts/'
                        'validation/qaf_q_study_forms.json')
    ap.add_argument('--out', default=None)
    ap.add_argument('--report', default='/home/z/my-project/work/qaf_results/'
                        'qaf_upgrade_run1_report.md')
    args = ap.parse_args()
    out = args.out or args.forms

    run = json.load(open(args.run, encoding='utf-8'))
    forms = json.load(open(args.forms, encoding='utf-8'))
    entries = forms['entries']
    study_all = run.get('study_all', {})
    if len(study_all) != len(entries):
        sys.exit(f'[خطأ] study_all={len(study_all)} بينما المعجم={len(entries)} — عدم تطابق')

    new = copy.deepcopy(forms)
    ne = new['entries']
    changes = []

    # 1) تسجيل القياس الآلي لكل هيكل
    for sk, info in study_all.items():
        if sk not in ne:
            sys.exit(f'[خطأ] هيكل {sk} في النتائج وغير موجود في المعجم')
        ne[sk]['machine_run1'] = {
            'verdict': info.get('verdict'),
            'detail': info.get('detail'),
            'form_used': info.get('form'),
            'wav': info.get('wav'),
            'speaker': RUN_META['speaker'],
        }

    # 2) الترقيات → verified
    for sk, why in PROMOTE_VERIFIED.items():
        old = ne[sk]['tier']
        ne[sk]['tier'] = 'verified'
        ne[sk]['provenance'] = ne[sk].get('provenance', '') + ' | ترقية run1: ' + why
        changes.append((sk, ne[sk]['word_ar'], old, 'verified', why))

    # 3) التنزيلات → tier2
    for sk, why in DEMOTE_TIER2.items():
        old = ne[sk]['tier']
        ne[sk]['tier'] = 'tier2'
        ne[sk]['provenance'] = ne[sk].get('provenance', '') + ' | تنزيل run1: ' + why
        changes.append((sk, ne[sk]['word_ar'], old, 'tier2', why))

    # 4) تحديث الصيغ (بعد التنزيل — الصيغة الجديدة تخدم qaf والعلامة {ق})
    for sk, (new_form, old_form, why) in FORM_UPDATES.items():
        if ne[sk]['study_form'] != old_form:
            sys.exit(f'[خطأ] صيغة {sk} الحالية {ne[sk]["study_form"]} ≠ المتوقع {old_form}')
        ne[sk]['study_form'] = new_form
        ne[sk]['provenance'] = ne[sk].get('provenance', '') + ' | صيغة run1: ' + why
        changes.append((sk, ne[sk]['word_ar'], f'صيغة {old_form}', f'صيغة {new_form}', why))

    # 5) تاكيد البقاء المؤكد سمعيًا
    for sk, note in KEEP_VERIFIED.items():
        ne[sk]['provenance'] = ne[sk].get('provenance', '') + ' | run1: ' + note

    # 6) الميتا
    from collections import Counter
    tc = Counter(v['tier'] for v in ne.values())
    m = new['meta']
    m['version'] = 'study-v2.1'
    m['date'] = RUN_META['date']
    m['tiers']['verified'] = ('زرع في auto+qaf داخل الجملة والمنفرد '
                              '(تأكيد سمعي من المستخدم أو قياس آلي q? من run1)')
    m['tiers']['tier2'] = 'زرع في qaf + علامة {ق} فقط (ملتبسة بفعل/عامية غالبة/زرعها يقيس همزة)'
    m['n_verified'] = tc.get('verified', 0)
    m['n_tier1'] = tc.get('tier1', 0)
    m['n_tier2'] = tc.get('tier2', 0)
    m['machine_runs'] = {RUN_META['name']: RUN_META}
    m['calibration'] = CALIBRATION_NOTE
    m['sources'] = m.get('sources', []) + [
        'run1 (2026-10-02): qaf_experiment.py على جهاز المستخدم — speaker 0، '
        '24 كلمة × 5 تصييرات + 29 هيكلًا study-all + 12 جملة استماع '
        '(validation/qaf_experiment_run1.json)',
    ]

    json.dump(new, open(out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

    # 7) التقرير
    lines = [
        '# PATCH 10 — ترقية طبقات الأشكال المدروسة (run1)',
        '',
        f'- المصدر: {RUN_META["source"]} — speaker {RUN_META["speaker"]}',
        f'- المعجم الجديد: {out} (study-v2.1)',
        f'- العد الجديد: verified={m["n_verified"]} tier1={m["n_tier1"]} tier2={m["n_tier2"]}',
        '',
        '## التغييرات',
        '',
        '| الهيكل | الكلمة | من | إلى | السبب |',
        '|---|---|---|---|---|',
    ]
    for sk, w, old, newt, why in changes:
        lines.append(f'| {sk} | {w} | {old} | {newt} | {why} |')
    lines += [
        '',
        '## مصفوفة القياس الكاملة (study_all ×29)',
        '',
        '| الهيكل | الكلمة | الصيغة | القياس | الحكم | الطبقة الجديدة |',
        '|---|---|---|---|---|---|',
    ]
    for sk, info in study_all.items():
        e = ne[sk]
        det = (info.get('detail', '') or '').split('→')[-1].strip()
        lines.append(f'| {sk} | {e["word_ar"]} | {info.get("form")} | '
                     f'{det} | {info.get("verdict")} | {e["tier"]} |')
    lines += ['', '## المعايرة', '', CALIBRATION_NOTE, '']
    open(args.report, 'w', encoding='utf-8').write('\n'.join(lines))

    print(f'[ok] كُتب المعجم الجديد: {out}')
    print(f'[ok] كُتب التقرير: {args.report}')
    print(f'    verified={m["n_verified"]} tier1={m["n_tier1"]} tier2={m["n_tier2"]}')
    for sk, w, old, newt, why in changes:
        print(f'    {sk:<8s} {w:<8s} {old} → {newt}')


if __name__ == '__main__':
    main()
