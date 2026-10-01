# -*- coding: utf-8 -*-
"""det_build_lists.py (v2) — يبني det_lists.json من دليل التشكيل v2.1.

مصدر الحقيقة الواحد: work/tashkeel_guide_v2.md + tanween_closed_list.json
+ tier_a_map.json. مخرجه: inference/lib/det_data/det_lists.json

v2 إصلاحات جوهرية:
  - طي الهمزات (أ إ آ → ا) في كل المفاتيح والبحث: الخام يكتب كثيرًا بلا
    همزة (ابدا/احيانا) بينما الدليل يشكّل بهمزة (أَبَدَاً) — المفاتيح المطوية
    تتطابق، والاستبدال في الوحدة «زرع حركات فقط» يحفظ حروف الخام حرفيًا.
  - ي-1أ/ب تنتهي عند '**ج)' (كانت تسحب عبارات ي-1ج بالخطأ).
  - ي-1د يقسم على '/' أيضًا (بِنَاءَاً/بَنَاا).
  - Tier C تُقرأ ككتلة متعددة الأسطر.
  - تحقق: هيكل كل شكل (مطويًا) == مفتاحه.
"""
import json
import os
import re

GUIDE = '/home/z/my-project/work/tashkeel_guide_v2.md'
TANWEEN_LIST = '/home/z/my-project/work/tanween_closed_list.json'
TIER_A_MAP = '/home/z/my-project/work/tier_a_map.json'
OUT = ('/home/z/my-project/work/github_repo/eiqaz-tts/'
       'inference/lib/det_data/det_lists.json')

STRIP = re.compile(r'[\u064B-\u0652\u0653-\u0655\u0670]')
AR_ONLY = re.compile(r'^[\u0621-\u064A\u064B-\u0652\u0653-\u0655\u0670 ]+$')
ALEF_FOLD = str.maketrans({'\u0623': '\u0627', '\u0625': '\u0627',
                           '\u0622': '\u0627'})

# هياكل غير لائقة (طبقة 1 من كاشف الأمان) — تُستبعد أي قائمة مجمدة تصطدم بها
TIER1_SK = {
    'كس', 'خرا', 'زب', 'زبر', 'ينك', 'ناك', 'قحبة', 'شرموطة', 'خول',
    'متناك', 'متناكة', 'عرص', 'فشخ', 'طيزة', 'است', 'قواد', 'كوادة',
}

# كلمات ملتبسة/مرجأة — لا تُجمّد آليًا (يبقى شكل catt_eo ويُعلَّم للمراجعة):
#   اما = أَمَّا «حينئذ» vs إِمَّا «أو» (معنيان — السياق فقط يحسم)
#   ان  = إِنَّ/إِنِّ (توكيد بكسرة مشددة — سياقي)
#   هو  = هُوَ/هُوْ (§ج-5: حكم على الحرف التالي — قاعدة مخصصة في الوحدة)
#   هي  = هِيَ دائمًا (لكن توحّد المعالجة في قاعدة §ج-5 نفسها)
#   فارغا/فيلا = موسومان «يُراجع» في ي-1د نفسها
AMBIGUOUS = {'اما', 'ان', 'هو', 'هي', 'فارغا', 'فيلا'}

skel = lambda w: STRIP.sub('', w)
fold = lambda s: s.translate(ALEF_FOLD)
fskel = lambda w: fold(skel(w))     # هيكل مطوي الهمزات — مفتاح البحث


def section(text, start, end):
    i = text.find(start)
    j = text.find(end, i + len(start))
    return text[i:j]


def clean_tok(t):
    t = re.sub(r'\([^)]*\)', '', t)
    return t.strip().strip('|').strip()


def valid_form(form):
    return (bool(form) and bool(AR_ONLY.match(form))
            and bool(skel(form)) and '  ' not in form)


def add_map(mp, form, src, report, allow_conflict_skip=True, override=False):
    """يضيف شكلًا مجمدًا بمفتاح الهيكل المطوي + تحقق أمانة الحروف."""
    s = fskel(form)
    if s in TIER1_SK:
        report['excluded'].append([src, form, 'تصادم أمان صوتي'])
        return
    if s in AMBIGUOUS:
        report['ambiguous_skipped'].append([src, form])
        return
    if s in mp and mp[s] != form:
        if override:
            report['overridden'].append([src, s, mp[s], form])
        elif allow_conflict_skip:
            report['conflicts'].append([src, s, mp[s], form])
        if not override:
            return
    mp[s] = form


def main():
    guide = open(GUIDE, encoding='utf-8').read()
    closed = {fold(w) for w in
              json.load(open(TANWEEN_LIST, encoding='utf-8'))['words']}
    ta = json.load(open(TIER_A_MAP, encoding='utf-8'))
    report = {'excluded': [], 'conflicts': [], 'counts': {},
              'ambiguous_skipped': [], 'overridden': []}

    # ---- 1) قائمة التنوين المجمدة (ي-1أ + ي-1ب + ي-1د) --------------------
    tanween_map = {}
    sec = section(guide, '**أ) الظروف القياسية', '**هـ) هياكل مستبعدة')
    for line in sec.splitlines():
        if '|' not in line:
            continue
        for tok in line.split('|'):
            f = clean_tok(tok)
            if not valid_form(f):
                if f:
                    report['excluded'].append(['ي-1', f, 'شكل غير صالح'])
                continue
            # ي-1د فقط: متغيرات بلا تنوين تُمنَّن كأصلها
            if '\u064B' not in f and fskel(f) in closed \
                    and 'صيغ فصيحة' not in line:
                pass  # يُعالج أدناه بمسح ي-1د المخصص
            add_map(tanween_map, f, 'ي-1', report)
    # ي-1د: أضف التنوين للمتغيرات المذكورة بلا ً (غَالَمَا → غَالَمَاً).
    # ي-1د لاحقة وأخص من ي-1أ/ب («تُمنَن كأصلها») → تتجاوز (override) لا تتعارض.
    sec_d = section(guide, '**د) متغيرات نقل', '**هـ) هياكل مستبعدة')
    for line in sec_d.splitlines():
        if '|' not in line:
            continue
        for tok in re.split(r'[|/]', line):
            f = clean_tok(tok)
            if not valid_form(f):
                continue
            if fskel(f) in AMBIGUOUS:
                report['ambiguous_skipped'].append(['ي-1د', f])
                continue
            if '\u064B' not in f:
                if f.endswith(('ا', 'ى', 'ة')):
                    f2 = f + '\u064B'
                else:
                    f2 = f[:-1] + '\u064E' + f[-1] + 'ا' + '\u064B'
                add_map(tanween_map, f2, 'ي-1د', report, override=True)
            else:
                add_map(tanween_map, f, 'ي-1د', report, override=True)

    # ---- 2) الصيغ المجمدة بلا تنوين (ي-1ج) — عبارات متعددة الكلمات -------
    frozen_phrases = []
    sec = section(guide, '**ج) صيغ فصيحة مجمدة', '**د) متغيرات نقل')
    for line in sec.splitlines():
        if '|' not in line:
            continue
        for tok in line.split('|'):
            f = clean_tok(tok)
            if not valid_form(f):
                if f:
                    report['excluded'].append(['ي-1ج', f, 'شكل غير صالح'])
                continue
            if any(fskel(w) in AMBIGUOUS for w in f.split()):
                report['ambiguous_skipped'].append(['ي-1ج', f])
                continue
            frozen_phrases.append({
                'pat': [fskel(w) for w in f.split()],
                'rep': f.split(),
            })

    # ---- 3) §ز الكلمات الشائعة -------------------------------------------
    common_map = {}
    sec = section(guide, '## ز. قائمة الكلمات الشائعة', '## ح. الكلمات الدخيلة')
    for line in sec.splitlines():
        if '|' not in line:
            continue
        for tok in line.split('|'):
            f = clean_tok(tok)
            parts = ([p.strip() for p in f.split('/') if p.strip()]
                     if '/' in f else ([f] if f else []))
            for p in parts:
                if not valid_form(p):
                    if p:
                        report['excluded'].append(['§ز', p, 'شكل غير صالح'])
                    continue
                add_map(common_map, p, '§ز', report)
    # هو (هُوَ/هُوْ) وإنَّ/إنِّ واما (أمَّا/إمَّا): ملتبسة — مستبعدة آليًا من
    # كل الخرائط؛ هو/هي تعالجهما قاعدة §ج-5 المخصصة في det_tashkeel،
    # والباقي يبقى على شكل catt_eo ويُعلَّم للمراجعة

    # ---- 4) §ح الدخائل (نثر مفصول بـ،) ------------------------------------
    sec = section(guide, '## ح. الكلمات الدخيلة', '## ط. المصطلحات العلمية')
    body = ' '.join(l for l in sec.splitlines()
                    if l.strip() and not l.strip().startswith(('تُشكَّل', '**')))
    for tok in re.split(r'[،,]', body):
        f = clean_tok(tok)
        if not valid_form(f) or ' ' in f:
            if f and ' ' not in f:
                report['excluded'].append(['§ح', f, 'شكل غير صالح'])
            continue
        add_map(common_map, f, '§ح', report)

    # ---- 5) Tier C الدينية (ي-3 — كتلة متعددة الأسطر) --------------------
    tier_c_map = {}
    m = re.search(r'\*\*Tier C \(9 كلمات[^\n]*\n(.+?)\*\*رمضان', guide, re.S)
    block = ' '.join(m.group(1).split())
    for tok in block.split('|'):
        f = clean_tok(tok)
        if not valid_form(f):
            if f:
                report['excluded'].append(['TierC', f, 'شكل غير صالح'])
            continue
        add_map(tier_c_map, f, 'TierC', report)

    # ---- التحقق النهائي ----------------------------------------------------
    for name, mp in [('tanween', tanween_map), ('common', common_map),
                     ('tier_c', tier_c_map)]:
        for k, v in mp.items():
            assert fskel(v) == k, f'{name}: هيكل {v!r} != مفتاح {k!r}'
    for p in frozen_phrases:
        assert [fskel(w) for w in p['rep']] == p['pat'], f'ي-1ج: {p}'

    # تغطية القائمة المغلقة: مباشرة أو بسلب سابقة و/ف
    def covered(s):
        if s in tanween_map:
            return 'direct'
        if s[0] in 'وف' and s[1:] in tanween_map:
            return 'prefix'
        return None
    cov = {s: covered(s) for s in closed}
    n_direct = sum(1 for v in cov.values() if v == 'direct')
    n_prefix = sum(1 for v in cov.values() if v == 'prefix')
    missing = sorted(s for s, v in cov.items() if not v)

    doc = {
        'version': 'det-v1.2 — 2026-10-01',
        'source': 'دليل التشكيل المصري v2.1 (المعتمد 2026-09-30) — بناء آلي '
                  'scripts/det_build_lists.py (طي الهمزات في المفاتيح؛ '
                  'الاستبدال زرع حركات يحفظ حروف الخام). v1.2: ي-1د تتجاوز '
                  'ي-1أ/ب؛ الملتبسات (اما/ان/هو/هي/فارغا/فيلا) مستبعدة من كل '
                  'الخرائط؛ إضافة closed_set الكاملة (186 هيكل §ب).',
        'tanween_map': dict(sorted(tanween_map.items())),
        'closed_set': sorted(closed),
        'frozen_phrases': frozen_phrases,
        'common_map': dict(sorted(common_map.items())),
        'tier_c_map': dict(sorted(tier_c_map.items())),
        'tier_a': [{'pat': [fskel(w) for w in e['raw'].split()],
                    'rep': e['out'].split()} for e in ta['entries']],
        'rabbena': [{'pat': [fskel(w) for w in e['raw'].split()],
                     'rep': e['out'].split()} for e in ta['rabbena']],
        'meta': {
            'tanween_closed_list_size': len(closed),
            'tanween_direct': n_direct,
            'tanween_via_prefix': n_prefix,
            'tanween_missing': missing,
            'frozen_phrases': len(frozen_phrases),
            'common_words': len(common_map),
            'tier_c_words': len(tier_c_map),
            'tier_a_forms': len(ta['entries']),
            'rabbena_forms': len(ta['rabbena']),
        },
        'build_report': report,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(doc, open(OUT, 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print(f'tanween: مباشر {n_direct} + بسابقة {n_prefix} / {len(closed)} '
          f'(بلا شكل: {len(missing)})')
    print(f'frozen phrases: {len(frozen_phrases)}')
    print(f'common: {len(common_map)} | tier_c: {len(tier_c_map)} | '
          f'tier_a: {len(ta["entries"])} | rabbena: {len(ta["rabbena"])}')
    print(f'مستبعدة: {len(report["excluded"])} | تعارضات متبقية: '
          f'{len(report["conflicts"])} | ملتبسة مستبعدة: '
          f'{len(report["ambiguous_skipped"])} | تجاوزات ي-1د: '
          f'{len(report["overridden"])}')
    for c in report['conflicts']:
        print('  تعارض متبقٍ:', c)
    for a in report['ambiguous_skipped']:
        print('  ملتبس مستبعد:', a)
    for o in report['overridden']:
        print('  تجاوز ي-1د:', o)
    print(f'بلا شكل مجمد نهائيًا: {missing}')


if __name__ == '__main__':
    main()
