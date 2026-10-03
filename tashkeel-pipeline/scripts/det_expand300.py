# -*- coding: utf-8 -*-
"""det_expand300.py — التحقق الموسّع (مرحلة 2): 300 وحدة niletts حية.

بلا مرجع معتمد — يقيس القيمة الميكانيكية فقط:
  * بوابات §ل (هيكل/تنوين/ة/كثافة) قبل/بعد الطبقة على مخرجات catt_eo الحية
  * هيستوغرام الإصلاحات، الأعلام، الارتجاعات الأمنية، الزمن
عينة: أول 300 وحدة من ph2_input_full (كما هي — بلا انتقاء) مع استبعاد
المحجورات الست (قرار المستخدم 2026-09-30).
"""
import json
import sys
import re
import time

sys.path.insert(0, '/home/z/my-project/work/github_repo/eiqaz-tts/inference/lib')
sys.path.insert(0, '/home/z/my-project/scripts')
sys.path.insert(0, '/home/z/my-project/work/tts_arabic_pkg')

from det_tashkeel import DetTashkeel, DIACS, TANWEEN, FATHATAN
from tts_arabic.vocalizer.models.core import vocalize
from ph1_safety import check_vocalized_word

QUARANTINED = json.load(open(
    '/home/z/my-project/work/ph1_quarantine.json'))
QIDS = set(QUARANTINED if isinstance(QUARANTINED, list)
           else QUARANTINED.get('ids', QUARANTINED.keys()))
STRIP_D = re.compile('[' + ''.join(DIACS) + ']')
FOLD = str.maketrans({'\u0623': '\u0627', '\u0625': '\u0627',
                      '\u0622': '\u0627'})


def gates(out, raw, closed):
    conv = STRIP_D.sub('', raw.replace('\u060C', ',')
                       .replace('\u061F', '?'))
    sk = STRIP_D.sub('', out) == conv
    tan_ok = True
    for w in out.split():
        kf = STRIP_D.sub('', w).strip(
            '.,?!:;"()«»\u201C\u201D…-').translate(FOLD)
        for d in w:
            if d in TANWEEN:
                if d != FATHATAN:
                    tan_ok = False
                else:
                    ks = kf[1:] if kf[:1] in 'وف' else kf
                    if kf not in closed and ks not in closed \
                            and kf != 'خيرا':
                        tan_ok = False
    words = [w for w in out.split()
             if any('\u0621' <= c <= '\u064A' for c in w)]
    dens = (sum(1 for c in out if c in DIACS) / len(words)
            if words else 0.0)
    taa_ok = not re.search(r'\u0629[\u064B-\u0652]$',
                           ' '.join(out.split()) if out else '')
    for w in out.split():
        if re.search(r'\u0629[\u064B-\u0652]+(\s|$)', w + ' '):
            taa_ok = False
    return {'skeleton': sk, 'tanween': tan_ok, 'taa': taa_ok,
            'density': round(dens, 2)}


if __name__ == '__main__':
    dt = DetTashkeel()
    closed = dt.closed_set | {'خيرا'}
    items = [x for x in json.load(open(
        '/home/z/my-project/work/ph2_input_full.json'))
        if x['id'] not in QIDS][:300]
    g_before = {'skeleton': 0, 'tanween': 0, 'taa': 0, 'density': 0}
    g_after = {'skeleton': 0, 'tanween': 0, 'taa': 0, 'density': 0}
    all_fixes, flag_kinds = {}, {}
    n_wc, n_skel, n_safety = 0, 0, 0
    t0 = time.time()
    t_catt = t_det = 0.0
    rows = []
    for it in items:
        raw = it['text']
        tc = time.time()
        catt = vocalize(raw, model='catt_eo')
        t_catt += time.time() - tc
        td = time.time()
        r = dt.process(raw, catt, safety_fn=check_vocalized_word)
        t_det += time.time() - td
        gb, ga = gates(catt, raw, closed), gates(r['out'], raw, closed)
        for g, x in ((g_before, gb), (g_after, ga)):
            for k in g:
                g[k] += int(x[k]) if k != 'density' else int(x[k] >= 1.2)
        for k, v in r['fixes'].items():
            all_fixes[k] = all_fixes.get(k, 0) + v
        for f in r['flags']:
            key = f.split(':')[0]
            flag_kinds[key] = flag_kinds.get(key, 0) + 1
        n_wc += int('word_count_mismatch' in ' '.join(r['flags']))
        n_skel += int(not r['checks']['skeleton_ok'])
        n_safety += r['fixes']['safety_reverted']
        rows.append({'id': it['id'], 'det': r['out'], 'gates': ga})
    n = len(items)
    total_t = time.time() - t0
    print(f'=== التحقق الموسّع: {n} وحدة niletts (حية catt_eo) ===\n')
    print('--- البوابات: قبل ← بعد ---')
    for k in ('skeleton', 'tanween', 'taa', 'density'):
        print(f'  {k:9s}: {g_before[k]:3d}/{n} ({g_before[k] / n * 100:.0f}%) '
              f'← {g_after[k]:3d}/{n} ({g_after[k] / n * 100:.0f}%)')
    print(f'\nوحدات فشل الهيكل (طبقة): {n_skel} | تعذر محاذاة كلمية: {n_wc} '
          f'| ارتجاعات أمنية: {n_safety}')
    print(f'الأعلام: {flag_kinds}')
    print('\n--- هيستوغرام الإصلاحات ---')
    for k, v in sorted(all_fixes.items(), key=lambda x: -x[1]):
        if v:
            print(f'  {k:22s}: {v}')
    print(f'\nالزمن: catt {t_catt:.1f}s + طبقة {t_det:.1f}s = '
          f'{total_t:.1f}s لـ{n} وحدة '
          f'({t_det / n * 1000:.1f} ms/وحدة للطبقة)')
    out = ('/home/z/my-project/work/github_repo/eiqaz-tts/'
           'tashkeel-pipeline/state/det_expand300_report.json')
    json.dump({'n': n, 'gates_before': g_before, 'gates_after': g_after,
               'all_fixes': all_fixes, 'flag_kinds': flag_kinds,
               'skeleton_fail': n_skel, 'word_count_mismatch': n_wc,
               'safety_reverted': n_safety, 'rows': rows},
              open(out, 'w', encoding='utf-8'), ensure_ascii=False)
    print(f'حُفظ: {out}')
