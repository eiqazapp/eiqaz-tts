# -*- coding: utf-8 -*-
"""det_eval50.py — القياس الإلزامي قبل أي تفعيل (طلب المستخدم المصحَّح).

على عينة الـ50 جملة المعتمدة (step0_results.json):
  خط الأساس: catt_eo الخام مقابل التشكيل اليدوي/GLM المعتمد (my_text)
  الطبقة   : det_tashkeel(catt_eo الخام) مقابل المعتمد نفسه
كلمة-بكلمة، كل الحركات (ليس نهاية الكلمة فقط) — نسبة التطابق الحقيقية،
مع عدّ الانحدارات (كلمات كانت مطابقة في catt فأفسدتها الطبقة) وعدّ
الإصلاحات (كلمات غير مطابقة في catt فأصلحتها الطبقة).
"""
import json
import sys
import re

sys.path.insert(0, '/home/z/my-project/work/github_repo/eiqaz-tts/inference/lib')
sys.path.insert(0, '/home/z/my-project/scripts')

from det_tashkeel import DetTashkeel, DIACS, TANWEEN
from ph1_safety import check_vocalized_word

RES = ('/home/z/my-project/work/github_repo/eiqaz-tts/'
       'tashkeel-pipeline/state/early/step0_results.json')
STRIP_D = re.compile('[' + ''.join(DIACS) + ']')


def _units(w):
    out, cur = [], None
    for c in w:
        if '\u0621' <= c <= '\u064A' and c != '\u0640':
            if cur:
                out.append(cur)
            cur = [c, '']
        elif cur is not None and c in DIACS:
            cur[1] += c
        elif cur is not None:
            out.append(cur)
            cur = None
    if cur:
        out.append(cur)
    return out


def cmp_word(a, b):
    """تصنيف زوج كلمات: exact / final_only / internal / skeleton."""
    if a == b:
        return 'exact'
    sa, sb = STRIP_D.sub('', a), STRIP_D.sub('', b)
    if sa != sb:
        return 'skeleton'
    ua = [(l, d) for l, d in _units(a)]
    ub = [(l, d) for l, d in _units(b)]
    if len(ua) != len(ub):
        return 'skeleton'
    diffs = [i for i, (x, y) in enumerate(zip(ua, ub)) if x != y]
    if not diffs:
        return 'exact'
    if diffs == [len(ua) - 1]:
        return 'final_only'
    return 'internal'


if __name__ == '__main__':
    dt = DetTashkeel()
    rows = json.load(open(RES))['results']
    tally = {'baseline': {'exact': 0, 'final_only': 0, 'internal': 0,
                          'skeleton': 0, 'total': 0, 'misalign': 0},
             'layered': {'exact': 0, 'final_only': 0, 'internal': 0,
                         'skeleton': 0, 'total': 0, 'misalign': 0}}
    fixed_by_layer, broken_by_layer = [], []
    all_fixes = {}
    gate_fail = {'skeleton': 0, 'n': 0}
    examples = {'internal': [], 'final_only': [], 'broken': []}

    for r in rows:
        raw, catt, my = r['raw'], r['catt_text'], r['my_text']
        det = dt.process(raw, catt, safety_fn=check_vocalized_word)
        for k, v in det['fixes'].items():
            all_fixes[k] = all_fixes.get(k, 0) + v
        gate_fail['n'] += 1
        gate_fail['skeleton'] += int(not det['checks']['skeleton_ok'])
        mw = my.split()
        cw = catt.replace('\u060C', ',').replace('\u061F', '?').split()
        dw = det['out'].split()
        if not (len(mw) == len(cw) == len(dw)):
            tally['baseline']['misalign'] += 1
            tally['layered']['misalign'] += 1
            continue
        for b_w, d_w, m_w in zip(cw, dw, mw):
            cb = cmp_word(b_w, m_w)
            cd = cmp_word(d_w, m_w)
            tally['baseline'][cb if cb != 'exact' else 'exact'] += 1
            tally['baseline']['total'] += 1
            tally['layered'][cd if cd != 'exact' else 'exact'] += 1
            tally['layered']['total'] += 1
            if cb != 'exact' and cd == 'exact':
                fixed_by_layer.append((r['utt'], b_w, d_w))
            if cb == 'exact' and cd != 'exact':
                broken_by_layer.append((r['utt'], b_w, d_w, m_w))
                if len(examples['broken']) < 8:
                    examples['broken'].append((r['utt'], b_w, d_w, m_w))
            if cd == 'internal' and len(examples['internal']) < 10:
                examples['internal'].append((r['utt'], d_w, m_w))
            if cd == 'final_only' and len(examples['final_only']) < 6:
                examples['final_only'].append((r['utt'], d_w, m_w))

    t = tally['baseline']['total']
    print('=== القياس الإلزامي: عينة الـ50 المعتمدة (كلمة-بكلمة، '
          'كل الحركات) ===\n')
    for tag in ('baseline', 'layered'):
        x = tally[tag]
        name = ('خط الأساس: catt_eo الخام' if tag == 'baseline'
                else 'الطبقة الحتمية (catt_eo + تصحيح جزئي)')
        print(f'--- {name} ---')
        for k in ('exact', 'final_only', 'internal', 'skeleton'):
            print(f'  {k:11s}: {x[k]:4d}/{t} ({x[k] / t * 100:5.1f}%)')
        print(f'  تطابق تام: {x["exact"] / t * 100:.1f}% | '
              f'+تسامح النهايات: '
              f'{(x["exact"] + x["final_only"]) / t * 100:.1f}%\n')
    b, l = tally['baseline'], tally['layered']
    print(f'فضل الطبقة على التطابق التام: '
          f'{(l["exact"] - b["exact"]) / t * 100:+.1f} نقطة مئوية '
          f'({b["exact"]} → {l["exact"]} كلمة)')
    print(f'فضلها بعد تسامح النهايات: '
          f'{((l["exact"] + l["final_only"]) - (b["exact"] + b["final_only"])) / t * 100:+.1f} نقطة')
    print(f'وحدات بلا محاذاة كلمية: {b["misalign"]}')
    print(f'وحدات فشل هيكل بعد الطبقة: {gate_fail["skeleton"]}/'
          f'{gate_fail["n"]}\n')
    print(f'--- إصلاحات الطبقة: {len(fixed_by_layer)} كلمة صارت مطابقة ---')
    from collections import Counter
    print('  أكثر الكلمات المُصلحة:', Counter(
        STRIP_D.sub('', w[1]) for w in fixed_by_layer).most_common(12))
    print(f'\n--- انحدارات (كانت مطابقة ففسدت): {len(broken_by_layer)} ---')
    for u, bw, dww, mww in examples['broken']:
        print(f'  {u}: catt={bw} | طبقة={dww} | معتمد={mww}')
    print('\n--- عينة فروق داخلية (طبقة ← معتمد) — خارج نطاق الطبقة ---')
    for u, dww, mww in examples['internal']:
        print(f'  {u}: {dww} ← {mww}')
    print('\n--- عينة فروق نهايات فقط (طبقة ← معتمد) ---')
    for u, dww, mww in examples['final_only']:
        print(f'  {u}: {dww} ← {mww}')
    print('\n--- إجمالي الإصلاحات الميكانيكية على الـ50 ---')
    for k, v in sorted(all_fixes.items(), key=lambda x: -x[1]):
        if v:
            print(f'  {k:22s}: {v}')

    out = ('/home/z/my-project/work/github_repo/eiqaz-tts/'
           'tashkeel-pipeline/state/det_eval50_report.json')
    json.dump({'tally': tally, 'fixed_by_layer':
               [[u, b, d] for u, b, d in fixed_by_layer],
               'broken_by_layer':
               [[u, b, d, m] for u, b, d, m in broken_by_layer],
               'all_fixes': all_fixes,
               'gate_skeleton_fail': gate_fail['skeleton']},
              open(out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f'\nحُفظ: {out}')
