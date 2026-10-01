# -*- coding: utf-8 -*-
"""det_smoke14.py — اختبار دخان الطبقة الحتمية على عينة الـ14 وحدة.

التسلسل: خام → catt_eo (حي) → الطبقة الحتمية → مقارنة ثنائية:
  1) بوابات §ل (هيكل/تنوين/ة/كثافة) قبل الطبقة مقابل بعدها — القيمة
     الميكانيكية: هل صار مخرج catt متوافقًا مع أنبوب الجودة؟
  2) التطابق كلمة-بكلمة (كل الحركات) مع مخرجات v5.3 المعتمدة — جودة
     مقابل المعيار المرجعي؛ متوقع أن يكون جزئيًا (تحسين جزئي معلن).

الوحدة المحجورة 0_04186_00 (قرار المستخدم 2026-09-30): تُعامل محجورة —
لا مقارنة آلية لها، وتُذكر في التقرير فقط.
"""
import json
import sys
import re

sys.path.insert(0, '/home/z/my-project/work/github_repo/eiqaz-tts/inference/lib')
sys.path.insert(0, '/home/z/my-project/scripts')
sys.path.insert(0, '/home/z/my-project/work/tts_arabic_pkg')

from det_tashkeel import DetTashkeel, DIACS, TANWEEN, FATHATAN
from tts_arabic.vocalizer.models.core import vocalize
from ph1_safety import check_vocalized_word

STATE = ('/home/z/my-project/work/github_repo/eiqaz-tts/'
         'tashkeel-pipeline/state')
QUARANTINED = {'0_04186_00'}
STRIP_D = re.compile('[' + ''.join(DIACS) + ']')


def gates(out, raw):
    """بوابات §ل: هيكل/تنوين/ة — نفس منطق المحرك."""
    conv = raw.replace('\u060C', ',').replace('\u061F', '?')
    # المحرك v5.3 يجرد الحركات من الطرفين — الخام نفسه قد يحمل تنوينًا
    conv = STRIP_D.sub('', conv)
    sk = STRIP_D.sub('', out) == conv
    tan_ok, taa_ok = True, True
    for w in out.split():
        bare = STRIP_D.sub('', w).strip('.,?!:;"()«»\u201C\u201D…-')
        kf = bare.translate(str.maketrans({'\u0623': '\u0627',
                                           '\u0625': '\u0627',
                                           '\u0622': '\u0627'}))
        for d in w:
            if d in TANWEEN and d != FATHATAN:
                tan_ok = False
            if d == FATHATAN and kf not in CLOSED:
                ks = kf[1:] if kf[:1] in 'وف' else kf
                if ks not in CLOSED and kf != 'خيرا':
                    tan_ok = False
        if w.endswith('\u0629') and w[:-1].endswith(tuple(DIACS)) is False \
                and re.search(r'\u0629[' + ''.join(DIACS) + r']+$', w):
            taa_ok = False
    words = [w for w in out.split() if any('\u0621' <= c <= '\u064A'
                                           for c in w)]
    dens = (sum(1 for c in out if c in DIACS) / len(words)
            if words else 0.0)
    return {'skeleton': sk, 'tanween': tan_ok, 'taa': taa_ok,
            'density': round(dens, 2)}


def cmp_words(a, b):
    """تصنيف زوج كلمات: تطابق تام / نهاية فقط / داخلية / هيكل."""
    if a == b:
        return 'exact'
    sa, sb = STRIP_D.sub('', a), STRIP_D.sub('', b)
    if sa != sb:
        return 'skeleton'
    da = [c for c in a if c in DIACS]
    db = [c for c in b if c in DIACS]
    # مقارنة موقعية للحركات على الحروف المشتركة
    ua = [(l, ''.join(d for d in x if d in DIACS))
          for l, x in _units(a)]
    ub = [(l, ''.join(d for d in x if d in DIACS))
          for l, x in _units(b)]
    if len(ua) != len(ub):
        return 'skeleton'
    diffs = [i for i, (x, y) in enumerate(zip(ua, ub)) if x != y]
    if not diffs:
        return 'exact'
    if diffs == [len(ua) - 1]:
        return 'final_only'
    return 'internal'


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


if __name__ == '__main__':
    dt = DetTashkeel()
    CLOSED = dt.closed_set | {'خيرا'}
    items = json.load(open(f'{STATE}/ph3_smoke2_input.json'))
    approved = {}
    for line in open(f'{STATE}/ph3_smoke2_v53_output.jsonl'):
        r = json.loads(line)
        approved[r['id']] = r

    results, tally = [], {'exact': 0, 'final_only': 0, 'internal': 0,
                          'skeleton': 0, 'total': 0}
    g_before = {'skeleton': 0, 'tanween': 0, 'taa': 0, 'density': 0, 'n': 0}
    g_after = {'skeleton': 0, 'tanween': 0, 'taa': 0, 'density': 0, 'n': 0}
    all_fixes = {}
    cat_examples = {'final_only': [], 'internal': []}

    for it in items:
        uid, raw = it['id'], it['text']
        if uid in QUARANTINED:
            results.append({'id': uid, 'status': 'quarantined'})
            continue
        catt = vocalize(raw, model='catt_eo')
        r = dt.process(raw, catt, safety_fn=check_vocalized_word)
        gb, ga = gates(catt, raw), gates(r['out'], raw)
        for g, x in ((g_before, gb), (g_after, ga)):
            g['n'] += 1
            for k in ('skeleton', 'tanween', 'taa'):
                g[k] += int(x[k])
            g['density'] += int(x['density'] >= 1.2)
        for k, v in r['fixes'].items():
            all_fixes[k] = all_fixes.get(k, 0) + v
        appr = approved.get(uid, {}).get('out')
        wc = {'exact': 0, 'final_only': 0, 'internal': 0, 'skeleton': 0}
        if appr:
            dw, aw = r['out'].split(), appr.split()
            if len(dw) == len(aw):
                for a, b in zip(dw, aw):
                    c = cmp_words(a, b)
                    wc[c] += 1
                    tally[c] += 1
                    tally['total'] += 1
                    if c in cat_examples and len(cat_examples[c]) < 6:
                        cat_examples[c].append((uid, a, b))
            else:
                wc['skeleton'] += abs(len(dw) - len(aw))
        results.append({'id': uid, 'status': 'ok', 'raw': raw,
                        'catt': catt, 'det': r['out'],
                        'approved': appr, 'gates_before': gb,
                        'gates_after': ga, 'word_cmp': wc,
                        'fixes': r['fixes'], 'flags': r['flags']})

    n_ok = sum(1 for x in results if x.get('status') == 'ok')
    print(f'=== دخان الطبقة الحتمية: {n_ok} وحدة (+1 محجورة) ===\n')
    print('--- البوابات: قبل الطبقة ← بعدها (القيمة الميكانيكية) ---')
    for k in ('skeleton', 'tanween', 'taa', 'density'):
        b = g_before[k] / g_before['n'] * 100
        a = g_after[k] / g_after['n'] * 100
        print(f'  {k:9s}: {g_before[k]}/{g_before["n"]} ({b:.0f}%) '
              f'← {g_after[k]}/{g_after["n"]} ({a:.0f}%)')
    print('\n--- إجمالي الإصلاحات الميكانيكية ---')
    for k, v in sorted(all_fixes.items(), key=lambda x: -x[1]):
        if v:
            print(f'  {k:22s}: {v}')
    print('\n--- التطابق مع v5.3 المعتمد (كلمة-بكلمة، كل الحركات) ---')
    t = tally['total']
    for k in ('exact', 'final_only', 'internal', 'skeleton'):
        print(f'  {k:11s}: {tally[k]:3d}/{t} ({tally[k] / t * 100:.1f}%)')
    print(f'  → التطابق التام الكامل: {tally["exact"] / t * 100:.1f}%')
    print(f'  → بعد تسامح النهايات فقط: '
          f'{(tally["exact"] + tally["final_only"]) / t * 100:.1f}%')
    print('\n--- أمثلة فروق النهايات فقط (الطبقة ← المعتمد) ---')
    for uid, a, b in cat_examples['final_only'][:5]:
        print(f'  {uid}: {a} ← {b}')
    print('\n--- أمثلة الفروق الداخلية (الطبقة ← المعتمد) ---')
    for uid, a, b in cat_examples['internal'][:6]:
        print(f'  {uid}: {a} ← {b}')

    out_path = ('/home/z/my-project/work/github_repo/eiqaz-tts/'
                'tashkeel-pipeline/state/det_smoke14_report.json')
    json.dump({'results': results, 'gates_before': g_before,
               'gates_after': g_after, 'word_tally': tally,
               'all_fixes': all_fixes},
              open(out_path, 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print(f'\nحُفظ: {out_path}')
