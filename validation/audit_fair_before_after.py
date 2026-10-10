#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""audit_fair_before_after.py — مقارنة عادلة قبل/بعد على المجموعة الثابتة
(تصحيح تدقيق 2026-10-10 للمقارنة بأقسام مختلفة في رسالة db8f6b1).

المشكلة المُصحَّحة: رسالة الالتزام db8f6b1 قارنت
  قبل: 17/29 (58.62%) — تقرير الأساس (8 حالات)
  بعد: 52/57 (91.23%) — المرجع الموسع (19 حالة)
المقامات مختلفان (29 مقابل 57 موضعًا داخليًا) فلا يثبت التقدم بنفسه.

هذا السكربت يقيس المقارنة العادلة:
  1) نفس الأمثلة: حالات الأساس الثمانية فقط (موجودة حرفيًا في المرجع).
  2) نفس المواضع المستهدفة: focus_word_indices متطابقة بين الملفين.
  3) نفس المرجع: expected متطابق حرفيًا.
  4) طريقة عدّ ثابتة: الحروف الداخلية فقط (باستثناء الأول والأخير)
     للكلمة المستهدفة، مطابقة حركة-بحركة، ويعاد عدّ جانب «قبل» من
     كلمات التقرير القديم نفسها بالطريقة نفسها.
  5) فصل الحركات الداخلية (internal) عن التطابق الكامل للكلمة (exact).
  6) فصل المرجع اليدوي (expected) عن مرجع corpus (قياس ثانٍ).
  7) الحالات السياقية لا تُخلط (لا توجد في مجموعة الأساس أصلًا).
  8) لا استبعاد صامت — كل فشل معروض بموضعه وحرفه.
  9) عدد المواضع القابلة للمقارنة محسوب في الجانبين (29 = 29).
  10) كل تراجع/تحسن معروض على مستوى الكلمة والموضع.

الاستخدام (قابل لإعادة التشغيل — يحتاج تشغيل المسار الحي):
    python validation/audit_fair_before_after.py
المخرجات: validation/audit_fair_report.json + .csv
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
INF = os.path.join(REPO, 'inference')
for p in (INF, os.path.join(INF, 'lib'), os.path.join(INF, 'tests')):
    if p not in sys.path:
        sys.path.insert(0, p)

import infer  # noqa: E402

BASELINE = os.path.join(INF, 'tests', 'egyptian_tashkeel_baseline_before_fix.json')
REFERENCE = os.path.join(INF, 'tests', 'egyptian_tashkeel_reference.json')
OUT_JSON = os.path.join(HERE, 'audit_fair_report.json')
OUT_CSV = os.path.join(HERE, 'audit_fair_report.csv')

DIACS = set(chr(i) for i in range(0x064B, 0x0653))
ARABIC = lambda c: '\u0621' <= c <= '\u063A' or '\u0641' <= c <= '\u064A'


def letter_units(word):
    units, i = [], 0
    chars = list(word)
    while i < len(chars):
        if ARABIC(chars[i]):
            marks = []
            j = i + 1
            while j < len(chars) and chars[j] in DIACS:
                marks.append(chars[j])
                j += 1
            units.append((chars[i], ''.join(marks)))
            i = j
        else:
            i += 1
    return units


def internal_positions(word):
    """[(pos, letter, marks)] للحروف الداخلية فقط (بلا الأول والأخير)."""
    u = letter_units(word)
    return [(i, u[i][0], u[i][1]) for i in range(1, len(u) - 1)]


def main():
    with open(BASELINE, encoding='utf-8') as f:
        baseline = json.load(f)
    with open(REFERENCE, encoding='utf-8') as f:
        reference = json.load(f)
    refmap = {c['id']: c for c in reference['cases']}

    # فحص ثبات المجموعة (نفس الأمثلة/المرجع/المواضع)
    unstable = {}
    for c in baseline['cases']:
        rc = refmap.get(c['id'])
        unstable[c['id']] = not (
            rc is not None
            and rc['input'] == c['input']
            and rc['expected'] == c['expected']
            and rc['focus_word_indices'] == [
                fw.get('word_index') for fw in c.get('focus_words', [])])
    if any(unstable.values()):
        print('[تحذير] حالات أساس غير مستقرة مقابل المرجع الحالي — '
              'المقارنة غير صالحة لهذه الحالات')

    b_map = {}
    for c in baseline['cases']:
        for fw in c.get('focus_words', []):
            b_map[(c['id'], fw['word_index'])] = fw

    rows = []
    agg = dict(before_correct=0, after_correct=0, n_positions=0,
               improved=0, regressed=0, unchanged=0,
               before_exact=0, after_exact=0, n_words=0)
    for c in baseline['cases']:
        cid = c['id']
        rc = refmap[cid]
        res = infer.prepare_text_rich(
            rc['input'], rc.get('requested_mode', 'auto'),
            rc.get('dialect', 'egy'))
        actual_words = res['text'].split()
        expected_words = rc['expected'].split()
        for idx in rc['focus_word_indices']:
            bfw = b_map.get((cid, idx))
            if bfw is None:
                continue
            want = expected_words[idx]
            got = actual_words[idx] if idx < len(actual_words) else ''
            w_pos = internal_positions(want)
            b_word = bfw.get('actual_word', '')
            b_pos = internal_positions(b_word)
            g_pos = internal_positions(got)
            b_correct = sum(1 for i, (bp, wp) in enumerate(zip(b_pos, w_pos))
                            if bp[2] == wp[2])
            a_correct = sum(1 for i, (ap, wp) in enumerate(zip(g_pos, w_pos))
                            if ap[2] == wp[2])
            det = []
            for i in range(len(w_pos)):
                pos, letter, wm = w_pos[i]
                bm = b_pos[i][2] if i < len(b_pos) else None
                gm = g_pos[i][2] if i < len(g_pos) else None
                det.append({
                    'pos': pos, 'letter': letter, 'expected': wm,
                    'before': bm, 'after': gm,
                    'before_ok': bm == wm, 'after_ok': gm == wm,
                    'delta': ('improved' if (not (bm == wm)) and (gm == wm)
                              else 'regressed' if (bm == wm)
                              and not (gm == wm) else 'same')})
            agg['n_positions'] += len(w_pos)
            agg['before_correct'] += b_correct
            agg['after_correct'] += a_correct
            agg['improved'] += sum(1 for d in det if d['delta'] == 'improved')
            agg['regressed'] += sum(
                1 for d in det if d['delta'] == 'regressed')
            agg['unchanged'] += sum(1 for d in det if d['delta'] == 'same')
            agg['n_words'] += 1
            agg['before_exact'] += int(letter_units(b_word) ==
                                       letter_units(want))
            agg['after_exact'] += int(letter_units(got) ==
                                      letter_units(want))
            rows.append({
                'case': cid, 'word_index': idx,
                'input_word': rc['input'].split()[idx],
                'expected_word': want, 'before_word': b_word,
                'after_word': got, 'n_comparable_positions': len(w_pos),
                'before_internal_correct': b_correct,
                'after_internal_correct': a_correct,
                'before_exact_word': letter_units(b_word) ==
                letter_units(want),
                'after_exact_word': letter_units(got) == letter_units(want),
                'positions': det})

    b_reported = sum(fw.get('internal_correct', 0)
                     for c in baseline['cases']
                     for fw in c.get('focus_words', []))
    b_reported_total = sum(fw.get('internal_letters', 0)
                           for c in baseline['cases']
                           for fw in c.get('focus_words', []))
    summary = {
        'purpose': 'fair fixed-set comparison correcting the mixed-'
                   'denominator claim (17/29 vs 52/57) in commit db8f6b1',
        'method': 'same 8 baseline cases, same focus positions, same '
                  'expected reference, same internal-letter counting '
                  '(first/last excluded); before side recounted from the '
                  'old report own words',
        'n_cases': len(baseline['cases']),
        'n_focus_words': agg['n_words'],
        'n_comparable_positions': agg['n_positions'],
        'before_reported_by_old_report': f'{b_reported}/{b_reported_total}',
        'before_recounted': f"{agg['before_correct']}/{agg['n_positions']}",
        'after_live_run': f"{agg['after_correct']}/{agg['n_positions']}",
        'before_pct': round(100 * agg['before_correct'] /
                            agg['n_positions'], 2),
        'after_pct': round(100 * agg['after_correct'] /
                           agg['n_positions'], 2),
        'positions_improved': agg['improved'],
        'positions_regressed': agg['regressed'],
        'positions_unchanged': agg['unchanged'],
        'exact_word_before': agg['before_exact'],
        'exact_word_after': agg['after_exact'],
        'expanded_set_note': 'the 91.23% (52/57) figure remains valid but '
                             'ONLY on the expanded 19-case fixture — it is '
                             'not comparable to the 8-case baseline',
        'no_silent_exclusions': 'كل حالات الأساس الثماني معروضة؛ لا '
                                'حالات سياقية في مجموعة الأساس',
        'rows': rows,
    }
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    with open(OUT_CSV, 'w', encoding='utf-8') as f:
        f.write('case,word_index,input_word,expected_word,before_word,'
                'after_word,before_correct,after_correct,n_positions,'
                'improved,regressed,before_exact,after_exact\n')
        for r in rows:
            f.write('{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(
                r['case'], r['word_index'], r['input_word'],
                r['expected_word'], r['before_word'], r['after_word'],
                r['before_internal_correct'], r['after_internal_correct'],
                r['n_comparable_positions'],
                sum(1 for d in r['positions'] if d['delta'] == 'improved'),
                sum(1 for d in r['positions'] if d['delta'] == 'regressed'),
                int(r['before_exact_word']), int(r['after_exact_word'])))

    hdr = {k: v for k, v in summary.items() if k != 'rows'}
    print(json.dumps(hdr, ensure_ascii=False, indent=2))
    print(f'JSON: {OUT_JSON}\nCSV:  {OUT_CSV}')


if __name__ == '__main__':
    main()
