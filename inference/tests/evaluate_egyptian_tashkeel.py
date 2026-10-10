#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""تقييم تشكيل المصرية الداخلي — نسخة موسعة قابلة للمقارنة قبل/بعد.

يشغّل مسار الاستدلال الفعلي (infer.prepare_text_rich مع مرحلاته)، بلا
checkpoint صوتي ولا توليد WAV. لكل حالة يسجل:
  النص الأصلي · اللهجة · الوضع المطلوب والفعلي · النص بعد التطبيع ·
  بعد التشكيل الأولي (catt) · بعد القواعد المصرية (det) · الناتج النهائي ·
  الحركات المتوقعة/الفعلية في المواضع المستهدفة (حروف داخلية) ·
  الحركات المضافة/المتغيرة/المفقودة · نجاح/فشل مع السبب.

قياس مزدوج: مقابل المرجع اليدوي المبدئي (expected) وم مقابل الشكل
المهيمن في corpus المشكيل (corpus_reference) — فصلًا صريحًا بين
الفرضيات اللغوية والنتائج المقيسة على معيار المشروع.

الحالات السياقية (context_dependent=true) تُعرض وتُستثنى من المجاميع.

الاستخدام:
    python evaluate_egyptian_tashkeel.py [--out FILE] [--before FILE]
    [--csv FILE]
--before: مقارنة بتقرير سابق وكشف حالات التراجع حرفًا-بحرف.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
INF = os.path.dirname(HERE)
for path in (INF, os.path.join(INF, 'lib')):
    if path not in sys.path:
        sys.path.insert(0, path)

import infer  # noqa: E402

FIXTURE = os.path.join(HERE, 'egyptian_tashkeel_reference.json')
DIACS = set(chr(i) for i in range(0x064B, 0x0653))
ALEF_FOLD = str.maketrans({'\u0623': '\u0627', '\u0625': '\u0627',
                           '\u0622': '\u0627'})
ARABIC = lambda c: '\u0621' <= c <= '\u063A' or '\u0641' <= c <= '\u064A'


def letter_units(word):
    """[(base letter, sorted attached marks), ...] — بلا ترتيب الحركات."""
    chars = list(word)
    units = []
    i = 0
    while i < len(chars):
        ch = chars[i]
        if ARABIC(ch):
            marks = []
            j = i + 1
            while j < len(chars) and chars[j] in DIACS:
                marks.append(chars[j])
                j += 1
            units.append((ch, ''.join(sorted(marks))))
            i = j
        else:
            i += 1
    return units


def skel(word):
    return ''.join(c for c in word if ARABIC(c)).translate(ALEF_FOLD)


def compare_marks(exp, act):
    """تصنيف فرق حركات حرف واحد: (added, changed, lost)."""
    if exp == act:
        return None
    if not act:
        return ('', exp, 'lost') if exp else None
    if not exp:
        return (act, '', 'added')
    return (act, exp, 'changed')


def evaluate_case(case):
    res = infer.prepare_text_rich(
        case['input'], case.get('requested_mode', 'auto'),
        case.get('dialect', 'egy'))
    st = res.get('stages', {})
    actual_words = res['text'].split()
    input_words = case['input'].split()
    expected_words = case['expected'].split()
    row = {
        'id': case['id'],
        'input': case['input'],
        'input_type': case.get('input_type', 'unvocalized'),
        'dialect': case.get('dialect', 'egy'),
        'requested_mode': st.get('requested_mode'),
        'effective_mode': st.get('effective_mode', res['diacritize']),
        'normalized': st.get('normalized', res['normalized']),
        'after_catt': st.get('after_catt'),
        'after_det_or_restore': st.get('after_det', st.get('after_restore')),
        'after_merge': st.get('after_merge'),
        'final': res['text'],
        'expected': case['expected'],
        'context_dependent': bool(case.get('context_dependent')),
        'expected_semantics': case.get('expected_semantics'),
        'skeleton_match': len(actual_words) == len(expected_words),
        'focus_words': [],
    }
    stats = dict(internal_correct=0, internal_total=0, exact=0,
                 n_focus=0, corpus_correct=0, corpus_total=0,
                 letters_lost=0, letters_changed=0, letters_added=0)
    if not row['skeleton_match']:
        row['error'] = 'word_count_mismatch'
        row['fail_reason'] = f'عدد الكلمات: متوقع {len(expected_words)} ' \
                             f'فعلي {len(actual_words)}'
        return row, stats
    corpus_ref = case.get('corpus_reference') or {}
    for idx in case['focus_word_indices']:
        got = letter_units(actual_words[idx])
        want = letter_units(expected_words[idx])
        in_word = input_words[idx] if idx < len(input_words) else ''
        ckey = skel(in_word)
        item = {
            'word_index': idx,
            'input_word': in_word,
            'actual_word': actual_words[idx],
            'expected_word': expected_words[idx],
            'skeleton_match': [x[0] for x in got] == [x[0] for x in want],
        }
        if not item['skeleton_match']:
            item['error'] = 'letter_skeleton_mismatch'
            row['focus_words'].append(item)
            continue
        diffs = []
        n = len(want)
        correct = 0
        for i in range(1, n - 1):          # الحروف الداخلية فقط
            exp_m, act_m = want[i][1], got[i][1]
            stats['internal_total'] += 1
            d = compare_marks(exp_m, act_m)
            if d is None:
                correct += 1
                stats['internal_correct'] += 1
            else:
                act_e, exp_e, kind = d
                diffs.append({
                    'pos': i, 'letter': want[i][0],
                    'expected': exp_e or '(بلا حركة)',
                    'actual': act_e or '(بلا حركة)', 'kind': kind})
                stats['letters_' + kind] += 1
        item['internal_letters'] = max(0, n - 2)
        item['internal_correct'] = correct
        item['internal_accuracy'] = round(correct / (n - 2), 4) if n > 2 \
            else None
        item['mark_diffs'] = diffs
        item['exact_word_match'] = got == want
        stats['exact'] += int(got == want)
        stats['n_focus'] += 1
        # القياس الثاني: الشكل المهيمن في corpus (إن وُجد)
        cref = corpus_ref.get(ckey)
        if cref and n > 2:
            cu = letter_units(cref['form'])
            if len(cu) == len(got) and [x[0] for x in cu] == [x[0]
                                                              for x in got]:
                cc = sum(cu[i][1] == got[i][1]
                         for i in range(1, n - 1))
                item['corpus_form'] = cref['form']
                item['corpus_agrees'] = cc == (n - 2)
                if cc == (n - 2):
                    stats['corpus_correct'] += 1
                stats['corpus_total'] += 1
        row['focus_words'].append(item)
    ok = (stats['internal_total'] == 0
          or stats['internal_correct'] == stats['internal_total'])
    row['pass'] = ok
    if not ok:
        row['fail_reason'] = 'حركات داخلية مخالفة للمرجع: ' + '; '.join(
            f"{d['letter']}@{d['pos']} {d['kind']} "
            f"({d['expected']}→{d['actual']})"
            for fw in row['focus_words']
            for d in fw.get('mark_diffs', []))
    return row, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default=os.path.join(
        HERE, 'egyptian_tashkeel_report.json'))
    parser.add_argument('--csv', default=os.path.join(
        HERE, 'egyptian_tashkeel_report.csv'))
    parser.add_argument('--before', default=None,
                        help='تقرير سابق لكشف حالات التراجع')
    args = parser.parse_args()
    with open(FIXTURE, encoding='utf-8') as f:
        fixture = json.load(f)

    rows = []
    agg = dict(internal_correct=0, internal_total=0, exact=0, n_focus=0,
               corpus_correct=0, corpus_total=0, letters_lost=0,
               letters_changed=0, letters_added=0)
    n_ctx = 0
    for case in fixture['cases']:
        row, stats = evaluate_case(case)
        rows.append(row)
        if row.get('context_dependent'):
            n_ctx += 1
            print(f"[CTX] {row['id']} {row['input']} → {row['final']}")
            continue
        for k in agg:
            agg[k] += stats[k]
        mark = 'PASS' if row.get('pass') else 'FAIL'
        print(f"[{mark}] {row['id']} {row['input']} → {row['final']} "
              f"(داخلي {stats['internal_correct']}/{stats['internal_total']})")

    evaluated = [r for r in rows if not r.get('context_dependent')]
    n_pass = sum(1 for r in evaluated if r.get('pass'))
    n_fail = len(evaluated) - n_pass
    report = {
        'fixture_status': fixture['status'],
        'warning': fixture['warning'],
        'n_cases': len(rows),
        'n_evaluated': len(evaluated),
        'n_context_dependent_excluded': n_ctx,
        'n_pass': n_pass,
        'n_fail': n_fail,
        'internal_diacritic_accuracy': round(
            agg['internal_correct'] / agg['internal_total'], 4)
        if agg['internal_total'] else None,
        'internal_letters_correct': agg['internal_correct'],
        'internal_letters_total': agg['internal_total'],
        'corpus_agreement_accuracy': round(
            agg['corpus_correct'] / agg['corpus_total'], 4)
        if agg['corpus_total'] else None,
        'corpus_focus_words': agg['corpus_total'],
        'exact_focus_word_accuracy': round(
            agg['exact'] / agg['n_focus'], 4) if agg['n_focus'] else None,
        'marks_lost': agg['letters_lost'],
        'marks_changed': agg['letters_changed'],
        'marks_added': agg['letters_added'],
        'cases': rows,
    }

    # مقارنة قبل/بعد وكشف التراجع — مفتاح الربط (id, word_index)
    # متوافق مع تنسيق التقرير القديم والحديث معًا
    regressions = []
    if args.before and os.path.exists(args.before):
        with open(args.before, encoding='utf-8') as f:
            before = json.load(f)
        bmap = {}
        for c in before.get('cases', []):
            for fw in c.get('focus_words', []):
                k = (c['id'], fw.get('word_index'))
                bmap[k] = fw.get('internal_correct', 0)
        n_compared = 0
        for r in rows:
            for fw in r.get('focus_words', []):
                k = (r['id'], fw.get('word_index'))
                if k not in bmap:
                    continue
                n_compared += 1
                if (fw.get('internal_correct', 0) is not None
                        and fw['internal_correct'] < bmap[k]):
                    regressions.append({
                        'case': r['id'], 'word': fw.get('input_word'),
                        'word_index': fw.get('word_index'),
                        'before': bmap[k],
                        'after': fw['internal_correct']})
        report['before_file'] = args.before
        report['n_focus_words_compared_vs_before'] = n_compared
        report['regressions_vs_before'] = regressions
        report['n_regressions'] = len(regressions)

    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    # CSV: صف لكل كلمة مستهدفة
    with open(args.csv, 'w', encoding='utf-8') as f:
        f.write('case,input_type,effective_mode,word_index,input_word,'
                'expected_word,actual_word,internal_correct,'
                'internal_total,exact_word,mark_diffs\n')
        for r in rows:
            for fw in r.get('focus_words', []):
                diffs = '|'.join(
                    f"{d['letter']}@{d['pos']}:{d['kind']}"
                    for d in fw.get('mark_diffs', []))
                f.write('{},{},{},{},{},{},{},{},{},{},{}\n'.format(
                    r['id'], r.get('input_type'), r.get('effective_mode'),
                    fw.get('word_index'), fw.get('input_word'),
                    fw.get('expected_word'), fw.get('actual_word'),
                    fw.get('internal_correct', 0),
                    fw.get('internal_letters', 0),
                    int(bool(fw.get('exact_word_match'))), diffs))

    print()
    print(json.dumps({k: report[k] for k in (
        'fixture_status', 'n_cases', 'n_evaluated',
        'n_context_dependent_excluded', 'n_pass', 'n_fail',
        'internal_diacritic_accuracy', 'corpus_agreement_accuracy',
        'exact_focus_word_accuracy', 'marks_lost', 'marks_changed',
        'marks_added') + (('n_regressions',) if args.before else ())
        if k in report}, ensure_ascii=False, indent=2))
    if regressions:
        print('تحذير — تراجعات مقارنة بالنسخة الأساسية:')
        for rg in regressions:
            print(f"  {rg['case']} / {rg['word']}: "
                  f"{rg['before']} → {rg['after']}")
    print(f'Report: {args.out}')
    print(f'CSV:    {args.csv}')
    # المقيّم تشخيصي لا بوابة: رمز الخروج 1 فقط عند الانهيار أو التراجع
    sys.exit(1 if regressions else 0)


if __name__ == '__main__':
    main()
