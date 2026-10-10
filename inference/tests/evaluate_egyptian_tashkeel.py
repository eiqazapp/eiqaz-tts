#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""تقييم تشخيصي لتشكيل المصرية الداخلي على مجموعة مرجعية أولية.

يشغّل مسار الاستدلال الفعلي، لكنه لا يحمّل checkpoint الصوتي ولا يولّد WAV.
المرجع الحالي مبدئي وغير معتمد حتى يراجعه متحدث مصري مختص.
"""
import argparse
import json
import os
import sys
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
INF = os.path.dirname(os.path.dirname(HERE))
for path in (INF, os.path.join(INF, 'lib')):
    if path not in sys.path:
        sys.path.insert(0, path)

import infer  # noqa: E402

FIXTURE = os.path.join(HERE, 'egyptian_tashkeel_reference.json')
DIACS = set(chr(i) for i in range(0x064B, 0x0653))
ARABIC = lambda c: '\u0621' <= c <= '\u063A' or '\u0641' <= c <= '\u064A'


def letter_units(word):
    """Return [(base letter, canonical attached marks), ...]."""
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


def evaluate_case(case):
    result = infer.prepare_text_rich(case['input'], 'auto', 'egy')
    actual_words = result['text'].split()
    expected_words = case['expected'].split()
    row = {
        'id': case['id'],
        'input': case['input'],
        'actual': result['text'],
        'expected': case['expected'],
        'skeleton_match': len(actual_words) == len(expected_words),
        'focus_words': [],
    }
    internal_total = internal_correct = 0
    exact_words = total_focus_words = 0
    if len(actual_words) != len(expected_words):
        row['error'] = 'word_count_mismatch'
        return row, 0, 0, 0, 0
    for idx in case['focus_word_indices']:
        got = letter_units(actual_words[idx])
        want = letter_units(expected_words[idx])
        item = {'word_index': idx, 'actual_word': actual_words[idx],
                'expected_word': expected_words[idx],
                'skeleton_match': [x[0] for x in got] == [x[0] for x in want]}
        if not item['skeleton_match']:
            item['error'] = 'letter_skeleton_mismatch'
            row['focus_words'].append(item)
            continue
        # Count only internal letters; initial/final behavior is evaluated elsewhere.
        n = max(0, len(want) - 2)
        correct = sum(got[i][1] == want[i][1] for i in range(1, len(want) - 1))
        item['internal_letters'] = n
        item['internal_correct'] = correct
        item['internal_accuracy'] = round(correct / n, 4) if n else None
        item['exact_word_match'] = got == want
        internal_total += n
        internal_correct += correct
        exact_words += int(got == want)
        total_focus_words += 1
        row['focus_words'].append(item)
    row['internal_accuracy'] = (round(internal_correct / internal_total, 4)
                                if internal_total else None)
    return row, internal_correct, internal_total, exact_words, total_focus_words


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default=os.path.join(HERE, 'egyptian_tashkeel_baseline.json'))
    args = parser.parse_args()
    with open(FIXTURE, encoding='utf-8') as f:
        fixture = json.load(f)
    rows = []
    correct = total = exact = n_focus = 0
    for case in fixture['cases']:
        row, c, n, e, nf = evaluate_case(case)
        rows.append(row)
        correct += c
        total += n
        exact += e
        n_focus += nf
        print('[{}] {} | internal={}/{} | {}'.format(
            row['id'], row['input'], c, n, row['actual']))
    report = {
        'fixture_status': fixture['status'],
        'warning': fixture['warning'],
        'n_cases': len(rows),
        'internal_diacritic_accuracy': round(correct / total, 4) if total else None,
        'exact_focus_word_accuracy': round(exact / n_focus, 4) if n_focus else None,
        'internal_letters_correct': correct,
        'internal_letters_total': total,
        'focus_words_exact': exact,
        'focus_words_total': n_focus,
        'cases': rows,
    }
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: report[k] for k in (
        'fixture_status', 'n_cases', 'internal_diacritic_accuracy',
        'exact_focus_word_accuracy', 'internal_letters_correct',
        'internal_letters_total', 'focus_words_exact', 'focus_words_total')},
        ensure_ascii=False, indent=2))
    print('Report:', args.out)
    # A baseline is diagnostic, not a pass/fail gate before the gold is reviewed.


if __name__ == '__main__':
    main()
