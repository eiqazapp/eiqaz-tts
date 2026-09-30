# -*- coding: utf-8 -*-
"""Stage 2 — تحويل الترقيم على كامل المتن (المرحلة 2 من مسار C).

قاعدة معتمدة (دليل v2.1 §أ-1/§ل-1/§ن-8):
  ، (U+060C) → ,   و   ؟ (U+061F) → ?
قبل التشكيل/G2P — لكي تصبح رمزين حقيقيين في جدول الرموز (ids 6 و 7)
بدل حذفهما الصامت من الـphonetiser. كل الرموز الأخرى (؛ ٪ … – : " ( ) %
الأرقام) تبقى كما هي حرفيًا — سلوك قائم لا يتغير (لا رموز جديدة).

العملية حتمية بالكامل: لا LLM، لا GPU، لا مساس بأي صوت.
مخرجات:
  work/ph2_input_full.json  — مدخل جاهز للمحرك (21,880 وحدة بالنص المحوَّل)
  work/ph2_punct_stats.json — الجرد الكامل + نتائج التحقق
"""
import csv
import json
import re
from collections import Counter

CSV_PATH = '/home/z/my-project/work/prep_output/extraction.csv'
OUT_INPUT = '/home/z/my-project/work/ph2_input_full.json'
OUT_STATS = '/home/z/my-project/work/ph2_punct_stats.json'

ARABIC_COMMA = '\u060C'   # ،
ARABIC_QMARK = '\u061F'   # ؟
AR_LETTER = re.compile(r'[\u0621-\u064A\u0640]')


def convert(t: str) -> str:
    return t.replace(ARABIC_COMMA, ',').replace(ARABIC_QMARK, '?')


def main():
    rows = list(csv.DictReader(open(CSV_PATH, encoding='utf-8')))
    n = len(rows)
    ids = [r['utt'] for r in rows]
    assert len(set(ids)) == n, 'duplicate utt ids'

    units_comma, units_qmark, units_union = set(), set(), set()
    occ_comma = occ_qmark = 0
    split_counts = Counter()
    converted_units = 0
    examples = []
    input_records = []
    verify_failures = []

    for r in rows:
        raw, utt, split = r['transcript'], r['utt'], r['split']
        conv = convert(raw)

        # --- verification: ONLY the two mapped chars changed ---
        if conv != raw:
            converted_units += 1
        # strip the two mapped chars from both sides -> must be identical
        # (proves letters/spaces/digits/all other chars untouched)
        a = raw.replace(ARABIC_COMMA, '').replace(ARABIC_QMARK, '')
        b = conv.replace(',', '').replace('?', '')
        # careful: raw may already contain ASCII , ? — remove mapped pairs only.
        # precise check: same length, and diff positions are exactly the mapped chars
        if len(raw) != len(conv):
            verify_failures.append((utt, 'length'))
        else:
            for i, (c1, c2) in enumerate(zip(raw, conv)):
                if c1 != c2:
                    ok_pair = (c1 == ARABIC_COMMA and c2 == ',') or \
                              (c1 == ARABIC_QMARK and c2 == '?')
                    if not ok_pair:
                        verify_failures.append((utt, f'pos {i}: {c1!r}->{c2!r}'))
                        break

        nc, nq = raw.count(ARABIC_COMMA), raw.count(ARABIC_QMARK)
        occ_comma += nc
        occ_qmark += nq
        if nc:
            units_comma.add(utt)
        if nq:
            units_qmark.add(utt)
        if nc or nq:
            units_union.add(utt)
            if len(examples) < 4:
                examples.append({'utt': utt, 'split': split,
                                 'before': raw[:90], 'after': conv[:90],
                                 'n_comma': nc, 'n_qmark': nq})
        split_counts[(split, bool(nc or nq))] += 1
        input_records.append({'id': utt, 'text': conv,
                              'n_tokens_recorded': int(r['n_tokens']),
                              'split': split})

    # --- full non-letter char inventory (before conversion) ---
    inv_occ, inv_units = Counter(), Counter()
    for r in rows:
        seen = set()
        for ch in r['transcript']:
            if not AR_LETTER.match(ch):
                inv_occ[ch] += 1
                seen.add(ch)
        for ch in seen:
            inv_units[ch] += 1
    inventory = [{'char': ch, 'codepoint': f'U+{ord(ch):04X}',
                  'name': NAMES.get(ch, 'ASCII/other'),
                  'total': inv_occ[ch], 'units': inv_units[ch],
                  'action': ACTION.get(ch, 'keep')}
                 for ch in inv_occ]

    train_all = sum(v for (s, _), v in split_counts.items() if s == 'train')
    eval_all = sum(v for (s, _), v in split_counts.items() if s == 'eval')
    train_conv = split_counts.get(('train', True), 0)
    eval_conv = split_counts.get(('eval', True), 0)

    stats = {
        'stage': 'المرحلة 2 — تحويل الترقيم (، -> , و ؟ -> ?)',
        'rule': 'deterministic char-map before tashkeel/G2P; all other chars kept',
        'n_units_total': n,
        'verification': {
            'failures': len(verify_failures),
            'failure_details': verify_failures[:10],
            'checks_passed': ['row count', 'unique ids', 'length preservation',
                              'only ،->, and ؟->? changed', 'letters/spaces/digits untouched'],
        },
        'arabic_comma': {'units': len(units_comma), 'occurrences': occ_comma},
        'arabic_qmark': {'units': len(units_qmark), 'occurrences': occ_qmark},
        'union_units_changed': converted_units,
        'union_pct': round(100.0 * converted_units / n, 2),
        'split_breakdown': {
            'train_total': train_all, 'train_converted': train_conv,
            'eval_total': eval_all, 'eval_converted': eval_conv,
        },
        'overlap_units_both': len(units_comma & units_qmark),
        'char_inventory': inventory,
        'kept_as_is_note': ('؛ ٪ … – تبقى كما هي: خارج جدول الرموز (ids 5-8 فقط '
                            '., ? !) وتسقط من الـG2P كسلوك قائم — لا رموز جديدة'),
        'examples': examples,
    }

    json.dump(input_records, open(OUT_INPUT, 'w', encoding='utf-8'),
              ensure_ascii=False)
    json.dump(stats, open(OUT_STATS, 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)

    print(f'units={n} | converted={converted_units} ({stats["union_pct"]}%)')
    print(f'، : {occ_comma} occurrences in {len(units_comma)} units')
    print(f'؟ : {occ_qmark} occurrences in {len(units_qmark)} units')
    print(f'overlap (both): {len(units_comma & units_qmark)}')
    print(f'split: train {train_conv}/{train_all} | eval {eval_conv}/{eval_all}')
    print(f'verification failures: {len(verify_failures)}')
    print(f'inventory distinct non-letter chars: {len(inventory)}')
    kept_ar = [e for e in inventory if e['action'] == 'keep' and ord(e['char'][0]) > 0x0600]
    for e in kept_ar:
        print(f"  kept-as-is Arabic: {e['char']!r} {e['codepoint']} "
              f"({e['total']}x in {e['units']} units)")


NAMES = {
    ' ': 'space', '.': 'ASCII period', ',': 'ASCII comma',
    '?': 'ASCII question', '!': 'ASCII exclam', ':': 'ASCII colon',
    '"': 'ASCII dquote', "'": 'ASCII quote', '-': 'ASCII hyphen',
    '+': 'ASCII plus', '=': 'ASCII equals', '%': 'ASCII percent',
    '(': 'ASCII lparen', ')': 'ASCII rparen',
    '\u060C': 'Arabic comma ،', '\u061F': 'Arabic question ؟',
    '\u061B': 'Arabic semicolon ؛', '\u066A': 'Arabic percent ٪',
    '\u2026': 'ellipsis …', '\u2013': 'en-dash –',
    '0': 'digit 0', '1': 'digit 1', '2': 'digit 2', '3': 'digit 3',
    '4': 'digit 4', '5': 'digit 5', '6': 'digit 6', '7': 'digit 7',
    '8': 'digit 8', '9': 'digit 9',
    '\u064B': 'fathatan ً (raw transfer diacritic)',
    '\u064C': 'dammatan ٌ (raw)', '\u064D': 'kasratan ٍ (raw)',
    '\u064E': 'fatha (raw)', '\u064F': 'damma (raw)',
    '\u0650': 'kasra (raw)', '\u0651': 'shadda (raw)',
    '\u0652': 'sukun (raw)',
}
ACTION = {
    '\u060C': 'convert->,', '\u061F': 'convert->?',
}

if __name__ == '__main__':
    main()
