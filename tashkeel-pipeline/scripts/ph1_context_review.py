# -*- coding: utf-8 -*-
"""Phase 1 — COMPLETE contextual review extraction.

Goal: give the user (native speaker) every actual context in which each
"context-confirmed exclusion" word (guide v2 §ك / ي-3) appears, so he can
verify the secular/religious classification himself before final approval.

Scope of matching is a deliberate SUPERSET of ph1_scan_all.py's religious
scan (which required ال for الحساب/النار and exact-equality variants):
  - bare forms (حساب، نار) in addition to definite/prefixed forms
  - orthographic variants (final ة↔ه، final ى↔ي، أ/إ/آ→ا)
  - attached pronoun suffixes and plurals (دينك، حسابات...)
  - proactive scans: قضاء (unclassified gap), قدر (القضاء والقدر pair),
    رب/ربنا (scope of the newly adopted رَبِّنَا rule), صلاة (count
    reconciliation 6→4), عقاب (4→3), عقبة (conditional), نبي* (verify 0)

Every corpus row is categorised:
  train_pool      = split train AND n_tokens<=160 AND n_frames<=950 (19,721)
  train_borderline= split train, filters violated (2 units)
  val             = split eval (2,157)

Also re-derives the 6 phonetic-safety quarantine units (raw spelling level)
with their IDs so the user-approved quarantine is enforceable.

Writes: work/ph1_context_review.json, work/ph1_quarantine.json
Prints: full console report (safe: vulgar lexicon itself never printed).
"""
import ast
import csv
import json
import re
from collections import Counter

TASH = re.compile(r'[\u064B-\u0652\u0653-\u0655\u0670]')
PUNCT = '.,?!:;"()«»\u060C\u061F\u061B\u201C\u201D\u2026-'

CSV_PATH = '/home/z/my-project/work/prep_output/extraction.csv'
OUT_JSON = '/home/z/my-project/work/ph1_context_review.json'
QUAR_JSON = '/home/z/my-project/work/ph1_quarantine.json'
SAFETY_SRC = '/home/z/my-project/scripts/ph1_safety.py'


def skel(w):
    return TASH.sub('', w).strip(PUNCT)


def norm(s):
    s = s.replace('أ', 'ا').replace('إ', 'ا').replace('آ', 'ا')
    if s.endswith('ة'):
        s = s[:-1] + 'ه'
    if s.endswith('ى'):
        s = s[:-1] + 'ي'
    return s


def cores(tok):
    """Prefix-stripped normalized lookup cores of a raw token."""
    s = norm(skel(tok))
    if not s:
        return []
    out = {s}
    if s[:1] in 'وف' and len(s) > 3:
        out.add(s[1:])
    for p in ('وال', 'فال', 'بال', 'كال', 'لل'):
        if s.startswith(p) and len(s) - len(p) >= 3:
            out.add(s[len(p):])
    if s.startswith('ال') and len(s) > 4:
        out.add(s[2:])
    for c in 'وفبلك':
        if s[:1] == c and len(s) >= 4:
            out.add(s[1:])
    return sorted(c for c in out if c)


# attached pronoun suffixes allowed after a true stem
SUFF = {'ه', 'ها', 'ك', 'كي', 'كم', 'هم', 'هن', 'نا', 'ي', 'يه', 'يها'}


def true_din(c):
    return c == 'دين' or c[3:] in SUFF


def true_hisab(c):
    return (c == 'حساب' or c[4:] in SUFF or c.startswith('حسابات')
            or c.startswith('حساباتا'))


def true_nar(c):
    return c == 'نار' or c[3:] in SUFF


def true_baraka(c):
    return c.startswith('بركه') or c.startswith('بركات')


def true_iqab(c):
    return c == 'عقاب' or c[4:] in SUFF


def true_uqba(c):
    return c.startswith('عقبه')


def true_wahy(c):
    return c == 'وحي' or c[3:] in SUFF


def true_akhira(c):
    # all اخره-containing tokens are candidates (few); manual classification
    return 'اخره' in c


def true_nabi(c):
    return 'نبي' in c


def true_qada(c):
    return c == 'قضاء' or c[5:] in SUFF


def true_qadr(c):
    # bare قدر / القدر (+ pronouns); قدرة/قدرات = ability (near-miss,
    # counted only) — contexts of bare forms decide secular vs decree
    return c == 'قدر' or c[3:] in SUFF


def true_rab(c):
    return c in {'رب', 'ربنا', 'ربي', 'الرب'} or c.startswith('ربنا')


def true_sala(c):
    return c.startswith('صلا')


TARGETS = [
    ('دين', 'startswith', 'دين', true_din),
    ('حساب', 'startswith', 'حساب', true_hisab),
    ('نار', 'startswith', 'نار', true_nar),
    ('بركة', 'startswith', 'برك', true_baraka),
    ('عقاب', 'startswith', 'عقاب', true_iqab),
    ('عقبة', 'startswith', 'عقب', true_uqba),
    ('وحي', 'startswith', 'وحي', true_wahy),
    ('آخرة/باخرة', 'contains', 'اخره', true_akhira),
    ('نبي*', 'contains', 'نبي', true_nabi),
    ('قضاء', 'startswith', 'قضاء', true_qada),
    ('قدر', 'startswith', 'قدر', true_qadr),
    ('رب/ربنا', 'startswith', 'رب', true_rab),
    ('صلاة', 'startswith', 'صلا', true_sala),
]


def main():
    rows = list(csv.DictReader(open(CSV_PATH, encoding='utf-8')))
    print(f'rows: {len(rows)}')

    def cat(r):
        if r['split'] == 'train':
            if int(r['n_tokens']) <= 160 and int(r['n_frames']) <= 950:
                return 'train_pool'
            return 'train_borderline'
        return 'val'

    cats = Counter(cat(r) for r in rows)
    print('categories:', dict(cats))

    results = {}
    for name, mode, key, is_true in TARGETS:
        occ_true, occ_near = [], []
        forms_true, forms_near = Counter(), Counter()
        for r in rows:
            for w in r['transcript'].split():
                cs = cores(w)
                if not cs:
                    continue
                hit = False
                for c in cs:
                    if mode == 'startswith' and c.startswith(key):
                        hit = True
                        break
                    if mode == 'contains' and key in c:
                        hit = True
                        break
                if not hit:
                    continue
                entry = {'utt': r['utt'], 'cat': cat(r), 'token': w,
                         'transcript': r['transcript']}
                if any(is_true(c) for c in cs):
                    occ_true.append(entry)
                    forms_true[norm(skel(w))] += 1
                else:
                    occ_near.append(entry)
                    forms_near[norm(skel(w))] += 1
        results[name] = {
            'mode': mode, 'key': key,
            'n_true': len(occ_true), 'n_near': len(occ_near),
            'n_true_by_cat': dict(Counter(e['cat'] for e in occ_true)),
            'forms_true': dict(forms_true.most_common()),
            'forms_near': dict(forms_near.most_common()),
            'occ_true': occ_true,
            'occ_near': occ_near,
        }
        print(f'\n== {name} ==  true={len(occ_true)}  '
              f'near-miss={len(occ_near)}  '
              f'by_cat={dict(Counter(e["cat"] for e in occ_true))}')
        print('  true forms:', dict(forms_true.most_common()))
        if forms_near:
            print('  near forms:', dict(forms_near.most_common()))
        for e in occ_true:
            print(f"  [{e['cat']}] {e['utt']} | {e['token']} | "
                  f"{e['transcript']}")
        if len(occ_near) <= 40:
            for e in occ_near:
                print(f"  NEAR [{e['cat']}] {e['utt']} | {e['token']} | "
                      f"{e['transcript']}")

    # ---- quarantine re-derivation (raw spelling level) ----
    src = open(SAFETY_SRC, encoding='utf-8').read()

    def extract_list(name):
        m = re.search(name + r'\s*=\s*\[(.*?)\]', src, re.S)
        return ast.literal_eval('[' + m.group(1) + ']')

    t1 = extract_list('TIER1')
    t2 = extract_list('TIER2')
    sk_bad = {TASH.sub('', w) for w in t1 + t2}
    quarantine = []
    for r in rows:
        hits = sorted({skel(w) for w in r['transcript'].split()
                       if skel(w) in sk_bad})
        if hits:
            quarantine.append({'utt': r['utt'], 'cat': cat(r),
                               'split': r['split'],
                               'n_tokens': int(r['n_tokens']),
                               'words': hits,
                               'transcript': r['transcript']})
    print(f'\n== quarantine re-derivation ==  units={len(quarantine)}')
    for q in quarantine:
        print(f"  [{q['cat']}] {q['utt']} words={q['words']} "
              f"n_tokens={q['n_tokens']}")
        print(f"    {q['transcript']}")

    json.dump({'criterion': 'superset token-level prefix-tolerant matching '
              '(bare+definite+prefixed forms, ة/ه ي/ى variants, pronoun '
              'suffixes, plurals); categories: train_pool=19721 / '
              'train_borderline=2 / val=2157',
               'targets': results,
               'category_counts': dict(cats)},
              open(OUT_JSON, 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    json.dump({'approved_by_user': '2026-09-30 (quarantine + no automated '
              'use)', 'units': quarantine},
              open(QUAR_JSON, 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print(f'\nDONE — saved {OUT_JSON} and {QUAR_JSON}')


if __name__ == '__main__':
    main()
