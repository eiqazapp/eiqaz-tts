# -*- coding: utf-8 -*-
"""Phase 1 — Tier C singleton context verification.

The guide's Tier C list includes several words with only 1-2 corpus hits.
This quick check prints every actual context so the review is complete:
آية، الرب، الإله، الحلال، الحرام، الأذان، جهنم، قرآن، الشيطان، الجنة،
العمرة، المسجد، السورة، الدعاء، الذنب، الثواب.
"""
import csv
import re
from collections import Counter

TASH = re.compile(r'[\u064B-\u0652\u0653-\u0655\u0670]')
PUNCT = '.,?!:;"()«»\u060C\u061F\u061B\u201C\u201D\u2026-'

WORDS = ['آية', 'اية', 'الرب', 'الإله', 'اله', 'الحلال', 'حلال',
         'الحرام', 'حرام', 'الأذان', 'اذان', 'جهنم', 'قرآن', 'قران',
         'الشيطان', 'شيطان', 'الجنة', 'جنة', 'العمرة', 'عمره', 'عمرة',
         'المسجد', 'مسجد', 'السورة', 'سورة', 'الدعاء', 'دعاء', 'الذنب',
         'ذنب', 'الثواب', 'ثواب']


def skel(w):
    return TASH.sub('', w).strip(PUNCT)


def cores(tok):
    s = skel(tok)
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
    return out


rows = list(csv.DictReader(open(
    '/home/z/my-project/work/prep_output/extraction.csv', encoding='utf-8')))


def cat(r):
    if r['split'] == 'train':
        if int(r['n_tokens']) <= 160 and int(r['n_frames']) <= 950:
            return 'train_pool'
        return 'train_borderline'
    return 'val'


hits = {}
for r in rows:
    for w in r['transcript'].split():
        for c in cores(w):
            if c in WORDS:
                hits.setdefault(c, []).append((cat(r), r['utt'], w,
                                               r['transcript']))
                break

for word in WORDS:
    if word in hits:
        print(f'\n== {word} == {len(hits[word])}')
        for c, u, w, t in hits[word]:
            print(f'  [{c}] {u} | {w} | {t}')
