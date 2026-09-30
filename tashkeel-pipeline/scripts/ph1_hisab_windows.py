# -*- coding: utf-8 -*-
"""Compact window listing for the حساب family (150 occurrences).

Produces one line per occurrence: utt + category + ~3-word window around
the matched token — for inline review of the trivially-secular bulk.
"""
import json

data = json.load(open(
    '/home/z/my-project/work/ph1_context_review.json', encoding='utf-8'))
occ = data['targets']['حساب']['occ_true']

CAT2LBL = {'train_pool': 'P', 'train_borderline': 'B', 'val': 'V'}

out = []
for e in occ:
    words = e['transcript'].split()
    # find token index (first token whose skeleton contains حساب)
    idx = None
    for i, w in enumerate(words):
        if 'حساب' in w or 'حسابات' in w:
            idx = i
            break
    if idx is None:
        idx = 0
    lo, hi = max(0, idx - 3), min(len(words), idx + 4)
    window = ' '.join(words[lo:hi])
    out.append(f"{CAT2LBL[e['cat']]} {e['utt']} | {window}")

print(f'# {len(out)} occurrences (P=train_pool V=val)')
for line in out:
    print(line)
