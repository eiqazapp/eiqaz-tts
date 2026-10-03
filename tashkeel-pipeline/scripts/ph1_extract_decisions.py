# -*- coding: utf-8 -*-
"""Phase 1 — empirical extraction of my tashkeel decisions from the 50-sample.

For every word (space-aligned, skeleton-matched) record the vocalized form.
Aggregate: skeleton -> {vocalization: count}. Flag skeletons with >1 form
(inconsistencies to pin in the fixed guide).
Also: tanween inventory + word-final-pattern inventory.
"""
import json
import re
import unicodedata
from collections import Counter, defaultdict

TASHKEEL = re.compile(r'[\u064B-\u0652\u0653-\u0655\u0670]')

MY50 = json.load(open('/home/z/my-project/scripts/step0_my_tashkeel.json',
                      encoding='utf-8')) if False else None

# import MY50 from the python file directly
import importlib.util
spec = importlib.util.spec_from_file_location(
    'm50', '/home/z/my-project/scripts/step0_my_tashkeel.py')
m50 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m50)
MY50 = m50.MY50

sample = json.load(open('/home/z/my-project/work/step0_sample50.json',
                        encoding='utf-8'))
raw_of = {r['utt']: r['transcript'] for r in sample}

# ---------- sanity: skeleton match (letters+punct+spaces identical) ----------
mismatch = 0
for utt, voc in MY50.items():
    raw = raw_of[utt]
    if TASHKEEL.sub('', voc) != raw:
        mismatch += 1
        print('SKELETON MISMATCH:', utt)
print(f'skeleton check: {50 - mismatch}/50 OK')

# ---------- word-level decisions ----------
word_forms = defaultdict(Counter)          # skeleton -> Counter(vocalized)
tanween_words = Counter()                  # skeletons carrying tanween
final_patterns = Counter()                 # last-char class of vocalized words
PUNCT = '.,?!:"\'\u060C\u061F'

for utt, voc in MY50.items():
    raw_words = raw_of[utt].split()
    voc_words = voc.split()
    assert len(raw_words) == len(voc_words), utt
    for rw, vw in zip(raw_words, voc_words):
        skel_word = rw.strip(PUNCT)
        core = vw.strip(PUNCT)
        if skel_word:
            word_forms[skel_word][core] += 1
            if '\u064B' in vw or '\u064C' in vw or '\u064D' in vw \
               or '\u064E\u064B' in vw or 'ً' in vw:
                tanween_words[core] += 1
            last = core[-1] if core else ''
            cls = ('DIACRITIC' if unicodedata.category(last) == 'Mn'
                   else 'letter:' + last)
            final_patterns[cls] += 1

# ---------- inconsistencies ----------
incons = {s: dict(c) for s, c in word_forms.items()
          if len(c) > 1 and sum(c.values()) >= 2}
multi = {s: sum(c.values()) for s, c in word_forms.items()
         if sum(c.values()) >= 2}

print('\n=== TANWEEN INVENTORY (forms carrying tanween) ===')
for w, n in tanween_words.most_common():
    print(f'  {w}  x{n}')

print('\n=== WORD-FINAL PATTERNS ===')
for cls, n in final_patterns.most_common(15):
    print(f'  {cls}: {n}')

print(f'\n=== INCONSISTENT SKELETONS (same word, >1 vocalization) ===')
print(f'count: {len(incons)}')
for s in sorted(incons, key=lambda x: -multi[x]):
    print(f'  {s} (x{multi[s]}): {incons[s]}')

print(f'\n=== CONSISTENT REPEATED WORDS: {len(multi) - len(incons)} ===')
out = {
    'inconsistent': {s: dict(c) for s, c in sorted(
        incons.items(), key=lambda kv: -multi[kv[0]])},
    'consistent_repeated': {s: c.most_common(1)[0][0]
                            for s, c in word_forms.items()
                            if sum(c.values()) >= 2 and len(c) == 1},
    'tanween_inventory': dict(tanween_words.most_common()),
    'n_unique_words': len(word_forms),
    'n_repeated': len(multi),
}
json.dump(out, open('/home/z/my-project/work/ph1_decisions.json', 'w',
                    encoding='utf-8'), ensure_ascii=False, indent=1)
print('saved -> work/ph1_decisions.json')
