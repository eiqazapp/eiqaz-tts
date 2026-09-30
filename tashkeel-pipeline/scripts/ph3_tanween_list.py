# -*- coding: utf-8 -*-
"""Generate the corpus-grounded tanween closed list (raw tanween inventory).
Principle: Whisper transcribed these words WITH tanween fath => actually
spoken with -an in the audio => allowed in our tashkeel output."""
import csv
import json
import re
from collections import Counter

TASH = re.compile(r'[\u064B-\u0652\u0653-\u0655\u0670]')
STRIP_CHARS = '.,?!:;"()\u060C\u061F\u061B\u201C\u201D'

skeletons = Counter()
with open('/home/z/my-project/work/prep_output/extraction.csv',
          encoding='utf-8') as f:
    for r in csv.DictReader(f):
        for w in r['transcript'].split():
            if '\u064B' in w:
                skel = TASH.sub('', w).strip(STRIP_CHARS)
                # strip leading conjunction prefixes for the lookup list too
                core = skel[1:] if skel[:1] in ('و', 'ف') and len(skel) > 3 else skel
                if core:
                    skeletons[core] += 1
                if skel and skel != core:
                    skeletons[skel] += 0  # ensure prefixed variant maps to core

closed = sorted(skeletons.keys())
json.dump({
    'source': 'extraction.csv raw tanween-fath inventory (Whisper-marked)',
    'principle': 'tanween fath allowed ONLY on these skeletons (spoken -an '
                 'evidence from the corpus transcripts); everything else '
                 'must end in sukun/unmarked/long vowel',
    'prefix_rule': 'leading و/ف conjunction is stripped before lookup',
    'n_words': len(closed),
    'words': closed,
}, open('/home/z/my-project/work/tanween_closed_list.json', 'w',
        encoding='utf-8'), ensure_ascii=False, indent=1)
print(f'closed list: {len(closed)} skeletons')
print('top 25:', closed[:25])
