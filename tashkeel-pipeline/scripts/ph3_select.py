# -*- coding: utf-8 -*-
"""Phase 3 — smoke sample selection: 7 random (no Arabic punct) + 3 random
(with Arabic punct) from the frozen train pool (19,721)."""
import csv
import json
import random

rows = list(csv.DictReader(open(
    '/home/z/my-project/work/prep_output/extraction.csv', encoding='utf-8')))
train = [r for r in rows if r['split'] == 'train'
         and int(r['n_tokens']) <= 160 and int(r['n_frames']) <= 950]
assert len(train) == 19721, len(train)

pool_punct = [r for r in train
              if '\u060C' in r['transcript'] or '\u061F' in r['transcript']]
pool_nopunct = [r for r in train if r not in pool_punct]

random.seed(42)
sample = random.sample(pool_nopunct, 7) + random.sample(pool_punct, 3)

out = [{'id': r['utt'], 'text': r['transcript'],
        'n_tokens_recorded': int(r['n_tokens']), 'split': r['split']}
       for r in sample]
json.dump(out, open('/home/z/my-project/work/ph3_smoke_input.json', 'w',
                    encoding='utf-8'), ensure_ascii=False, indent=1)
for s in out:
    has_ar = '\u060C' in s['text'] or '\u061F' in s['text']
    print(f"{s['id']}  ar_punct={has_ar}  rec={s['n_tokens_recorded']}")
print('saved -> work/ph3_smoke_input.json (10 units)')
