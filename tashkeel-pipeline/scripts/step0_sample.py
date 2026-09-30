# -*- coding: utf-8 -*-
"""Step 0 — draw the 50-unit sample from the REAL train pool.

Population: extraction.csv units with split=='train', n_tokens<=160,
n_frames<=950  (== the exact pool the Saturday train kernel loads:
19,721 units).  Seeded (42) => reproducible.
"""
import json
import random
import pandas as pd

df = pd.read_csv('/home/z/my-project/work/prep_output/extraction.csv')
pool = df[(df['split'] == 'train') & (df.n_tokens <= 160)
          & (df.n_frames <= 950)].reset_index(drop=True)
print('pool size:', len(pool))

rng = random.Random(42)
idx = rng.sample(range(len(pool)), 50)
rows = [pool.iloc[i] for i in sorted(idx)]

out = []
for r in rows:
    out.append({'utt': r.utt, 'transcript': r.transcript,
                'n_tokens_catt': int(r.n_tokens),
                'n_frames': int(r.n_frames),
                'duration': float(r.duration), 'spk': int(r.spk_idx)})

with open('/home/z/my-project/work/step0_sample50.json', 'w',
          encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=1)

import numpy as np
t = [o['n_tokens_catt'] for o in out]
print('sample n_tokens: min/med/mean/max =', min(t),
      int(np.median(t)), round(float(np.mean(t)), 1), max(t))
print('sample >=140 tokens:', sum(1 for x in t if x >= 140))
print('sample >=120 tokens:', sum(1 for x in t if x >= 120))
for o in out:
    print(f"{o['utt']}|{o['n_tokens_catt']}|{o['transcript']}")
