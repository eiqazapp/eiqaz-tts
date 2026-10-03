# -*- coding: utf-8 -*-
"""Punctuation-alignment diagnostic — 01: sample selection.

Selects 20 train-pool units from the NileTTS 4h extraction.csv that contain
Arabic punctuation (comma/question marks) in their transcripts, so we can:
  - reproduce the exact training features (ids + mel) for each unit,
  - insert an ASCII punctuation token (',' or '?') at those positions
    (text-only change) and re-run the SAME alignment mechanism used by
    the "NileTTS 4h Train" kernel.

Constraints honored:
  * split == 'train' (actual training pool)
  * n_tokens <= 160 and n_frames <= 950 (the train kernel's pool filters)
  * duration 2.5-9.0 s
  * 1-3 punctuation marks per transcript (clean interpretation)
  * both speakers, at most 2 units per source row (download economy)
"""
import json
import random

import pandas as pd

CSV = '/home/z/my-project/work/prep_output/extraction.csv'
OUT = '/home/z/my-project/work/punct_diag/selection.json'
N_PER_SPK = 10
MAX_PER_ROW = 2
SEED = 42


def punct_marks(t):
    return [(i, w[-1]) for i, w in enumerate(t.split())
            if w and w[-1] in '\u060c\u061f']


def main():
    df = pd.read_csv(CSV)
    pool = df[(df.split == 'train') & (df.n_tokens <= 160)
              & (df.n_frames <= 950) & (df.duration >= 2.5)
              & (df.duration <= 9.0)].copy()
    pool['marks'] = pool.transcript.map(lambda t: punct_marks(t))
    pool['n_marks'] = pool['marks'].map(len)
    cand = pool[(pool.n_marks >= 1) & (pool.n_marks <= 3)]
    print(f'pool with 1-3 punct marks: {len(cand)} units '
          f'(of {len(pool)} pool units)')

    selected = []
    per_row = {}
    for spk in (0, 1):
        sub = cand[cand.spk_idx == spk].sample(
            frac=1, random_state=SEED).reset_index(drop=True)
        got = 0
        for _, r in sub.iterrows():
            if per_row.get(r.src_row, 0) >= MAX_PER_ROW:
                continue
            selected.append({
                'utt': r.utt, 'spk_idx': int(r.spk_idx),
                'src_row': r.src_row, 'i0': int(r.i0), 'i1': int(r.i1),
                'transcript': r.transcript,
                'duration': float(r.duration),
                'n_tokens': int(r.n_tokens), 'n_frames': int(r.n_frames),
                'marks': [[i, ch] for i, ch in r.marks],
            })
            per_row[r.src_row] = per_row.get(r.src_row, 0) + 1
            got += 1
            if got >= N_PER_SPK:
                break

    rows = sorted(set(u['src_row'] for u in selected))
    print(f'selected {len(selected)} units | {len(rows)} distinct source rows')
    print('marks breakdown:',
          {'comma': sum(1 for u in selected for _, ch in u['marks']
                        if ch == '\u060c'),
           'question': sum(1 for u in selected for _, ch in u['marks']
                           if ch == '\u061f')})
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump({'units': selected, 'rows': rows}, f,
                  ensure_ascii=False, indent=1)
    print('saved ->', OUT)
    for u in selected[:5]:
        print(f"  {u['utt']} spk{u['spk_idx']} marks={u['marks']} "
              f"tok={u['n_tokens']} frames={u['n_frames']} "
              f"\"{u['transcript'][:60]}\"")


if __name__ == '__main__':
    main()
