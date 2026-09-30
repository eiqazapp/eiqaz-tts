# -*- coding: utf-8 -*-
"""Punctuation-alignment diagnostic — 06: illustrative examples.

Builds the concrete per-example table for the final report: for selected
punctuation tokens show the word that carried the comma/question mark,
the aligned frame range in seconds, the nearest real pause in seconds,
purity, and verdict. Also prints the token neighbourhood (what surrounds
the punctuation token in the token stream).
"""
import glob
import json
import os

import numpy as np

RES = '/home/z/my-project/work/punct_diag/results'
SILENCE_DB = -45.0
MIN_PAUSE_FRAMES = 4
MS = 256 / 22.05

PUNCT_ID = {5: '.', 6: ',', 7: '?', 8: '!'}
SYM = ['_pad_', '_eos_', '_sil_', '_dbl_', '_+_', '.', ',', '?', '!',
       '<', 'b', 't', '^', 'j', 'H', 'x', 'd', '*', 'r', 'z', 's', '$',
       'S', 'D', 'T', 'Z', 'E', 'g', 'f', 'q', 'k', 'l', 'm', 'n', 'h',
       'w', 'y', 'v', 'a', 'u', 'i', 'aa', 'uu', 'ii']


def pause_runs(silent):
    runs, i, T = [], 0, len(silent)
    while i < T:
        if silent[i]:
            j = i
            while j < T and silent[j]:
                j += 1
            if j - i >= MIN_PAUSE_FRAMES:
                runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


def main():
    ana = json.load(open(os.path.join(RES, 'analysis.json'),
                         encoding='utf-8'))
    rep = json.load(open(os.path.join(RES, 'run_report.json'),
                         encoding='utf-8'))
    by_utt = {r['utt']: r for r in rep}
    sel = json.load(open('/home/z/my-project/work/punct_diag/selection.json',
                         encoding='utf-8'))
    tr_by_utt = {u['utt']: u for u in sel['units']}

    examples = []
    for path in sorted(glob.glob(os.path.join(RES, 'raw_*.npz'))):
        utt = os.path.basename(path)[4:-4]
        meta = by_utt[utt]
        if not meta['round_trip'] or not meta['inserted']:
            continue
        z = np.load(path, allow_pickle=True)
        ids_punct = z['ids_punct'].tolist()
        dur_punct = z['dur_punct']
        dur_base = z['dur_base']
        rms = z['rms_db']
        T = int(meta['T'])
        silent = rms < SILENCE_DB
        runs = pause_runs(silent)
        starts = np.concatenate([[0], np.cumsum(dur_punct)[:-1]]).astype(int)

        # words carrying marks (from original transcript)
        marks = {wi: ch for wi, ch in meta['marks']}
        tr_words = tr_by_utt[utt]['transcript'].split()
        marked_words = {wi: tr_words[wi] for wi in marks}

        # count tokens per word to find which word each punct belongs to:
        # walk the token stream; separators '_+_' (id 4) delimit words.
        # simpler: the punct token is the LAST token before its '_+_'.
        for kp in meta['inserted']:
            d = int(dur_punct[kp])
            s, e = starts[kp], starts[kp] + d
            # neighbourhood tokens
            lo = max(0, kp - 3)
            hi = min(len(ids_punct), kp + 4)
            nb = [SYM[i] if i < len(SYM) else str(i)
                  for i in ids_punct[lo:hi]]
            # which word? count separators before kp in baseline ids
            n_sep_before = sum(1 for i in ids_punct[:kp] if i == 4)
            word = marked_words.get(n_sep_before, '?')
            # nearest pause
            center = (s + e) / 2
            if runs:
                best = min(runs, key=lambda ab: abs((ab[0] + ab[1]) / 2
                                                    - center))
                pause_ms = ((best[0] + best[1]) / 2) * MS
                pause_len = (best[1] - best[0]) * MS
            else:
                pause_ms, pause_len = None, None
            purity = float(silent[s:e].mean()) if d else 0.0
            examples.append({
                'utt': utt, 'word': word,
                'mark': marks.get(
                    [wi for wi in marks
                     if marked_words[wi] == word][0], '?'),
                'token': SYM[ids_punct[kp]],
                'n_sep_before': n_sep_before,
                'dur_frames': d,
                'range_ms': (round(s * MS), round(e * MS)),
                'purity': round(purity, 2),
                'pause_center_ms': (round(pause_ms, 0)
                                    if pause_ms is not None else None),
                'pause_len_ms': (round(pause_len, 0)
                                 if pause_len is not None else None),
                'neighbourhood': nb,
                'verdict': ('CORRECT' if purity >= 0.5 else
                            'PARTIAL' if any(s < b and a < e
                                             for a, b in runs) else 'WRONG'),
            })

    examples.sort(key=lambda x: -x['dur_frames'])
    with open(os.path.join(RES, 'examples.json'), 'w', encoding='utf-8') as f:
        json.dump(examples, f, ensure_ascii=False, indent=1)
    print(f'{len(examples)} examples saved')
    print()
    hdr = (f"{'utt':12s} {'word':14s} {'tok':2s} {'dur':>3s} "
           f"{'range_ms':>14s} {'purity':>6s} {'pause@ms':>8s} "
           f"{'pause_len':>9s} verdict")
    print(hdr)
    print('-' * len(hdr))
    for ex in examples:
        print(f"{ex['utt']:12s} {ex['word']:14s} {ex['token']:2s} "
              f"{ex['dur_frames']:3d} "
              f"[{ex['range_ms'][0]:6d},{ex['range_ms'][1]:6d}] "
              f"{ex['purity']:6.2f} "
              f"{str(ex['pause_center_ms']):>8s} "
              f"{str(ex['pause_len_ms']):>9s} {ex['verdict']}")
    print()
    print('token neighbourhoods (first 8 examples):')
    for ex in examples[:8]:
        print(f"  {ex['utt']} ... {' '.join(ex['neighbourhood'])}")


if __name__ == '__main__':
    main()
