# -*- coding: utf-8 -*-
"""Punctuation-alignment diagnostic — 05: scientific controls.

Three robustness checks on the surprising 04 result (untrained ','
tokens landing on real pauses with purity 0.82 vs random 0.16):

  A) PRIOR-ONLY control: recompute the hard alignment using ONLY the
     beta-binomial prior (attn = prior / rowsum, no learned distances,
     no model). If punct purity stays ~0.8 -> the placement is
     positional (prior) luck; if it drops -> the learned aligner
     distances actively push silence frames onto the punct token.

  B) STEAL analysis: for every punctuation range, which tokens owned
     those frames in the BASELINE alignment, and were those frames
     silent back then? Tells us whether punct takes frames that were
     already silence (harmless/beneficial) or steals speech frames
     from neighbors (harmful).

  C) Threshold sensitivity: recompute verdicts at -40 / -50 dB.
"""
import glob
import json
import os
import sys

import numpy as np

MIXER = '/home/z/my-project/work/scratch_data/mixer_repo'
TTSPKG = '/home/z/my-project/work/tts_arabic_pkg'
for p in (MIXER, TTSPKG):
    if p not in sys.path:
        sys.path.insert(0, p)

RES = '/home/z/my-project/work/punct_diag/results'
PUNCT_ID = {5, 6, 7, 8}
MIN_PAUSE_FRAMES = 4


def pause_runs(silent, min_frames=MIN_PAUSE_FRAMES):
    runs, i, T = [], 0, len(silent)
    while i < T:
        if silent[i]:
            j = i
            while j < T and silent[j]:
                j += 1
            if j - i >= min_frames:
                runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


def mas_from_log_attn(log_attn):
    """mas_width1 on a (T, L) log-attention matrix (single item)."""
    from models.mixer_tts.modules.helpers import mas_width1
    return mas_width1(log_attn.astype(np.float64))


def main():
    from models.mixer_tts.modules.data_function import (
        BetaBinomialInterpolator)
    betabin = BetaBinomialInterpolator(round_mel_len_to=200,
                                       round_text_len_to=40)
    rep = json.load(open(os.path.join(RES, 'run_report.json'),
                        encoding='utf-8'))
    by_utt = {r['utt']: r for r in rep}

    control = {'prior_only': [], 'steal': [], 'threshold': {}}
    for thr in (-40.0, -45.0, -50.0):
        control['threshold'][str(thr)] = {'CORRECT': 0, 'PARTIAL': 0,
                                          'WRONG': 0, 'n': 0}

    for path in sorted(glob.glob(os.path.join(RES, 'raw_*.npz'))):
        utt = os.path.basename(path)[4:-4]
        meta = by_utt[utt]
        if not meta['round_trip'] or not meta['inserted']:
            continue
        z = np.load(path, allow_pickle=True)
        ids_punct = z['ids_punct'].tolist()
        dur_punct = z['dur_punct']
        dur_base = z['dur_base']
        ids_base = z['ids_base'].tolist()
        rms_db = z['rms_db']
        T = int(meta['T'])
        inserted = z['inserted'].tolist()

        starts_p = np.concatenate([[0], np.cumsum(dur_punct)[:-1]]).astype(int)
        starts_b = np.concatenate([[0], np.cumsum(dur_base)[:-1]]).astype(int)
        ends_b = np.cumsum(dur_base).astype(int)

        # ---- A) prior-only alignment (no model at all) ----
        L = len(ids_punct)
        prior = betabin(T, L)                      # (T, L)
        prior_n = prior / prior.sum(axis=1, keepdims=True)
        log_attn = np.log(prior_n + 1e-12)
        hard = mas_from_log_attn(log_attn)          # (T, L) 0/1
        dur_prior = hard.sum(axis=0)
        starts_pr = np.concatenate(
            [[0], np.cumsum(dur_prior)[:-1]]).astype(int)

        silent45 = rms_db < -45.0
        for thr in (-40.0, -45.0, -50.0):
            sil = rms_db < thr
            runs = pause_runs(sil)
            for kp in inserted:
                d = int(dur_punct[kp])
                s, e = starts_p[kp], starts_p[kp] + d
                seg = sil[s:e]
                purity = float(seg.mean()) if d else 0.0
                overlap = any(s < b and a < e for a, b in runs)
                v = ('CORRECT' if purity >= 0.5 else
                     'PARTIAL' if overlap else 'WRONG')
                c = control['threshold'][str(thr)]
                c[v] += 1
                c['n'] += 1

        for kp in inserted:
            d = int(dur_punct[kp])
            s, e = starts_p[kp], starts_p[kp] + d
            # prior-only purity for this token
            dp = int(dur_prior[kp])
            sp, ep = starts_pr[kp], starts_pr[kp] + dp
            seg_pr = silent45[sp:ep]
            purity_pr = float(seg_pr.mean()) if dp else 0.0
            # steal: baseline owner of each taken frame
            owners = []
            for f in range(s, e):
                kb = int(np.searchsorted(ends_b, f, side='right'))
                owners.append(ids_base[kb] if kb < len(ids_base) else -1)
            taken_silent = float(silent45[s:e].mean()) if d else 0.0
            control['prior_only'].append({
                'utt': utt, 'pos': kp, 'dur_model': d, 'dur_prior': dp,
                'purity_model': float(silent45[s:e].mean()) if d else 0.0,
                'purity_prior': purity_pr})
            control['steal'].append({
                'utt': utt, 'pos': kp, 'dur': d,
                'taken_frames_silent_frac': taken_silent,
                'owners': owners})

    # aggregate prior-only
    po = control['prior_only']
    agg_po = {
        'n': len(po),
        'mean_dur_model': float(np.mean([p['dur_model'] for p in po])),
        'mean_dur_prior': float(np.mean([p['dur_prior'] for p in po])),
        'mean_purity_model': float(np.mean([p['purity_model'] for p in po])),
        'mean_purity_prior': float(np.mean([p['purity_prior'] for p in po])),
        'model_ge_50': float(np.mean([p['purity_model'] >= 0.5 for p in po])),
        'prior_ge_50': float(np.mean([p['purity_prior'] >= 0.5 for p in po])),
    }
    st = control['steal']
    all_stolen = [f for s in st for f in s['owners']]
    from collections import Counter
    owner_counts = Counter(all_stolen)
    agg_st = {
        'n_tokens': len(st),
        'mean_taken_silent_frac': float(np.mean(
            [s['taken_frames_silent_frac'] for s in st])),
        'stolen_frames_total': len(all_stolen),
        'stolen_from_silent_frames_frac': float(np.mean(
            [s['taken_frames_silent_frac'] for s in st])),
        'top_owners': owner_counts.most_common(8),
    }
    out = {'prior_only': agg_po, 'prior_only_tokens': po,
           'steal': agg_st, 'steal_tokens': st,
           'threshold': control['threshold']}
    with open(os.path.join(RES, 'controls.json'), 'w',
              encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=str)

    print('A) PRIOR-ONLY CONTROL')
    for k, v in agg_po.items():
        print(f'   {k:26s} {v}')
    print()
    print('B) STEAL ANALYSIS')
    for k, v in agg_st.items():
        print(f'   {k:26s} {v}')
    print()
    print('C) THRESHOLD SENSITIVITY (verdicts)')
    for thr, c in control['threshold'].items():
        print(f'   {thr} dB: {c}')
    print()
    print('per-token prior-only comparison:')
    for p in po:
        print(f"   {p['utt']} pos={p['pos']:3d} dur_model={p['dur_model']:2d}"
              f" dur_prior={p['dur_prior']:2d} "
              f"purity_model={p['purity_model']:.2f} "
              f"purity_prior={p['purity_prior']:.2f}")


if __name__ == '__main__':
    main()
