# -*- coding: utf-8 -*-
"""Punctuation-alignment diagnostic — 04: analysis + report.

Loads raw results from 03_run.py, applies silence detection
(rms_db < SILENCE_DB, pause runs >= MIN_PAUSE_FRAMES) and computes:

  1. BASELINE (no punctuation, exact training conditions):
     how the trained mechanism treats the non-phonetic '_' separator
     tokens — do they receive real silence?
  2. PUNCTUATED (text-only change): where does each inserted ','/'?'
     token (untrained embedding in the 4h checkpoint) land?
       - purity (fraction of its frames that are real silence)
       - center-in-pause, distance to nearest pause
       - frames stolen from speech vs silence (vs the baseline alignment)
       - random-placement baseline for the same durations
       - disturbance of neighboring token durations
  3. Pause availability at punctuation positions.

Verdicts per punctuation token:
  CORRECT  : purity >= 0.5 (majority of its frames are real silence)
  PARTIAL  : overlaps a pause run but purity < 0.5
  WRONG    : no overlap with any silent frame
"""
import glob
import json
import os

import numpy as np

RES = '/home/z/my-project/work/punct_diag/results'
SILENCE_DB = -45.0
MIN_PAUSE_FRAMES = 4        # >= ~46 ms
NEARBY_FRAMES = 20          # +- 232 ms pause-availability window
N_DRAWS = 2000

PUNCT_ID = {5: '.', 6: ',', 7: '?', 8: '!'}
SEP_ID = 4
EOS_ID = 1
DBL_ID = 3


def pause_runs(silent):
    """consecutive silent-frame runs >= MIN_PAUSE_FRAMES."""
    runs, i = [], 0
    T = len(silent)
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


def ranges_from_durs(dur):
    starts = np.concatenate([[0], np.cumsum(dur)[:-1]]).astype(int)
    ends = np.cumsum(dur).astype(int)
    return starts, ends


def analyze():
    rep = json.load(open(os.path.join(RES, 'run_report.json'),
                        encoding='utf-8'))
    by_utt = {r['utt']: r for r in rep}

    out = {'config': {'silence_db': SILENCE_DB,
                      'min_pause_frames': MIN_PAUSE_FRAMES,
                      'nearby_frames': NEARBY_FRAMES},
           'units': [], 'punct_tokens': [], 'separators': [],
           'eos_tokens': []}

    for path in sorted(glob.glob(os.path.join(RES, 'raw_*.npz'))):
        utt = os.path.basename(path)[4:-4]
        meta = by_utt[utt]
        z = np.load(path, allow_pickle=True)
        ids_base = z['ids_base'].tolist()
        dur_base = z['dur_base']
        rms_db = z['rms_db']
        T = len(dur_base.sum()) if False else int(meta['T'])
        silent = rms_db < SILENCE_DB
        runs = pause_runs(silent)
        pauses = [(a, b) for a, b in runs]

        # ---------- 1) baseline: separator / eos behaviour ----------
        starts_b, ends_b = ranges_from_durs(dur_base)
        for k, tid in enumerate(ids_base):
            if tid == SEP_ID:
                s, e = starts_b[k], ends_b[k]
                seg = silent[s:e]
                out['separators'].append({
                    'utt': utt, 'k': k, 'dur': int(dur_base[k]),
                    'purity': float(seg.mean()) if len(seg) else 0.0,
                    'in_pause': bool(any(s < b and a < e for a, b in runs)),
                })
            elif tid == EOS_ID:
                s, e = starts_b[k], ends_b[k]
                seg = silent[s:e]
                out['eos_tokens'].append({
                    'utt': utt, 'dur': int(dur_base[k]),
                    'purity': float(seg.mean()) if len(seg) else 0.0})

        # ---------- 2) punctuated variant ----------
        if not meta['round_trip'] or len(z['inserted']) == 0:
            out['units'].append({
                'utt': utt, 'T': T, 'n_pauses': len(pauses),
                'silence_frac': float(silent.mean()),
                'skip_punct': True, 'reason': 'round_trip_fail'})
            continue
        ids_punct = z['ids_punct'].tolist()
        dur_punct = z['dur_punct']
        starts_p, ends_p = ranges_from_durs(dur_punct)
        inserted = z['inserted'].tolist()

        # shared-token duration disturbance (baseline vs punctuated)
        shared_dur_base, shared_dur_punct = [], []
        kb = 0
        for kp, tid in enumerate(ids_punct):
            if tid in PUNCT_ID:
                continue
            shared_dur_punct.append(float(dur_punct[kp]))
            shared_dur_base.append(float(dur_base[kb]))
            kb += 1
        corr = float(np.corrcoef(shared_dur_base, shared_dur_punct)[0, 1])

        rng = np.random.default_rng(1234)
        for kp in inserted:
            tid = ids_punct[kp]
            s, e = starts_p[kp], ends_p[kp]
            d = int(dur_punct[kp])
            seg = silent[s:e]
            purity = float(seg.mean()) if d else 0.0
            overlap = any(s < b and a < e for a, b in runs)
            center = (s + e) / 2.0
            in_pause = any(a <= center < b for a, b in runs)
            if runs:
                dist_frames = min(abs(center - (a + b) / 2.0)
                                  for a, b in runs)
            else:
                dist_frames = 1e9
            # frames stolen (owner under baseline alignment)
            owners = np.searchsorted(ends_b, np.arange(s, e), side='right')
            stolen_silent = float(silent[s:e].mean()) if d else 0.0
            # random baseline for the same duration
            if d and T > d:
                rstarts = rng.integers(0, T - d, N_DRAWS)
                rp = [silent[rs:rs + d].mean() for rs in rstarts]
                rand_purity = float(np.mean(rp))
                rcenters = rstarts + d / 2.0
                rand_in_pause = float(np.mean([
                    any(a <= c < b for a, b in runs) for c in rcenters]))
            else:
                rand_purity, rand_in_pause = 0.0, 0.0
            # pause available nearby?
            nearby = any(abs(center - (a + b) / 2.0) <= NEARBY_FRAMES
                         for a, b in runs)
            soft = z[f'soft_{kp}'] if f'soft_{kp}' in z.files else None
            soft_peak_frame = int(np.argmax(soft)) if soft is not None else None
            out['punct_tokens'].append({
                'utt': utt, 'token': PUNCT_ID[tid], 'pos': kp,
                'dur': d, 'start': int(s), 'end': int(e),
                'purity': purity, 'overlap_pause': bool(overlap),
                'center_in_pause': bool(in_pause),
                'dist_to_pause_ms': round(dist_frames * 256 / 22.05, 1),
                'nearby_pause': bool(nearby),
                'rand_purity': rand_purity,
                'rand_center_in_pause': rand_in_pause,
                'soft_peak_frame': soft_peak_frame,
                'soft_peak_t_ms': (round(soft_peak_frame * 256 / 22.05, 1)
                                   if soft_peak_frame is not None else None),
                'verdict': ('CORRECT' if purity >= 0.5 else
                            'PARTIAL' if overlap else 'WRONG'),
            })

        out['units'].append({
            'utt': utt, 'T': T, 'n_pauses': len(pauses),
            'silence_frac': float(silent.mean()),
            'L_base': len(ids_base), 'L_punct': len(ids_punct),
            'dur_corr_shared': corr,
            'pauses_ms': [(round(a * 256 / 22.05, 1),
                           round((b - a) * 256 / 22.05, 1))
                          for a, b in pauses],
            'skip_punct': False})

    # ---------- aggregate ----------
    pt = out['punct_tokens']
    seps = out['separators']
    agg = {
        'n_units': len(out['units']),
        'n_units_punct': sum(1 for u in out['units'] if not u['skip_punct']),
        'n_punct_tokens': len(pt),
        'punct_dur_mean': float(np.mean([p['dur'] for p in pt])) if pt else 0,
        'verdicts': {v: sum(1 for p in pt if p['verdict'] == v)
                     for v in ('CORRECT', 'PARTIAL', 'WRONG')},
        'purity_mean': float(np.mean([p['purity'] for p in pt])) if pt else 0,
        'rand_purity_mean': (float(np.mean([p['rand_purity'] for p in pt]))
                             if pt else 0),
        'center_in_pause_rate': (float(np.mean([p['center_in_pause']
                                                for p in pt])) if pt else 0),
        'rand_center_in_pause_rate': (float(np.mean(
            [p['rand_center_in_pause'] for p in pt])) if pt else 0),
        'nearby_pause_rate': (float(np.mean([p['nearby_pause'] for p in pt]))
                              if pt else 0),
        'conditional_capture': (
            float(np.mean([p['center_in_pause'] for p in pt
                           if p['nearby_pause']]))
            if any(p['nearby_pause'] for p in pt) else None),
        'conditional_capture_rand': (
            float(np.mean([p['rand_center_in_pause'] for p in pt
                           if p['nearby_pause']]))
            if any(p['nearby_pause'] for p in pt) else None),
        'sep_dur_mean': float(np.mean([s['dur'] for s in seps])) if seps else 0,
        'sep_purity_mean': (float(np.mean([s['purity'] for s in seps]))
                            if seps else 0),
        'sep_frac_purity_ge_50': (float(np.mean([s['purity'] >= 0.5
                                                 for s in seps]))
                                  if seps else 0),
        'sep_frac_purity_ge_80': (float(np.mean([s['purity'] >= 0.8
                                                 for s in seps]))
                                  if seps else 0),
        'eos_dur_mean': (float(np.mean([e['dur'] for e in
                                        out['eos_tokens']]))
                         if out['eos_tokens'] else 0),
        'eos_purity_mean': (float(np.mean([e['purity'] for e in
                                           out['eos_tokens']]))
                            if out['eos_tokens'] else 0),
        'silence_frac_mean': float(np.mean([u['silence_frac']
                                            for u in out['units']])),
        'n_pauses_total': sum(u['n_pauses'] for u in out['units']),
        'dur_corr_shared_mean': float(np.mean(
            [u['dur_corr_shared'] for u in out['units']
             if not u['skip_punct']])),
    }
    out['aggregate'] = agg
    with open(os.path.join(RES, 'analysis.json'), 'w',
              encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    # ---------- console summary ----------
    print('=' * 72)
    print('AGGREGATE')
    print('=' * 72)
    for k, v in agg.items():
        print(f'  {k:32s} {v}')
    print()
    print('PUNCT TOKENS (per token):')
    for p in pt:
        print(f"  {p['utt']} tok={p['token']} pos={p['pos']:3d} "
              f"dur={p['dur']:2d}f purity={p['purity']:.2f} "
              f"(rand {p['rand_purity']:.2f}) center_in_pause="
              f"{p['center_in_pause']} nearby={p['nearby_pause']} "
              f"dist={p['dist_to_pause_ms']:.0f}ms "
              f"verdict={p['verdict']}")
    print()
    print('SEPARATOR tokens: n=%d mean_dur=%.1f purity=%.2f '
          'frac>=0.5=%.2f frac>=0.8=%.2f' % (
              len(seps), agg['sep_dur_mean'], agg['sep_purity_mean'],
              agg['sep_frac_purity_ge_50'], agg['sep_frac_purity_ge_80']))
    print('saved ->', os.path.join(RES, 'analysis.json'))


if __name__ == '__main__':
    analyze()
