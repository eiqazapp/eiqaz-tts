# -*- coding: utf-8 -*-
"""Step 0 — the actual measurement: catt_eo tokens vs my Egyptian tashkeel
tokens, through the EXACT Prep-kernel counting function."""
import json
import re
import sys

sys.path.insert(0, '/home/z/my-project/scripts')
from step0_counter import n_tokens_of, get_catt  # verified replica
from step0_my_tashkeel import MY50

STRIP = re.compile(r'[\u064B-\u0652\u0670]')   # all harakat + shadda + sukun + tanween + superscript-alef


def strip_harakat(t):
    return STRIP.sub('', t)


sample = json.load(open('/home/z/my-project/work/step0_sample50.json', encoding='utf-8'))
print(f'sample units: {len(sample)}; my tashkeel entries: {len(MY50)}')

# ---- 1) letter-integrity check (zero letter/space/punct changes) ----
bad = []
for s in sample:
    mine = MY50.get(s['utt'])
    if mine is None:
        bad.append((s['utt'], 'MISSING_ENTRY'))
        continue
    if strip_harakat(mine) != strip_harakat(s['transcript']):
        bad.append((s['utt'], 'LETTERS_DIFFER'))
if bad:
    print('INTEGRITY FAILURES:')
    for utt, why in bad:
        s = next(x for x in sample if x['utt'] == utt)
        print(f'  {utt}: {why}')
        print('    raw :', repr(strip_harakat(s['transcript'])))
        print('    mine:', repr(strip_harakat(MY50[utt])))
    sys.exit(1)
print('integrity check: ALL 50 PASS (letters/space/punct identical)')

# ---- 2) local catt re-run -> verify replica against recorded counts ----
catt = get_catt()
catt_mismatch = 0
results = []
for s in sample:
    raw = s['transcript']
    voc = catt.predict(raw)
    n_catt_local = n_tokens_of(voc)
    n_rec = s['n_tokens_catt']
    if n_catt_local != n_rec:
        catt_mismatch += 1
        print(f"  [catt-verify] {s['utt']}: recorded={n_rec} local={n_catt_local}")
    n_mine = n_tokens_of(MY50[s['utt']])
    results.append({'utt': s['utt'], 'raw': raw, 'catt_text': voc,
                    'my_text': MY50[s['utt']],
                    'n_catt': n_rec, 'n_catt_local': n_catt_local,
                    'n_mine': n_mine, 'n_frames': s['n_frames'],
                    'spk': s['spk']})
print(f'catt verification: {len(sample) - catt_mismatch}/{len(sample)} '
      f'match the kernel-recorded counts exactly')

# ---- 3) the headline numbers ----
cross = [r for r in results if r['n_catt'] <= 160 and r['n_mine'] > 160]
over180 = [r for r in results if r['n_mine'] > 180]
ratios = sorted(r['n_mine'] / r['n_catt'] for r in results)
import statistics as st
print('--- HEADLINE ---')
print(f'units under 160 with catt but OVER 160 with my tashkeel: '
      f'{len(cross)}/50')
for r in cross:
    print(f"   {r['utt']}: catt={r['n_catt']} -> mine={r['n_mine']}")
print(f'units over 180 (hard write-filter) with mine: {len(over180)}/50')
print(f'ratio mine/catt: min={ratios[0]:.3f} p25={ratios[12]:.3f} '
      f'median={ratios[25]:.3f} p75={ratios[37]:.3f} max={ratios[-1]:.3f} '
      f'mean={st.mean(ratios):.3f}')

# ---- 4) extrapolation to the FULL train pool (19,721 units) ----
import pandas as pd
df = pd.read_csv('/home/z/my-project/work/prep_output/extraction.csv')
pool = df[(df['split'] == 'train') & (df.n_tokens <= 160)
          & (df.n_frames <= 950)]
ntok = pool.n_tokens.values
for name, r in [('p25', ratios[12]), ('median', ratios[25]),
                ('p75', ratios[37]), ('mean', st.mean(ratios))]:
    est = int((ntok * r > 160).sum())
    est180 = int((ntok * r > 180).sum())
    print(f'extrapolation @ratio={name} ({r:.3f}): ~{est} pool units '
          f'cross 160 ({100 * est / len(ntok):.1f}%); ~{est180} cross 180')

# ---- 5) save ----
out = {'results': results,
       'headline': {
           'n_sample': len(results),
           'cross_160': len(cross),
           'cross_180': len(over180),
           'ratio_min': round(ratios[0], 3), 'ratio_p25': round(ratios[12], 3),
           'ratio_median': round(ratios[25], 3), 'ratio_p75': round(ratios[37], 3),
           'ratio_max': round(ratios[-1], 3), 'ratio_mean': round(st.mean(ratios), 3),
           'catt_verify_matches': len(sample) - catt_mismatch},
       'pool_size': int(len(ntok))}
json.dump(out, open('/home/z/my-project/work/step0_results.json', 'w',
                    encoding='utf-8'), ensure_ascii=False, indent=1)
print('saved -> work/step0_results.json')
