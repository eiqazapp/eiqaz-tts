#!/usr/bin/env python3
"""ph3_diag_diff.py — compare source vs model output skeletons for failing
diagnostic items, to find the systematic cause of skeleton mismatches."""
import json
import re

STRIP = re.compile('[\u064B-\u0652\u0653-\u0655\u0670]')

d = json.load(open('/home/z/my-project/work/ph3_diag_out_diag.json',
                   encoding='utf-8'))
src_map = {u['id']: u['text'].replace('\u060C', ',').replace('\u061F', '?')
           for u in json.load(
               open('/home/z/my-project/work/ph3_full_shard_15.json',
                    encoding='utf-8'))}

shown = 0
for it in d['diag']:
    if it['ok'] or shown >= 5:
        continue
    if it.get('batchLine') is None:
        continue
    shown += 1
    out = it['outAfterRestore'] or ''
    src = src_map.get(it['id'], '')
    skel_out = STRIP.sub('', out)
    skel_src = STRIP.sub('', src)
    print('=' * 72)
    print('ID:', it['id'], '| reason:',
          (it['reason'] or '')[:110])
    print('SRC :', src[:130])
    print('OUT :', out[:130])
    print('skel equal?', skel_out == skel_src,
          f'| len out={len(skel_out)} src={len(skel_src)}')
    n = min(len(skel_out), len(skel_src))
    for i in range(n):
        if skel_out[i] != skel_src[i]:
            lo, hi = max(0, i - 10), i + 12
            print(f'first diff @{i}:')
            print('   src ctx:', repr(skel_src[lo:hi]))
            print('   out ctx:', repr(skel_out[lo:hi]))
            break

# also: do OK items show the و-split style? count standalone و in outputs
n_standalone = sum(
    1 for it in d['diag']
    if (it['outAfterRestore'] or '').split()
    and any(w == 'وِ' or w == 'و' for w in
            (it['outAfterRestore'] or '').split()))
n_src_standalone = sum(
    1 for t in src_map.values()
    if ' و ' in t or t.startswith('و '))
print()
print(f'items with standalone و in output: {n_standalone}/50')
print(f'source lines containing " و ": '
     f'{sum(1 for t in src_map.values() if " و " in t)}/50')
