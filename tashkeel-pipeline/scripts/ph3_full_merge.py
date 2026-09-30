#!/usr/bin/env python3
"""ph3_full_merge.py — merge shard outputs (+ retry-round outputs) into the
canonical full-run output: one line per input unit, in input order.

Precedence per unit id:
  quarantined (deliberate state — never LLM'd) > ok (latest wins) > failed.
Missing ids (no record anywhere) are reported for the retry-prep step.

Usage: python3 scripts/ph3_full_merge.py
Outputs:
  work/ph3_full_output.jsonl       canonical, 21,880 lines, input order
  work/ph3_full_output_meta.json   provenance + counts
  stdout summary
"""
import glob
import hashlib
import json
import os

BASE = '/home/z/my-project/work'
INPUT = f'{BASE}/ph2_input_full.json'
SOURCES = sorted(glob.glob(f'{BASE}/ph3_full_shard_*.jsonl')) + \
          sorted(glob.glob(f'{BASE}/ph3_retry_round*_output.jsonl'))
OUT = f'{BASE}/ph3_full_output.jsonl'
META = f'{BASE}/ph3_full_output_meta.json'

with open(INPUT, encoding='utf-8') as f:
    inp = json.load(f)
inp_ids = [u['id'] for u in inp]

RANK = {'quarantined': 3, 'ok': 2, 'failed': 1}
best = {}          # id -> (rank, order, record)
order = 0
src_files = []
for path in SOURCES:
    if not os.path.exists(path):
        continue
    src_files.append({
        'path': os.path.basename(path),
        'sha256': hashlib.sha256(open(path, 'rb').read()).hexdigest(),
    })
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            rid = r.get('id')
            st = r.get('status')
            if rid is None or st not in RANK:
                continue
            order += 1
            cur = best.get(rid)
            if cur is None:
                best[rid] = (RANK[st], order, r)
            else:
                # higher rank wins; equal rank -> later record wins
                if RANK[st] >= cur[0]:
                    best[rid] = (RANK[st], order, r)

missing = [i for i in inp_ids if i not in best]
counts = {'ok': 0, 'failed': 0, 'quarantined': 0}
with open(OUT, 'w', encoding='utf-8') as f:
    for uid in inp_ids:
        rec = best.get(uid)
        if rec is None:
            continue  # missing — reported separately
        counts[rec[2]['status']] += 1
        f.write(json.dumps(rec[2], ensure_ascii=False) + '\n')

split_counts = {}
for u in inp:
    st = best.get(u['id'], (0, 0, {'status': 'missing'}))[2]['status']
    split_counts.setdefault(u.get('split', '?'), {}).setdefault(st, 0)
    split_counts[u.get('split', '?')][st] += 1

meta = {
    'engine': 'ph3_engine.mjs v5 (guide v2.1 — approved 2026-09-30)',
    'run_scope': 'full run over ph2_input_full.json (21,880 units)',
    'input_path': INPUT,
    'input_sha256': hashlib.sha256(open(INPUT, 'rb').read()).hexdigest(),
    'sources': src_files,
    'canonical_output': OUT,
    'canonical_sha256': hashlib.sha256(open(OUT, 'rb').read()).hexdigest(),
    'n_input': len(inp_ids),
    'n_records': sum(counts.values()),
    'n_missing': len(missing),
    'counts': counts,
    'counts_by_split': split_counts,
    'user_decisions_2026_09_30': [
        'smoke sample approved as-is',
        'اوي: NO Tier A rule — record all forms for post-run consistency '
        'analysis (this deliverable); no engine/guide change',
        '0_03547_01: stays in manual-review path — no manual correction '
        'enters the full run',
    ],
    'precedence': 'quarantined > ok (latest) > failed (latest)',
}
with open(META, 'w', encoding='utf-8') as f:
    json.dump(meta, f, ensure_ascii=False, indent=1)

print(f'input units      : {len(inp_ids)}')
print(f'records written  : {sum(counts.values())}  {counts}')
print(f'missing ids      : {len(missing)}')
for s, c in sorted(split_counts.items()):
    print(f'  split {s}: {c}')
if missing:
    print('missing sample  :', missing[:12])
