#!/usr/bin/env python3
"""ph3_full_retry_prep.py — prepare the retry-round input.

Collects ids that are 'failed' in the current record set, plus ids with no
record at all (excluding quarantined — deliberate state, never retried).
Writes work/ph3_retry_round<k>_input.json. The engine then runs on it with a
FRESH output file (its done-set starts empty). The engine itself is untouched.

Usage: python3 scripts/ph3_full_retry_prep.py [round]
"""
import glob
import json
import os
import sys

BASE = '/home/z/my-project/work'
ROUND = sys.argv[1] if len(sys.argv) > 1 else '1'
INPUT = f'{BASE}/ph2_input_full.json'
SOURCES = sorted(glob.glob(f'{BASE}/ph3_full_shard_*.jsonl')) + \
          sorted(glob.glob(f'{BASE}/ph3_retry_round*_output.jsonl')) + \
          ([f'{BASE}/ph3_full_output.jsonl']
           if os.path.exists(f'{BASE}/ph3_full_output.jsonl') else [])
OUT = f'{BASE}/ph3_retry_round{ROUND}_input.json'

with open(INPUT, encoding='utf-8') as f:
    inp = {u['id']: u for u in json.load(f)}

status = {}
for path in SOURCES:
    if not os.path.exists(path):
        continue
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            rid, st = r.get('id'), r.get('status')
            if rid is None:
                continue
            # precedence for retry decision: any ok anywhere -> done;
            # quarantined sticks; otherwise failed/missing -> retry
            if status.get(rid) in ('ok', 'quarantined'):
                continue
            status[rid] = st if st in ('ok', 'quarantined', 'failed') else 'failed'

retry_ids = [rid for rid, st in status.items()
             if st == 'failed' and rid in inp]
missing_ids = [rid for rid in inp if rid not in status]
targets = retry_ids + missing_ids

if not targets:
    print('nothing to retry — all units have ok/quarantined records')
    raise SystemExit(0)

with open(OUT, 'w', encoding='utf-8') as f:
    json.dump([inp[rid] for rid in targets], f, ensure_ascii=False)

print(f'round {ROUND}: retry input = {len(targets)} units '
      f'({len(retry_ids)} failed + {len(missing_ids)} missing)')
print('sample:', targets[:10])
