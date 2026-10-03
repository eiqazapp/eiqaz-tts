#!/usr/bin/env python3
"""ph3_full_shards.py — split ph2_input_full.json into N contiguous shards.

Pure operational sharding for parallel full-run; the engine v5 itself is
untouched. Each shard is a plain list of {id,text,n_tokens_recorded,split}.
"""
import json
import sys

N = int(sys.argv[1]) if len(sys.argv) > 1 else 16
SRC = '/home/z/my-project/work/ph2_input_full.json'
OUT = '/home/z/my-project/work/ph3_full_shard_{i}.json'

with open(SRC, encoding='utf-8') as f:
    data = json.load(f)

n = len(data)
base, extra = divmod(n, N)
idx = 0
for s in range(N):
    k = base + (1 if s < extra else 0)
    shard = data[idx:idx + k]
    idx += k
    with open(OUT.format(i=s), 'w', encoding='utf-8') as f:
        json.dump(shard, f, ensure_ascii=False)
    print(f'shard {s:02d}: {k:5d} units  [{shard[0]["id"]} .. {shard[-1]["id"]}]')
print(f'total={idx} (expected {n})')
assert idx == n
