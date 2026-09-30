#!/usr/bin/env python3
"""ph3_full_progress.py — live progress/quality dashboard for the full run.

Usage: python3 scripts/ph3_full_progress.py
"""
import glob
import json
from collections import Counter


def kind_of(rs):
    checks = [
        ('429-API', ('429',)),
        ('SKELETON', ('الهيكل',)),
        ('TAA', ('المربوطة',)),
        ('TANWEEN', ('تنوين',)),
        ('PARSE', ('التحليل',)),
        ('DENSITY', ('الحركات', 'خالٍ')),
    ]
    for name, keys in checks:
        if any(k in rs for k in keys):
            return name
    return 'OTHER'


st = Counter()
att = Counter()
kinds = Counter()
per_shard = {}
for p in sorted(glob.glob('/home/z/my-project/work/ph3_full_shard_*.jsonl')):
    ps = Counter()
    for l in open(p, encoding='utf-8'):
        l = l.strip()
        if not l:
            continue
        try:
            r = json.loads(l)
        except json.JSONDecodeError:
            continue
        st[r['status']] += 1
        ps[r['status']] += 1
        if r['status'] in ('ok', 'failed'):
            att[r['attempts']] += 1
            if r['status'] == 'failed':
                kinds[kind_of(r.get('reason') or '')] += 1
    per_shard[p.split('_')[-1].split('.')[0]] = dict(ps)

print('status      :', dict(st))
print('attempts    :', dict(sorted(att.items())))
print('failed kinds:', dict(kinds.most_common()))
llm = st.get('ok', 0) + st.get('failed', 0)
if llm:
    print(f'eventual success: {st.get("ok", 0)}/{llm} '
          f'= {st.get("ok", 0) / llm:.1%}')
print('per shard   :', {k: v for k, v in per_shard.items() if v})
