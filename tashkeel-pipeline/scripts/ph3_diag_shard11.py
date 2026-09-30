#!/usr/bin/env python3
"""ph3_diag_shard11.py — decompose shard 11 (batch=25 solo test) outcomes."""
import json


def kind_of(rs):
    if '429' in rs:
        return '429-API'
    if 'الهيكل' in rs:
        return 'SKELETON'
    if 'تنوين' in rs:
        return 'TANWEEN'
    if 'المربوطة' in rs:
        return 'TAA'
    if 'الحركات' in rs or 'خالٍ' in rs:
        return 'DENSITY'
    if 'التحليل' in rs:
        return 'PARSE'
    return 'OTHER'


recs = [json.loads(l) for l in open(
    '/home/z/my-project/work/ph3_full_shard_11.jsonl', encoding='utf-8')]
for r in recs:
    if r['status'] == 'quarantined':
        continue
    rs = r.get('reason') or 'OK'
    print(f"{r['id']} {r['status']:7s} att={r['attempts']} "
          f"{kind_of(rs):8s} | {rs[:95]}")
