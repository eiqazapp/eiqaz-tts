# -*- coding: utf-8 -*-
"""Phase 3 — smoke report: catt_eo, token counts, density stats."""
import json
import re
import sys

sys.path.insert(0, '/home/z/my-project/scripts')
from step0_counter import n_tokens_of, get_catt, catt_n_tokens

TASH = re.compile(r'[\u064B-\u0652\u0653-\u0655\u0670]')

inp = {r['id']: r for r in json.load(
    open('/home/z/my-project/work/ph3_smoke_input.json', encoding='utf-8'))}
outs = [json.loads(l) for l in open(
    '/home/z/my-project/work/ph3_smoke_output.jsonl', encoding='utf-8')]

catt = get_catt()

rows = []
for r in outs:
    raw = inp[r['id']]['text']
    rec = inp[r['id']]['n_tokens_recorded']
    catt_n, catt_voc = catt_n_tokens(catt, raw)
    mine_n = n_tokens_of(r['out']) if r['out'] else None
    words = [w for w in r['out'].split()
             if any('\u0621' <= c <= '\u064A' for c in w)] if r['out'] else []
    density = len(TASH.findall(r['out'])) / max(1, len(words)) if r['out'] else 0
    rows.append({
        'id': r['id'], 'status': r['status'], 'attempts': r['attempts'],
        'raw': raw, 'catt_eo': catt_voc, 'out': r['out'],
        'n_tok_recorded_catt': rec, 'n_tok_catt_recheck': catt_n,
        'n_tok_mine': mine_n, 'ratio_mine_vs_catt': (
            round(mine_n / catt_n, 3) if mine_n and catt_n else None),
        'density_marks_per_word': round(density, 2),
        'over_160': (mine_n > 160) if mine_n else None,
    })

json.dump(rows, open('/home/z/my-project/work/ph3_smoke_report.json', 'w',
                     encoding='utf-8'), ensure_ascii=False, indent=1)

ok = [r for r in rows if r['status'] == 'ok']
print(f'ok={len(ok)}/{len(rows)}')
print(f"{'id':14} {'st':5} {'att':4} {'catt':>5} {'mine':>5} {'ratio':>6} "
      f"{'dens':>5} {'>160':>5}")
for r in rows:
    print(f"{r['id']:14} {r['status']:5} {r['attempts']:<4} "
          f"{r['n_tok_recorded_catt']:>5} "
          f"{str(r['n_tok_mine']):>5} "
          f"{str(r['ratio_mine_vs_catt']):>6} "
          f"{r['density_marks_per_word']:>5} "
          f"{str(r['over_160']):>5}")
if ok:
    rs = [r['ratio_mine_vs_catt'] for r in ok]
    ds = [r['density_marks_per_word'] for r in ok]
    print(f'\nratio mine/catt: median={sorted(rs)[len(rs)//2]:.3f} '
          f'range={min(rs):.3f}-{max(rs):.3f}')
    print(f'density: median={sorted(ds)[len(ds)//2]:.2f} '
          f'range={min(ds):.2f}-{max(ds):.2f}  (my-manual=3.74)')
    print('over-160:', sum(1 for r in ok if r['over_160']))
