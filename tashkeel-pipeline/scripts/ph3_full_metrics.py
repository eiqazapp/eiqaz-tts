#!/usr/bin/env python3
"""ph3_full_metrics.py — post-run metrics package (§م/§ق gates).

1) Status / attempts / fixes histograms (per split too)
2) Diacritic density stats
3) Token ceiling gate (§ق-7): count tokens on the tashkeel'd text with the
   exact Prep-kernel function; compare with catt-recorded n_tokens; list any
   unit > 160.
4) Quarantine + letter-fix accounting.

Usage: python3 scripts/ph3_full_metrics.py
Output: work/ph3_full_metrics.json (+ stdout summary)
"""
import json
import statistics
import sys

sys.path.insert(0, '/home/z/my-project/scripts')
from step0_counter import n_tokens_of  # noqa: E402  (needs tts_arabic_pkg)

BASE = '/home/z/my-project/work'

with open(f'{BASE}/ph2_input_full.json', encoding='utf-8') as f:
    inp = {u['id']: u for u in json.load(f)}
records = []
with open(f'{BASE}/ph3_full_output.jsonl', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))

STRIP = '\u064B-\u0652\u0653-\u0655\u0670'
import re  # noqa: E402
STRIP_RE = re.compile(f'[{STRIP}]')

status_counts = {}
split_status = {}
attempts_hist = {}
fix_units = {}
fix_total = {}
density = []
letters_restored = 0
ok_records = []
for r in records:
    st = r.get('status')
    status_counts[st] = status_counts.get(st, 0) + 1
    sp = inp[r['id']].get('split', '?')
    split_status.setdefault(sp, {})
    split_status[sp][st] = split_status[sp].get(st, 0) + 1
    if st == 'ok':
        ok_records.append(r)
        attempts_hist[r.get('attempts', 0)] = \
            attempts_hist.get(r.get('attempts', 0), 0) + 1
        words = [w for w in r['out'].split()
                 if re.search(r'[\u0621-\u064A]', w)]
        marks = len(STRIP_RE.findall(r['out']))
        density.append(marks / max(1, len(words)))
    for k, v in (r.get('fixes') or {}).items():
        n = v if isinstance(v, (int, float)) else len(v)
        fix_units[k] = fix_units.get(k, 0) + 1
        fix_total[k] = fix_total.get(k, 0) + int(n)
    if r.get('letter_fix'):
        letters_restored += 1

# ---- token gate (§ق-7) on all ok units ----
tok_results = []
over160 = []
ratios = []
for r in ok_records:
    n = n_tokens_of(r['out'])
    rec = inp[r['id']].get('n_tokens_recorded')
    ratio = (n / rec) if rec else None
    if ratio is not None:
        ratios.append(ratio)
    row = {'id': r['id'], 'n_tokens_mine': n,
           'n_tokens_catt_recorded': rec, 'ratio': ratio,
           'split': inp[r['id']].get('split')}
    tok_results.append(row)
    if n > 160:
        over160.append(row)

metrics = {
    'engine': 'ph3_engine.mjs v5 (guide v2.1 — approved 2026-09-30)',
    'n_input': len(inp),
    'n_records': len(records),
    'status_counts': status_counts,
    'status_counts_by_split': split_status,
    'attempts_histogram_ok_units': dict(sorted(attempts_hist.items())),
    'fixes': {'units_touched': fix_units, 'total_operations': fix_total},
    'letter_fix_units': letters_restored,
    'density_per_ok_unit': {
        'n': len(density),
        'median': round(statistics.median(density), 3),
        'mean': round(statistics.mean(density), 3),
        'p10': round(statistics.quantiles(density, n=10)[0], 3),
        'p90': round(statistics.quantiles(density, n=10)[8], 3),
        'min': round(min(density), 3), 'max': round(max(density), 3),
    } if density else None,
    'token_gate': {
        'rule': 'n_tokens(tashkeel output) <= 160 (§ق-7)',
        'n_measured': len(tok_results),
        'over_160': over160,
        'n_over_160': len(over160),
        'ratio_mine_over_catt': {
            'median': round(statistics.median(ratios), 4),
            'mean': round(statistics.mean(ratios), 4),
            'p10': round(statistics.quantiles(ratios, n=10)[0], 4),
            'p90': round(statistics.quantiles(ratios, n=10)[8], 4),
            'min': round(min(ratios), 4), 'max': round(max(ratios), 4),
        } if ratios else None,
    },
}
with open(f'{BASE}/ph3_full_metrics.json', 'w', encoding='utf-8') as f:
    json.dump(metrics, f, ensure_ascii=False, indent=1)
with open(f'{BASE}/ph3_full_tokens.jsonl', 'w', encoding='utf-8') as f:
    for row in tok_results:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')

print('=== STATUS ===', json.dumps(status_counts, ensure_ascii=False))
print('=== BY SPLIT ===', json.dumps(split_status, ensure_ascii=False))
print('=== ATTEMPTS (ok units) ===', dict(sorted(attempts_hist.items())))
print('=== FIXES (units / total ops) ===')
for k in sorted(fix_units):
    print(f'  {k:16s} units={fix_units[k]:6d} ops={fix_total[k]}')
print('=== DENSITY ===', json.dumps(metrics['density_per_ok_unit'],
                                    ensure_ascii=False))
print('=== TOKEN GATE ===')
tg = metrics['token_gate']
print(f"  measured={tg['n_measured']}  over160={tg['n_over_160']}")
if tg['ratio_mine_over_catt']:
    print('  ratio mine/catt:', json.dumps(tg['ratio_mine_over_catt']))
if over160:
    for row in over160:
        print('  OVER160:', json.dumps(row, ensure_ascii=False))
