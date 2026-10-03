# -*- coding: utf-8 -*-
"""ph3_smoke2_v52_report.py — v5.2 smoke test report + comparison against the
APPROVED v5 smoke sample (work/ph3_smoke2_output.jsonl).

Compares per unit: status, attempts, output text (exact/valid-diff),
density, token ratio vs catt, over-160. Aggregates v5.2 token accounting
(measured usage when the backend exposes it; char-based estimate otherwise)
and reconstructs the v5.1-equivalent call cost from the approved run's
recorded attempts for an apples-to-apples economy comparison.

Usage:
  python3 scripts/ph3_smoke2_v52_report.py            # after the v5.2 run
"""
import json
import re
import sys

sys.path.insert(0, '/home/z/my-project/scripts')
from step0_counter import n_tokens_of, get_catt, catt_n_tokens  # noqa: E402

TASH = re.compile(r'[\u064B-\u0652\u0653-\u0655\u0670]')
BASE = '/home/z/my-project/work'

inp = {r['id']: r for r in json.load(
    open(f'{BASE}/ph3_smoke2_input.json', encoding='utf-8'))}
approved = {r['id']: r for r in (
    json.loads(l) for l in open(f'{BASE}/ph3_smoke2_output.jsonl',
                                encoding='utf-8') if l.strip())}
v52 = [json.loads(l) for l in open(f'{BASE}/ph3_smoke2_v52_output.jsonl',
                                   encoding='utf-8') if l.strip()]
meta = json.load(open(f'{BASE}/ph3_smoke2_v52_output_meta.json',
                      encoding='utf-8'))
catt = get_catt()

rows = []
for r in v52:
    uid = r['id']
    if r['status'] == 'quarantined':
        rows.append({'id': uid, 'status': 'quarantined'})
        continue
    raw = r['raw']
    catt_n, _ = catt_n_tokens(catt, raw)
    mine_n = n_tokens_of(r['out']) if r['out'] else None
    words = [w for w in r['out'].split()
             if any('\u0621' <= c <= '\u064A' for c in w)] if r['out'] else []
    density = len(TASH.findall(r['out'])) / max(1, len(words)) if r['out'] else 0
    a = approved.get(uid, {})
    rows.append({
        'id': uid, 'status': r['status'], 'attempts': r['attempts'],
        'approved_status': a.get('status'),
        'approved_attempts': a.get('attempts'),
        'out': r['out'], 'approved_out': a.get('out'),
        'text_identical_to_approved': r['out'] == a.get('out'),
        'same_valid_status': r['status'] == a.get('status'),
        'n_tok_catt': catt_n, 'n_tok_mine': mine_n,
        'ratio_mine_vs_catt': round(mine_n / catt_n, 3) if mine_n and catt_n else None,
        'density': round(density, 2),
        'over_160': (mine_n > 160) if mine_n else None,
        'fixes': r.get('fixes', {}),
    })

json.dump({'rows': rows, 'meta': meta}, open(
    f'{BASE}/ph3_smoke2_v52_report.json', 'w', encoding='utf-8'),
    ensure_ascii=False, indent=1)

# ---------------- console summary ----------------
llm = [r for r in rows if r.get('status') in ('ok', 'failed')]
okr = [r for r in llm if r['status'] == 'ok']
print(f"v5.2 smoke: ok={len(okr)}/{len(llm)} | approved run: "
      f"ok={sum(1 for r in llm if r.get('approved_status')=='ok')}/{len(llm)}")
print(f"{'id':14} {'st':7} {'att':>4} {'v5att':>5} {'same':>5} {'ident':>6} "
      f"{'catt':>5} {'mine':>5} {'ratio':>6} {'dens':>5} {'>160':>5}")
for r in llm:
    print(f"{r['id']:14} {r['status']:7} {r['attempts']:>4} "
          f"{str(r.get('approved_attempts')):>5} "
          f"{str(r.get('same_valid_status')):>5} "
          f"{str(r.get('text_identical_to_approved')):>6} "
          f"{r['n_tok_catt']:>5} {str(r['n_tok_mine']):>5} "
          f"{str(r['ratio_mine_vs_catt']):>6} {r['density']:>5} "
          f"{str(r['over_160']):>5}")
if okr:
    rs = sorted(r['ratio_mine_vs_catt'] for r in okr if r['ratio_mine_vs_catt'])
    ds = sorted(r['density'] for r in okr)
    print(f"\nv5.2  ratio mine/catt median={rs[len(rs)//2]:.3f} "
          f"range={rs[0]:.3f}-{rs[-1]:.3f}")
    print(f"v5.2  density median={ds[len(ds)//2]:.2f} range={ds[0]:.2f}-{ds[-1]:.2f}")
    print(f"v5.2  over-160: {sum(1 for r in okr if r['over_160'])}")
    print(f"approved ratio median was 0.877, density median 3.73")
    ident = sum(1 for r in okr if r.get('text_identical_to_approved'))
    same_st = sum(1 for r in llm if r.get('same_valid_status'))
    print(f"status parity with approved: {same_st}/{len(llm)} | "
          f"byte-identical outputs: {ident}/{len(okr)}")

# token economy
tok = meta.get('token_stats', {})
calls52 = tok.get('first_calls', 0) + tok.get('retry_calls', 0)
att_app = sum(r.get('approved_attempts') or 0 for r in llm)
n_first_app = -(-len(llm) // 2)  # approved run used batch=2
calls_app = n_first_app + (att_app - len(llm))
print(f"\n--- token economy (this smoke input, {len(llm)} units) ---")
if tok.get('usage_present'):
    tot = tok['prompt_tokens'] + tok['completion_tokens']
    print(f"v5.2 MEASURED: {calls52} calls ({tok.get('first_calls')} first + "
          f"{tok.get('retry_calls')} retry) = {tot:,} tokens "
          f"({tot/len(llm):,.0f}/unit)")
    per_call = tot / calls52
else:
    chars = tok.get('chars', {})
    sys_c = chars.get('system', 25398)
    per_call_est = (sys_c + 800) / 2.3  # ~2.3 chars/token for Arabic
    tot = calls52 * per_call_est
    print(f"v5.2 ESTIMATE (usage not exposed): {calls52} calls "
          f"({tok.get('first_calls')} first + {tok.get('retry_calls')} retry) "
          f"≈ {tot:,.0f} tokens ({tot/len(llm):,.0f}/unit)")
    per_call = per_call_est
print(f"v5.1-equivalent (approved attempts={att_app} -> {calls_app} calls "
      f"@ {per_call:,.0f}/call) ≈ {calls_app * per_call:,.0f} tokens "
      f"({calls_app * per_call / len(llm):,.0f}/unit)")
if calls_app:
    print(f"economy factor: {calls_app / max(1, calls52):.2f}x fewer calls "
          f"on this workload")
