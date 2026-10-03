#!/usr/bin/env python3
"""ph3_full_consistency.py — post-run consistency analysis (§م) + «اوي»
inventory (user decision 2026-09-30: NO Tier A rule; record all forms now).

1) Top-50 most frequent word skeletons across all ok outputs; for each, the
   distribution of diacritized forms and a consistency ratio.
2) «اوي» family inventory: every form produced by the engine, with counts and
   sample unit ids (اوي / اوية / اوى skeletons).
3) Unification PROPOSAL by majority (guide §م) — written as a proposal file
   only; nothing is rewritten in the canonical output without the user's
   explicit approval (اوي explicitly deferred to post-run review).

Usage: python3 scripts/ph3_full_consistency.py
Outputs: work/ph3_full_consistency.json, work/ph3_full_unify_proposal.json
"""
import json
import re
from collections import Counter, defaultdict

BASE = '/home/z/my-project/work'

STRIP_RE = re.compile('[\u064B-\u0652\u0653-\u0655\u0670]')
AR = re.compile(r'[\u0621-\u064A]')
# edge punctuation (incl. converted ASCII + leftovers) — not part of skeleton
EDGE = re.compile(r'^[^\u0621-\u064A\u0640]+|[^\u0621-\u064A\u0640]+$')

records = []
with open(f'{BASE}/ph3_full_output.jsonl', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))

form_counts = defaultdict(Counter)   # skeleton -> {form: count}
form_ids = defaultdict(lambda: defaultdict(list))  # skeleton -> form -> [ids]
n_tokens_total = 0
for r in records:
    if r.get('status') != 'ok':
        continue
    for w in r['out'].split():
        if not AR.search(w):
            continue
        n_tokens_total += 1
        form = EDGE.sub('', w)
        if not form:
            continue
        skel = STRIP_RE.sub('', form)
        if not skel:
            continue
        form_counts[skel][form] += 1
        if len(form_ids[skel][form]) < 5:
            form_ids[skel][form].append(r['id'])

skeleton_totals = Counter(
    {s: sum(c.values()) for s, c in form_counts.items()})
top50 = skeleton_totals.most_common(50)

def entry(skel, cnt):
    forms = form_counts[skel].most_common()
    total = sum(c for _, c in forms)
    return {
        'skeleton': skel, 'n_positions': total,
        'n_forms': len(forms),
        'consistency': round(forms[0][1] / total, 4),
        'forms': [
            {'form': f, 'count': c, 'pct': round(c / total, 4),
             'sample_ids': form_ids[skel][f]}
            for f, c in forms
        ],
    }

top50_entries = [entry(s, c) for s, c in top50]

# ---- «اوي» family inventory (user decision: record ALL forms) ----
AWI_SKELETONS = ['اوي', 'اوية', 'اوى', 'اويه']
awi_present = {s: entry(s, 0) for s in AWI_SKELETONS if s in form_counts}

# ---- unification proposal (majority) — top-50 words with >1 form ----
proposal = []
for e in top50_entries:
    if e['n_forms'] > 1:
        maj = e['forms'][0]
        outliers = e['forms'][1:]
        affected_positions = sum(f['count'] for f in outliers)
        affected_ids = sorted({
            i for f in outliers for i in f['sample_ids']})
        proposal.append({
            'skeleton': e['skeleton'],
            'majority_form': maj['form'],
            'majority_count': maj['count'],
            'outlier_forms': [
                {'form': f['form'], 'count': f['count'],
                 'sample_ids': f['sample_ids']} for f in outliers],
            'n_affected_positions': affected_positions,
            'n_affected_ids_sampled': len(affected_ids),
            'affected_ids_sample': affected_ids[:20],
        })

n_consistent = sum(1 for e in top50_entries if e['n_forms'] == 1)
low = [e for e in top50_entries if e['consistency'] < 0.95]

out = {
    'n_ok_units': sum(1 for r in records if r.get('status') == 'ok'),
    'n_word_positions': n_tokens_total,
    'n_distinct_skeletons': len(form_counts),
    'top50': top50_entries,
    'top50_summary': {
        'fully_consistent': n_consistent,
        'multi_form': 50 - n_consistent,
        'consistency_below_95pct': [
            {'skeleton': e['skeleton'], 'consistency': e['consistency'],
             'n_forms': e['n_forms'], 'n_positions': e['n_positions']}
            for e in low],
    },
    'awi_family_inventory': {
        'decision': 'NO Tier A rule for اوي (user, 2026-09-30) — all engine '
                    'forms recorded for post-run consistency analysis',
        'skeletons_present': awi_present,
        'skeletons_absent': [s for s in AWI_SKELETONS if s not in form_counts],
    },
}
with open(f'{BASE}/ph3_full_consistency.json', 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=1)

with open(f'{BASE}/ph3_full_unify_proposal.json', 'w', encoding='utf-8') as f:
    json.dump({
        'rule': 'guide §م: unify stragglers by majority + document affected '
                'words/positions — PROPOSAL ONLY, applied after user approval',
        'n_words_with_outliers': len(proposal),
        'n_affected_positions_total': sum(p['n_affected_positions']
                                          for p in proposal),
        'proposals': proposal,
    }, f, ensure_ascii=False, indent=1)

print(f"ok units={out['n_ok_units']}  word positions={n_tokens_total:,}  "
      f"distinct skeletons={len(form_counts):,}")
print(f"top-50: fully consistent={n_consistent}/50  "
      f"consistency<95%: {len(low)}")
for e in low:
    print(f"  LOW {e['skeleton']}: {e['consistency']:.2%} "
          f"({e['n_forms']} forms, {e['n_positions']} pos)")
print('=== اوي FAMILY (all engine forms) ===')
for s, e in awi_present.items():
    print(f"  {s}: {e['n_positions']} positions, {e['n_forms']} forms")
    for f2 in e['forms']:
        print(f"     {f2['count']:5d} × {f2['form']}")
print(f"unify proposal: {len(proposal)} words, "
      f"{sum(p['n_affected_positions'] for p in proposal)} affected positions")
