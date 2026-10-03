#!/usr/bin/env python3
"""ph4_build_dataset.py — assemble the text-only Kaggle dataset package
`ahmedyaseen86/niletts-4h-tashkeel-ai-v1` from the canonical full-run output.

Inclusion rule (guide §م): ONLY status=ok units. Failed units go to a
review-list (never into the dataset before the user's explicit decision).
Quarantined units are excluded by design (§ل-5). Audio is never touched —
this package contains TEXT + provenance only.

Outputs (under work/ph4_dataset/):
  transcripts.csv      utt,text,split          (tashkeel'd, ASCII punct)
  transcripts.jsonl    same records as JSONL
  review_needed.json   failed + quarantined ids with reasons (NOT in dataset)
  README.md            dataset card (provenance, engine, gates, counts)
  DATASET_META.json    machine-readable meta (hashes, decisions, metrics)

Usage: python3 scripts/ph4_build_dataset.py [--allow-partial]
"""
import glob
import hashlib
import json
import statistics
import sys

BASE = '/home/z/my-project/work'
OUT = f'{BASE}/ph4_dataset'
ALLOW_PARTIAL = '--allow-partial' in sys.argv

with open(f'{BASE}/ph2_input_full.json', encoding='utf-8') as f:
    inp = {u['id']: u for u in json.load(f)}

records = []
with open(f'{BASE}/ph3_full_output.jsonl', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))

ok = [r for r in records if r.get('status') == 'ok']
failed = [r for r in records if r.get('status') == 'failed']
quarantined = [r for r in records if r.get('status') == 'quarantined']
missing = [i for i in inp if i not in {r['id'] for r in records}]

if missing and not ALLOW_PARTIAL:
    print(f'REFUSING to build: {len(missing)} units have no record yet '
          f'(full run incomplete). Use --allow-partial to override.')
    raise SystemExit(1)

import os
os.makedirs(OUT, exist_ok=True)

with open(f'{OUT}/transcripts.csv', 'w', encoding='utf-8') as fc, \
     open(f'{OUT}/transcripts.jsonl', 'w', encoding='utf-8') as fj:
    fc.write('utt,text,split\n')
    for r in ok:
        u = inp[r['id']]
        t = r['out'].replace('"', '""')
        fc.write(f"{r['id']},\"{t}\",{u.get('split', '')}\n")
        fj.write(json.dumps({'utt': r['id'], 'text': r['out'],
                             'split': u.get('split', '')},
                            ensure_ascii=False) + '\n')

with open(f'{OUT}/review_needed.json', 'w', encoding='utf-8') as f:
    json.dump({
        'rule': 'NOT in dataset — awaiting explicit user decision (§م)',
        'failed': [{'id': r['id'], 'attempts': r.get('attempts'),
                    'reason': r.get('reason')} for r in failed],
        'quarantined': [{'id': r['id'],
                         'reason': r.get('reason')} for r in quarantined],
        'missing_no_record': missing,
    }, f, ensure_ascii=False, indent=1)

# quick density stat for the card
import re
STRIP = re.compile('[\u064B-\u0652\u0653-\u0655\u0670]')
dens = []
for r in ok:
    words = [w for w in r['out'].split() if re.search(r'[\u0621-\u064A]', w)]
    dens.append(len(STRIP.findall(r['out'])) / max(1, len(words)))

n_train = sum(1 for r in ok if inp[r['id']].get('split') == 'train')
n_eval = sum(1 for r in ok if inp[r['id']].get('split') == 'eval')

readme = f"""# niletts-4h-tashkeel-ai-v1

Egyptian-Arabic tashkeel'd transcripts for the NileTTS 4h corpus, produced by
a rule-validated LLM pipeline (Track C), intended to replace catt_eo output
as the text source for MixerTTS from-scratch training on Kaggle T4.

- Scope: NileTTS 4h train pool units (21,880 total input; this package holds
  the units whose tashkeel passed all engine gates).
- Engine: ph3_engine.mjs v5.1 (guide v2.1, approved 2026-09-30). Gates:
  exact letter/punctuation skeleton, closed-list tanween, taa-marbuta no
  haraka, density floor, deterministic Tier A + rabbena enforcement,
  letter-restore, quarantine skip.
- Punctuation: Arabic ، ؟ converted to ASCII , ? BEFORE tashkeel (the train
  kernel's symbol table only supports . , ? ! as ids 5-8).
- NO audio in this package. No manifest replacement — text only.
- Excluded by design: quarantined units (user-approved 2026-09-30), units
  still failing gates (see review_needed.json), units without records.

## Current build (partial runs are marked)

- input units: {len(inp)}
- included (ok): {len(ok)}  (train {n_train} / eval {n_eval})
- failed (under review): {len(failed)}
- quarantined: {len(quarantined)}
- no record yet: {len(missing)}
- diacritic density median: {statistics.median(dens):.2f} marks/word

Regeneration: scripts/ + work/ artifacts are archived alongside; the engine
is deterministic post-LLM (maps + fixes logged per unit in the JSONL).
"""
with open(f'{OUT}/README.md', 'w', encoding='utf-8') as f:
    f.write(readme)

meta = {
    'dataset': 'ahmedyaseen86/niletts-4h-tashkeel-ai-v1',
    'visibility': 'private (text-only, no audio)',
    'engine': 'ph3_engine.mjs v5.1 (guide v2.1 — approved 2026-09-30; '
              'tanween ف-prefix bugfix documented in engine header)',
    'n_input': len(inp), 'n_ok': len(ok), 'n_failed': len(failed),
    'n_quarantined': len(quarantined), 'n_missing': len(missing),
    'partial_build': bool(missing),
    'transcripts_csv_sha256': hashlib.sha256(
        open(f'{OUT}/transcripts.csv', 'rb').read()).hexdigest(),
    'transcripts_jsonl_sha256': hashlib.sha256(
        open(f'{OUT}/transcripts.jsonl', 'rb').read()).hexdigest(),
    'inclusion_rule': 'status==ok only (§م); failed->review; quarantined->excluded',
}
with open(f'{OUT}/DATASET_META.json', 'w', encoding='utf-8') as f:
    json.dump(meta, f, ensure_ascii=False, indent=1)

print(f'dataset package built at {OUT}')
print(f"  ok={len(ok)} (train {n_train}/eval {n_eval})  "
      f"failed={len(failed)}  quarantined={len(quarantined)}  "
      f"missing={len(missing)}  partial={bool(missing)}")
