#!/bin/bash
# ph3_smoke2_v53_run.sh — v5.3 small-batch validation run on the APPROVED
# smoke sample (same input, same first-pass batch=2 as the approved run;
# retry scheduling differs: 2 batched rounds + v5-style focused individual
# final round). Fires only when API quota is available. Then builds the
# comparison report (v5.3 vs approved v5 vs v5.2). NO full-run launch here.
cd /home/z/my-project || exit 1
echo "=== v5.3 smoke validation run ($(date '+%F %T')) ==="
echo "config: first-pass batch=2 (parity with approved run), retry-batch=5, final individual drain"
RETRY_BATCH=5 node scripts/ph3_engine_v53.mjs \
  work/ph3_smoke2_input.json \
  work/ph3_smoke2_v53_output.jsonl 2 || exit 1
echo
echo "=== comparison report vs approved smoke (and v5.2) ==="
/home/z/.venv/bin/python scripts/ph3_smoke2_v53_report.py
