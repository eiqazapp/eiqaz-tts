#!/bin/bash
# ph3_smoke2_v52_run.sh — v5.2 small-batch validation run on the APPROVED
# smoke sample (same input, same first-pass batch=2 as the approved run;
# only the retry mechanism differs). Fires only when API quota is available.
# Then builds the comparison report. NO full-run launch happens here.
cd /home/z/my-project || exit 1
echo "=== v5.2 smoke validation run ($(date '+%F %T')) ==="
echo "config: first-pass batch=2 (parity with approved run), retry-batch=5"
RETRY_BATCH=5 node scripts/ph3_engine_v52.mjs \
  work/ph3_smoke2_input.json \
  work/ph3_smoke2_v52_output.jsonl 2 || exit 1
echo
echo "=== comparison report vs approved smoke ==="
python3 scripts/ph3_smoke2_v52_report.py
