#!/usr/bin/env bash
# groq_relay.sh — Groq relay for the ph3 tashkeel engine (remote US runner).
#
# WHY THIS EXISTS: the workspace egress IP is in Hong Kong and Groq 403-blocks
# the HK region at the Cloudflare edge (cf-ray ...-HKG). The same API key
# works from US runners (Kaggle kernel / GitHub Actions). This script is the
# runner-side orchestrator: probe candidate models with an ENGINE-SHAPED
# request (small probes lie — they pass models that then hard-fail 413 on
# the frozen 12k-token system prompt), then run up to GROQ_MAX_MODELS engine
# passes — between passes only ok/quarantined records are kept in the output,
# so units the previous model failed are re-processed by the next model.
#
# Usage: bash groq_relay.sh <input.json> <output.jsonl>
# Env:   GROQ_API_KEY (required), GROQ_CANDIDATES (optional override),
#        GROQ_MAX_MODELS (default 3), GROQ_PASS_TIMEOUT (default 1500s),
#        PH3_WORK_DIR (engine workdir with guide/lists).
# Engine semantics: append-mode JSONL + resume-by-id (byte-identical
# validators/prompt sha 60130f6b…; only the HTTP client is Groq).
set -uo pipefail

INPUT="$1"
OUTPUT="$2"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# 2026-10 catalogue for this key (11 models visible; chat-capable only,
# Arabic strength first — allam-2-7b is Aramco's Arabic-centric model):
CANDIDATES="${GROQ_CANDIDATES:-allam-2-7b qwen/qwen3.8-27b openai/gpt-oss-20b openai/gpt-oss-120b}"
MAX_MODELS="${GROQ_MAX_MODELS:-3}"
PASS_TIMEOUT="${GROQ_PASS_TIMEOUT:-1500}"   # seconds per engine pass

echo "== groq relay: input=$INPUT output=$OUTPUT =="
echo "candidates: $CANDIDATES"

# ---- 0) diagnostics: key reachability + available model catalogue ----
echo "== [diag] GET /models =="
curl -s -m 30 https://api.groq.com/openai/v1/models \
  -H "Authorization: Bearer $GROQ_API_KEY" -o /tmp/groq_models.json -w "HTTP %{http_code}\n"
python3 - <<'PY' 2>/dev/null || echo "(models list not parsable)"
import json
try:
    d = json.load(open('/tmp/groq_models.json'))
    ids = sorted(m.get('id', '?') for m in d.get('data', []))
    print(f"[diag] {len(ids)} models visible:")
    for i in ids:
        print('  -', i)
except Exception as e:
    print('[diag] parse error:', e)
PY

# ---- 1) probe candidates with an engine-shaped request ----
echo "== [probe] engine-shaped requests (system 25,398 chars + 25 sentences, max_tokens 4096) =="
PROBE_OUT=$(python3 "$SCRIPT_DIR/probe_groq.py" "$INPUT" $CANDIDATES 2>&1)
echo "$PROBE_OUT"
WORKING=()
while IFS= read -r line; do
  case "$line" in
    *"HTTP 200 USABLE"*)
      m="${line#PROBE }"; m="${m%% *}"
      WORKING+=("$m")
      ;;
  esac
done <<< "$PROBE_OUT"

if [ "${#WORKING[@]}" -eq 0 ]; then
  echo "RELAY_RESULT ok=0 failed=0 error=no_working_model"
  echo "FATAL: no working Groq model for the engine-shaped request — see TPM limits above"
  exit 1
fi
echo "[probe] working models (priority order): ${WORKING[*]}"

# ---- 2) engine passes ----
TOTAL=$(node -e "console.log(JSON.parse(require('fs').readFileSync('$INPUT','utf8')).length)")
echo "input units: $TOTAL"

PASS=0
for m in "${WORKING[@]:0:$MAX_MODELS}"; do
  PASS=$((PASS + 1))
  if [ "$PASS" -gt 1 ]; then
    # keep only settled records so failed ids are re-processed by this model
    grep -e '"status":"ok"' -e '"status":"quarantined"' "$OUTPUT" > "$OUTPUT.keep" 2>/dev/null || true
    if [ -s "$OUTPUT.keep" ]; then
      mv "$OUTPUT.keep" "$OUTPUT"
    else
      : > "$OUTPUT"
    fi
    echo "[relay] pass $PASS: reset output to settled records only"
  fi
  echo "== [engine] pass $PASS model=$m =="
  GROQ_MODEL="$m" timeout "$PASS_TIMEOUT" node "$SCRIPT_DIR/ph3_engine_v53.mjs" \
    "$INPUT" "$OUTPUT" 25 || echo "[engine] pass $PASS ended rc=$? (continuing to next model)"
  OK=$(grep -c '"status":"ok"' "$OUTPUT" 2>/dev/null || echo 0)
  echo "[relay] after pass $PASS: ok=$OK / $TOTAL"
  if [ "$OK" -ge "$TOTAL" ]; then
    echo "[relay] all units settled — stopping"
    break
  fi
done

# ---- 3) final summary ----
OK=$(grep -c '"status":"ok"' "$OUTPUT" 2>/dev/null || echo 0)
FAILED=$(grep -c '"status":"failed"' "$OUTPUT" 2>/dev/null || echo 0)
QUAR=$(grep -c '"status":"quarantined"' "$OUTPUT" 2>/dev/null || echo 0)
echo "RELAY_RESULT ok=$OK failed=$FAILED quarantined=$QUAR total_input=$TOTAL"
echo "== groq relay done =="
