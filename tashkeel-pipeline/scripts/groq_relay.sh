#!/usr/bin/env bash
# groq_relay.sh — Groq relay for the ph3 tashkeel engine (remote US runner).
#
# WHY: the workspace egress is HK and Groq 403-blocks HK at the Cloudflare
# edge; the key works from US runners (Kaggle kernel / GitHub Actions).
# Groq on_demand tier limits (TPM 7-8k) are BELOW the frozen system prompt
# (~10.5k tokens) for qwen/gpt-oss, so probe_groq.py walks a shape ladder
# (batch25/batch5/single + completion-reserve ladder) and this relay maps
# each usable verdict to engine run parameters (batch size + GROQ_MAX_TOKENS).
# The system prompt, validators, and gates stay byte-identical (sha
# 60130f6b…); only the user-message batch shape and completion reserve vary.
#
# Usage: bash groq_relay.sh <input.json> <output.jsonl>
# Env:   GROQ_API_KEY (required), GROQ_CANDIDATES (optional override),
#        GROQ_MAX_MODELS (default 3), GROQ_PASS_TIMEOUT (default 1500s),
#        PH3_WORK_DIR (engine workdir with guide/lists).
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

# ---- 1) probe candidates with engine-shaped requests (shape ladder) ----
echo "== [probe] shape ladder: system 25,398 chars; batch25/mt4096 -> batch5/mt1024 -> single/mt512|256 =="
PROBE_OUT=$(python3 "$SCRIPT_DIR/probe_groq.py" "$INPUT" $CANDIDATES 2>&1)
echo "$PROBE_OUT"

# VERDICT lines -> parallel arrays: model / batch / max_tokens
declare -a OK_MODELS=() OK_BATCH=() OK_MT=()
while IFS= read -r line; do
  case "$line" in
    "VERDICT "*"USABLE-BATCH25"*)
      m=$(echo "$line" | awk '{print $2}')
      OK_MODELS+=("$m"); OK_BATCH+=(25); OK_MT+=("")
      ;;
    "VERDICT "*"USABLE-BATCH5"*)
      m=$(echo "$line" | awk '{print $2}')
      OK_MODELS+=("$m"); OK_BATCH+=(5); OK_MT+=("1024")
      ;;
    "VERDICT "*"USABLE-SINGLE mt="*)
      m=$(echo "$line" | awk '{print $2}')
      mt=$(echo "$line" | sed 's/.*USABLE-SINGLE mt=//')
      OK_MODELS+=("$m"); OK_BATCH+=(1); OK_MT+=("$mt")
      ;;
  esac
done <<< "$PROBE_OUT"

if [ "${#OK_MODELS[@]}" -eq 0 ]; then
  echo "RELAY_RESULT ok=0 failed=0 error=no_working_model"
  echo "FATAL: no usable model/shape — see limits above (on_demand tier caps)"
  exit 1
fi
echo "[probe] usable models (priority order):"
for i in "${!OK_MODELS[@]}"; do
  echo "  ${OK_MODELS[$i]} batch=${OK_BATCH[$i]} max_tokens=${OK_MT[$i]:-default}"
done

# ---- 2) engine passes ----
TOTAL=$(node -e "console.log(JSON.parse(require('fs').readFileSync('$INPUT','utf8')).length)")
echo "input units: $TOTAL"

PASS=0
N_MODELS=$(( ${#OK_MODELS[@]} < MAX_MODELS ? ${#OK_MODELS[@]} : MAX_MODELS ))
for i in $(seq 0 $((N_MODELS - 1))); do
  m="${OK_MODELS[$i]}"; b="${OK_BATCH[$i]}"; mt="${OK_MT[$i]}"
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
  echo "== [engine] pass $PASS model=$m batch=$b max_tokens=${mt:-default} =="
  if [ -n "$mt" ]; then
    export GROQ_MAX_TOKENS="$mt"
  else
    unset GROQ_MAX_TOKENS
  fi
  GROQ_MODEL="$m" timeout "$PASS_TIMEOUT" node "$SCRIPT_DIR/ph3_engine_v53.mjs" \
    "$INPUT" "$OUTPUT" "$b" || echo "[engine] pass $PASS ended rc=$? (continuing to next model)"
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
