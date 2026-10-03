#!/usr/bin/env bash
# groq_relay.sh — Groq relay for the ph3 tashkeel engine (GitHub Actions runner).
#
# WHY THIS EXISTS: the workspace egress IP is in Hong Kong and Groq 403-blocks
# the HK region at the Cloudflare edge (cf-ray ...-HKG). The same API key
# works from GitHub Actions runners (US). This script is the Actions-side
# orchestrator: probe candidate models, then run up to GROQ_MAX_MODELS engine
# passes — between passes only ok/quarantined records are kept in the output,
# so units the previous model failed are re-processed by the next model.
#
# Usage: bash groq_relay.sh <input.json> <output.jsonl>
# Env:   GROQ_API_KEY (required), GROQ_CANDIDATES (optional override),
#        GROQ_MAX_MODELS (default 3), PH3_WORK_DIR (engine workdir).
# Engine semantics: append-mode JSONL + resume-by-id (byte-identical
# validators/prompt sha 60130f6b…; only the HTTP client is Groq).
set -uo pipefail

INPUT="$1"
OUTPUT="$2"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CANDIDATES="${GROQ_CANDIDATES:-moonshotai/kimi-k2-instruct openai/gpt-oss-120b llama-3.3-70b-versatile qwen/qwen3-32b llama-3.1-8b-instant}"
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

# ---- 1) probe candidates with a tiny completion ----
WORKING=()
for m in $CANDIDATES; do
  code=$(curl -s -o /tmp/groq_probe.json -w '%{http_code}' -m 60 \
    https://api.groq.com/openai/v1/chat/completions \
    -H "Authorization: Bearer $GROQ_API_KEY" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$m\",\"messages\":[{\"role\":\"user\",\"content\":\"reply with the single word: ok\"}],\"max_tokens\":8}")
  if [ "$code" = "200" ]; then
    WORKING+=("$m")
    echo "[probe] $m -> HTTP 200 OK"
  else
    echo "[probe] $m -> HTTP $code $(head -c 160 /tmp/groq_probe.json 2>/dev/null)"
  fi
done

if [ "${#WORKING[@]}" -eq 0 ]; then
  echo "RELAY_RESULT ok=0 failed=0 error=no_working_model"
  echo "FATAL: no working Groq model — check key/quota/catalogue above"
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
