#!/bin/bash
# ph3_full_chunk.sh — run the worker pool for a bounded time slice, then stop.
# For tool environments that kill all descendant processes at command end:
# progress is preserved via per-unit JSONL resume (engine skips done ids).
# Orphaned locks (killed mid-shard, no rc=0) are reaped at chunk start —
# safe because no engine process can be alive between chunks.
# Usage: bash scripts/ph3_full_chunk.sh [CHUNK_SEC=540] [N_WORKERS=8]
cd /home/z/my-project || exit 1
CHUNK_SEC=${1:-540}
N_WORKERS=${2:-2}
BATCH=${BATCH:-3}   # smoke-validated small-batch regime (see worklog 2026-09-30)
export BATCH
LOGDIR=work/full_logs
mkdir -p "$LOGDIR" work/full_shards

# Reap orphaned locks: between chunks no engine process is alive, so any lock
# without rc=0 is orphaned. rc=0 means the shard fully completed.
for i in $(seq 0 15); do
  L=work/full_shards/lock_$i
  if [ -d "$L" ]; then
    if [ -f "$L/rc" ] && [ "$(cat "$L/rc" 2>/dev/null)" = "0" ]; then
      continue
    fi
    rm -rf "$L"
    echo "$(date '+%F %T') reaped orphan lock $i" >> "$LOGDIR/orchestrator.log"
  fi
done

date "+%F %T chunk start workers=$N_WORKERS sec=$CHUNK_SEC batch=$BATCH" >> "$LOGDIR/orchestrator.log"
bash scripts/ph3_full_run.sh "$N_WORKERS" >> "$LOGDIR/orch_main.log" 2>&1 &
POOL=$!
sleep "$CHUNK_SEC"
pkill -f "scripts/ph3_full_run.sh" 2>/dev/null
sleep 1
pkill -f "ph3_engine.mjs" 2>/dev/null
sleep 2
pkill -9 -f "ph3_engine.mjs" 2>/dev/null
wait "$POOL" 2>/dev/null
date "+%F %T chunk end" >> "$LOGDIR/orchestrator.log"

DONE=$(cat work/ph3_full_shard_*.jsonl 2>/dev/null | wc -l)
OKN=$(grep -h '"status": "ok"' work/ph3_full_shard_*.jsonl 2>/dev/null | wc -l)
OKN2=$(grep -h '"status":"ok"' work/ph3_full_shard_*.jsonl 2>/dev/null | wc -l)
echo "chunk summary: recorded=$DONE/21880  ok=$((OKN + OKN2))"
