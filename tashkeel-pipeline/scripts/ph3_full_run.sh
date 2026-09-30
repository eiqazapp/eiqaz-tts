#!/bin/bash
# ph3_full_run.sh v2 — worker-pool orchestrator over the 16 full-run shards.
# Engine (scripts/ph3_engine.mjs v5) NOT modified — pure ops.
# - Atomic claim via mkdir locks; multiple orchestrator instances are safe.
# - Stale-lock sweeping: a lock is stale if (no rc file OR rc!=0) AND older
#   than GRACE seconds AND no live engine process on that shard. Sweep frees
#   shards orphaned by killed workers (e.g. tool process-group kills).
# - Resume-safe: the engine itself skips already-done ids from its JSONL.
# Launch detached (survives parent shell death):
#   setsid nohup bash scripts/ph3_full_run.sh [N_WORKERS] \
#     > work/full_logs/orch_main.log 2>&1 < /dev/null &
cd /home/z/my-project || exit 1
N_WORKERS=${1:-8}
SHARDS=16
BASE=work
LOGDIR=$BASE/full_logs
SHARDDIR=$BASE/full_shards
BATCH=${BATCH:-50}
GRACE=${GRACE:-300}
mkdir -p "$LOGDIR" "$SHARDDIR"
date "+%F %T orchestrator start workers=$N_WORKERS batch=$BATCH grace=$GRACE" \
  >> "$LOGDIR/orchestrator.log"

sweep_stale() {
  for i in $(seq 0 $((SHARDS - 1))); do
    local L="$SHARDDIR/lock_$i"
    [ -d "$L" ] || continue
    if [ -f "$L/rc" ] && [ "$(cat "$L/rc" 2>/dev/null)" = "0" ]; then
      continue  # completed OK — stays locked forever
    fi
    local age=$(( $(date +%s) - $(stat -c %Y "$L" 2>/dev/null || echo 9999999999) ))
    [ "$age" -lt "$GRACE" ] && continue
    if pgrep -f "ph3_full_shard_$i\.json " >/dev/null 2>&1; then
      continue  # a live engine is working on it
    fi
    rm -rf "$L"
    date "+%F %T swept stale lock $i (age ${age}s)" >> "$LOGDIR/orchestrator.log"
  done
}

claim_next() {
  for i in $(seq 0 $((SHARDS - 1))); do
    if mkdir "$SHARDDIR/lock_$i" 2>/dev/null; then
      echo "$i"
      return 0
    fi
  done
  return 1
}

worker() {
  local wid=$1
  while true; do
    sweep_stale
    local s
    s=$(claim_next) || {
      sleep 20            # give a running engine a moment, sweep once more
      sweep_stale
      s=$(claim_next) || break
    }
    date "+%F %T w$wid -> shard $s" >> "$LOGDIR/orchestrator.log"
    node scripts/ph3_engine.mjs \
      "$BASE/ph3_full_shard_$s.json" \
      "$BASE/ph3_full_shard_$s.jsonl" \
      "$BATCH" >> "$LOGDIR/shard_$s.log" 2>&1
    local rc=$?
    echo "$rc" > "$SHARDDIR/lock_$s/rc"
    date "+%F %T w$wid <- shard $s rc=$rc" >> "$LOGDIR/orchestrator.log"
  done
  date "+%F %T w$wid exit (nothing left)" >> "$LOGDIR/orchestrator.log"
}

pids=()
for w in $(seq 1 "$N_WORKERS"); do
  worker "$w" &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
date "+%F %T ALL CLAIMED SHARDS COMPLETE" >> "$LOGDIR/orchestrator.log"
