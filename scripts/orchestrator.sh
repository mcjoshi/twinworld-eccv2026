#!/usr/bin/env bash
# Persistent experiment runner for TwinWorld.
#
# Drains a queue of experiment commands, one line at a time, respecting a
# concurrency cap. Survives the session that started it: it is launched with
# setsid + nohup, writes its own state to disk, and picks up where it left off
# if restarted.
#
# The point is that experiment throughput should not depend on anyone watching.
#
#   queue file    experiments/QUEUE.txt   - one shell command per line
#                                           '#' comments and blanks ignored
#   state         experiments/orchestrator/done.txt    completed command hashes
#                 experiments/orchestrator/running/    one file per live job
#                 experiments/orchestrator/orchestrator.log
#
# Usage:
#   setsid nohup bash scripts/orchestrator.sh > /dev/null 2>&1 &
#   bash scripts/orchestrator.sh --status
#   touch experiments/orchestrator/STOP     # graceful shutdown

set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
ROOT="$PWD"
QUEUE="$ROOT/experiments/QUEUE.txt"
STATE="$ROOT/experiments/orchestrator"
LOG="$STATE/orchestrator.log"
DONE="$STATE/done.txt"
RUNNING="$STATE/running"
STOP="$STATE/STOP"

# Concurrency cap. LightGBM runs single-threaded here by design, but the
# cKDTree queries use every core, so more than a few concurrent jobs thrash:
# on 19 Aug six parallel jobs drove the load average to 280 on 96 cores and
# everything slowed by roughly 3x.
MAX_JOBS="${MAX_JOBS:-3}"
POLL="${POLL:-60}"

mkdir -p "$STATE" "$RUNNING"
touch "$DONE" "$QUEUE"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" >> "$LOG"; }

if [[ "${1:-}" == "--status" ]]; then
    echo "queue:   $(grep -cvE '^\s*(#|$)' "$QUEUE" 2>/dev/null || echo 0) commands"
    echo "done:    $(wc -l < "$DONE") "
    echo "running: $(ls "$RUNNING" 2>/dev/null | wc -l) / $MAX_JOBS"
    ls "$RUNNING" 2>/dev/null | sed 's/^/    /'
    echo "--- last 15 log lines ---"
    tail -15 "$LOG" 2>/dev/null
    exit 0
fi

# Only one orchestrator at a time.
LOCK="$STATE/orchestrator.pid"
if [[ -f "$LOCK" ]] && kill -0 "$(cat "$LOCK")" 2>/dev/null; then
    echo "orchestrator already running as PID $(cat "$LOCK")"; exit 1
fi
echo $$ > "$LOCK"
log "orchestrator started (pid $$, max_jobs=$MAX_JOBS)"

reap() {
    for f in "$RUNNING"/*; do
        [[ -e "$f" ]] || continue
        pid=$(cut -d' ' -f1 < "$f")
        if ! kill -0 "$pid" 2>/dev/null; then
            tag=$(basename "$f")
            log "finished: $tag"
            cut -d' ' -f2- < "$f" >> "$DONE"
            rm -f "$f"
        fi
    done
}

while true; do
    [[ -f "$STOP" ]] && { log "STOP file present, exiting"; rm -f "$LOCK"; exit 0; }
    reap

    n=$(ls "$RUNNING" 2>/dev/null | wc -l)
    if (( n < MAX_JOBS )); then
        # first queued command whose hash is not in done.txt and not running
        while IFS= read -r cmd; do
            [[ -z "$cmd" || "$cmd" =~ ^[[:space:]]*# ]] && continue
            h=$(printf '%s' "$cmd" | md5sum | cut -c1-12)
            grep -qF "$h" "$DONE" 2>/dev/null && continue
            ls "$RUNNING" 2>/dev/null | grep -q "$h" && continue

            ts=$(date -u +%Y%m%d-%H%M%S)
            slug=$(printf '%s' "$cmd" | grep -oE -- '--tag [A-Za-z0-9_.-]+' | head -1 | cut -d' ' -f2)
            [[ -z "$slug" ]] && slug="job"
            out="$ROOT/logs/orch-${ts}-${slug}.log"
            log "launching [$h] $slug -> $(basename "$out")"
            (
                cd "$ROOT" || exit 1
                # shellcheck disable=SC1091
                source env.sh 2>/dev/null
                eval "$cmd"
            ) > "$out" 2>&1 &
            echo "$! $h $slug" > "$RUNNING/$h"
            break
        done < "$QUEUE"
    fi
    sleep "$POLL"
done
