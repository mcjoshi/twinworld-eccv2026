#!/usr/bin/env bash
# Wait for a running MVS stage to exit, then render the scenes it produced.
#
# Keeps both GPUs busy without needing anything to poll interactively: each GPU
# renders exactly the scenes whose clouds it just built.
#
#   ./scripts/chain_render.sh <gpu> <wait_pid> <run> <scene> [<scene> ...]
set -uo pipefail
cd "$(dirname "$0")/.."
source env.sh

GPU="$1"; shift
WAIT_PID="$1"; shift
RUN="$1"; shift
SCENES="$*"

echo "[chain gpu$GPU] waiting for pid $WAIT_PID (mvs) before rendering: $SCENES"
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 20; done
echo "[chain gpu$GPU] mvs finished at $(date -Is), starting render"

python3 scripts/run_pipeline.py --stage render --run "$RUN" --gpu "$GPU" \
    --scenes $SCENES 2>&1 | tee -a "logs/${RUN}_render_gpu${GPU}.log"

echo "[chain gpu$GPU] render finished at $(date -Is)"
