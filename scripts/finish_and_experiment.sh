#!/usr/bin/env bash
# Finish the run into an uploadable zip, then keep the GPUs busy with ablations.
#
# Ordering matters: the deliverable is a submission the user can upload, so the
# zip is built as soon as renders and labels exist, before any experiment
# starts. Ablations only run afterwards, on GPUs that would otherwise idle.
#
#   ./scripts/finish_and_experiment.sh <run> [deadline_minutes]
set -uo pipefail
cd "$(dirname "$0")/.."
source env.sh
export OMP_NUM_THREADS=4
export OMP_WAIT_POLICY=PASSIVE

RUN="${1:-2026-08-08-pipeline-v1}"
DEADLINE_MIN="${2:-180}"
DIR="experiments/$RUN"
START=$(date +%s)

echo "[finish] waiting for 13 rendered scenes under $DIR (deadline ${DEADLINE_MIN} min)"
while true; do
    SCENES=$(find "$DIR" -type d -path "*/gs/rgb" 2>/dev/null | wc -l)
    FRAMES=$(find "$DIR" -path "*/gs/rgb/*.png" 2>/dev/null | wc -l)
    ELAPSED=$(( ($(date +%s) - START) / 60 ))
    [ "$SCENES" -ge 13 ] && [ "$FRAMES" -ge 39 ] && { echo "[finish] all rendered after ${ELAPSED} min"; break; }
    [ "$ELAPSED" -ge "$DEADLINE_MIN" ] && { echo "[finish] deadline hit at $SCENES scenes / $FRAMES frames, proceeding"; break; }
    sleep 60
done

echo "=== $(date -Is) semantics ==="
timeout 7200 python3 scripts/run_pipeline.py --stage semantics --run "$RUN" --which all \
    2>&1 | tee -a "logs/${RUN}_semantics.log"

echo "=== $(date -Is) local score on development scenes ==="
timeout 5400 python3 scripts/run_pipeline.py --stage score --run "$RUN" --which dev \
    2>&1 | tee -a "logs/${RUN}_score.log"

# Build the zip even if semantics failed: `_artefacts` falls back to a constant
# majority label, which scores about 0.11 mIoU rather than nothing.
echo "=== $(date -Is) package ==="
timeout 7200 python3 scripts/run_pipeline.py --stage package --run "$RUN" --which all --require all \
    --out "submissions/${RUN}.zip" 2>&1 | tee -a "logs/${RUN}_package.log"

if [ -f "submissions/${RUN}.zip" ]; then
    echo "=== $(date -Is) SUBMISSION READY: submissions/${RUN}.zip ==="
else
    echo "=== $(date -Is) NO ZIP WRITTEN, see logs/${RUN}_package.log ==="
fi

# --- GPUs are free now; spend them on the unmeasured render settings ---------
echo "=== $(date -Is) render ablation ==="
CUDA_VISIBLE_DEVICES=2 python3 scripts/render_ablation.py --gpu 2 \
    --variants mvs_short mvs_long mvs_reg \
    --scenes tum/scene_000 gold_coast/scene_009 \
    --mvs-exp "experiments/$RUN" >> logs/render_ablation_gpu2.log 2>&1 &
A=$!
CUDA_VISIBLE_DEVICES=3 python3 scripts/render_ablation.py --gpu 3 \
    --variants sparse_init mvs_mcmc mvs_randombg \
    --scenes tum/scene_000 gold_coast/scene_009 \
    --mvs-exp "experiments/$RUN" >> logs/render_ablation_gpu3.log 2>&1 &
B=$!
wait $A $B
echo "=== $(date -Is) render ablation done ==="
