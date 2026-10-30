#!/usr/bin/env bash
# Unattended run: dense geometry for every scene, then splatting, then semantics,
# then a validated submission zip.
#
# Two GPUs are used (2 and 3 only; 0 and 1 belong to another user). Work is split
# so both stay busy, and every stage is resumable: re-running skips scenes whose
# artefacts already exist.
#
#   ./scripts/overnight.sh <mvs_variant> <run_name>
set -uo pipefail
cd "$(dirname "$0")/.."
source env.sh

VARIANT="${1:-loose}"
RUN="${2:-2026-08-08-pipeline-v1}"
mkdir -p logs

TUM_A="tum/scene_000 tum/scene_001 tum/scene_002 tum/scene_003 tum/scene_004"
TUM_B="tum/scene_005 tum/scene_006 tum/scene_007 tum/scene_008"
GC_A="gold_coast/scene_009 gold_coast/scene_010"
GC_B="gold_coast/scene_011 gold_coast/scene_012"

echo "=== $(date -Is) stage: mvs (variant=$VARIANT) ==="
python3 scripts/run_pipeline.py --stage mvs --run "$RUN" --gpu 2 --mvs-variant-tum "$VARIANT" \
    --scenes $TUM_A $GC_A >> "logs/${RUN}_mvs_gpu2.log" 2>&1 &
P2=$!
python3 scripts/run_pipeline.py --stage mvs --run "$RUN" --gpu 3 --mvs-variant-tum "$VARIANT" \
    --scenes $TUM_B $GC_B >> "logs/${RUN}_mvs_gpu3.log" 2>&1 &
P3=$!
wait $P2 $P3
echo "=== $(date -Is) mvs done ==="

echo "=== $(date -Is) stage: render ==="
python3 scripts/run_pipeline.py --stage render --run "$RUN" --gpu 2 \
    --scenes $TUM_A $GC_A >> "logs/${RUN}_render_gpu2.log" 2>&1 &
P2=$!
python3 scripts/run_pipeline.py --stage render --run "$RUN" --gpu 3 \
    --scenes $TUM_B $GC_B >> "logs/${RUN}_render_gpu3.log" 2>&1 &
P3=$!
wait $P2 $P3
echo "=== $(date -Is) render done ==="

echo "=== $(date -Is) stage: semantics ==="
python3 scripts/run_pipeline.py --stage semantics --run "$RUN" --which all \
    >> "logs/${RUN}_semantics.log" 2>&1
echo "=== $(date -Is) semantics done ==="

echo "=== $(date -Is) stage: score (dev scenes) ==="
python3 scripts/run_pipeline.py --stage score --run "$RUN" --which dev \
    2>&1 | tee -a "logs/${RUN}_score.log"

echo "=== $(date -Is) stage: package ==="
python3 scripts/run_pipeline.py --stage package --run "$RUN" --which all --require all \
    --out "submissions/${RUN}.zip" 2>&1 | tee -a "logs/${RUN}_package.log"

echo "=== $(date -Is) ALL DONE ==="
