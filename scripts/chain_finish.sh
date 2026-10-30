#!/usr/bin/env bash
# Wait until rendering has produced what it is going to, then finish the run:
# semantics, local scoring on the development scenes, and a validated zip.
#
# Waits on artefacts rather than on process IDs, so it survives a stage being
# restarted by hand, and it gives up after a deadline rather than hanging: a
# partial submission that exists beats a complete one that never got built.
#
#   ./scripts/chain_finish.sh <run> [deadline_minutes]
set -uo pipefail
cd "$(dirname "$0")/.."
source env.sh

RUN="${1:-2026-08-08-pipeline-v1}"
DEADLINE_MIN="${2:-300}"
DIR="experiments/$RUN"
EXPECTED=13
START=$(date +%s)

echo "[finish] waiting for renders under $DIR (deadline ${DEADLINE_MIN} min)"
while true; do
    DONE=$(find "$DIR" -path "*/gs/rgb/*.png" 2>/dev/null | wc -l)
    SCENES=$(find "$DIR" -type d -name rgb -path "*/gs/*" 2>/dev/null | wc -l)
    ELAPSED=$(( ($(date +%s) - START) / 60 ))
    if [ "$SCENES" -ge "$EXPECTED" ] && [ "$DONE" -ge $((EXPECTED * 3)) ]; then
        echo "[finish] all $EXPECTED scenes rendered ($DONE frames) after ${ELAPSED} min"
        break
    fi
    if [ "$ELAPSED" -ge "$DEADLINE_MIN" ]; then
        echo "[finish] deadline reached with $SCENES/$EXPECTED scenes and $DONE frames;"
        echo "[finish] continuing anyway so a submission exists"
        break
    fi
    sleep 60
done

echo "=== $(date -Is) semantics ==="
python3 scripts/run_pipeline.py --stage semantics --run "$RUN" --which all \
    2>&1 | tee -a "logs/${RUN}_semantics.log"

echo "=== $(date -Is) local score, development scenes ==="
python3 scripts/run_pipeline.py --stage score --run "$RUN" --which dev \
    2>&1 | tee -a "logs/${RUN}_score.log"

echo "=== $(date -Is) package ==="
python3 scripts/run_pipeline.py --stage package --run "$RUN" --which all --require all \
    --out "submissions/${RUN}.zip" 2>&1 | tee -a "logs/${RUN}_package.log"

echo "=== $(date -Is) FINISHED ==="
