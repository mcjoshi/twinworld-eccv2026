"""Does the coverage-maximal MVS config actually raise the semantic ceiling?

Gold Coast is never scored on geometric precision. The claim behind using a
looser, noisier config there is that the semantic score is capped by how much
ground truth has any submitted point within 10 cm:

    IoU_c <= coverage_c,   so   mIoU <= mean_c coverage_c

That is a theory until measured. This compares candidate clouds for the two
labelled Gold Coast scenes on the quantity that matters (`miou_ceiling`), not on
F-score, and reports the geometry F-score alongside only to show what is being
traded away.

    python scripts/gc_coverage_check.py \
        --clouds coverage=experiments/2026-08-08-pipeline-v1 \
                 loose=experiments/2026-08-08-geometry-bakeoff/loose
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from twinworld.dataio import CLASS_NAMES, NUM_CLASSES, Scene
from twinworld.scoring import (
    SemanticCounts, read_ply_xyz, score_geometry, score_semantics,
)


def find_cloud(root: Path, s: Scene) -> Path | None:
    for cand in (
        root / s.dataset / s.scene_id / "mvs" / "fused.ply",   # pipeline layout
        root / s.dataset / s.scene_id / "fused.ply",            # ablation layout
    ):
        if cand.exists():
            return cand
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clouds", nargs="+", required=True, help="name=path pairs")
    ap.add_argument("--out", default="experiments/2026-08-08-gc-coverage-check")
    args = ap.parse_args()

    scenes = [Scene("gold_coast", s) for s in ("scene_009", "scene_010")]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for spec in args.clouds:
        name, _, root = spec.partition("=")
        pooled = SemanticCounts()
        per_scene = []
        for s in scenes:
            ply = find_cloud(Path(root), s)
            if ply is None:
                print(f"[{name}] {s.key}: no cloud under {root}, skipping", flush=True)
                continue
            xyz, _ = read_ply_xyz(ply)
            gt_xyz, gt_lab = read_ply_xyz(s.gt_ply, with_classification=True)

            # ceiling only depends on geometry: give every point a label that can
            # never be right, so the only thing measured is what is reachable
            c = score_semantics(xyz, np.zeros(len(xyz), np.uint8), gt_xyz, gt_lab)
            g = score_geometry(xyz, gt_xyz, s.key)
            pooled.add(c)
            per_scene.append({
                "scene": s.key, "n_points": int(len(xyz)),
                "miou_ceiling": c.miou_ceiling,
                "per_class_coverage": {
                    CLASS_NAMES[i]: (None if np.isnan(c.per_class_coverage[i])
                                     else round(float(c.per_class_coverage[i]), 4))
                    for i in range(NUM_CLASSES)},
                "fscore_for_reference": g.fscore,
            })
            print(
                f"[{name}] {s.key}: N={len(xyz):9d}  ceiling {c.miou_ceiling:.4f}  "
                + "  ".join(f"{CLASS_NAMES[i]} {c.per_class_coverage[i]:.3f}"
                            for i in range(NUM_CLASSES))
                + f"   (F={g.fscore:.3f})",
                flush=True,
            )
        if per_scene:
            rows.append({"name": name, "root": root,
                         "pooled_miou_ceiling": pooled.miou_ceiling,
                         "pooled_per_class_coverage": [
                             None if np.isnan(v) else round(float(v), 4)
                             for v in pooled.per_class_coverage],
                         "scenes": per_scene})
            print(f"[{name}] POOLED ceiling = {pooled.miou_ceiling:.4f}\n", flush=True)

    (out / "ceilings.json").write_text(json.dumps(rows, indent=2))
    if len(rows) > 1:
        best = max(rows, key=lambda r: r["pooled_miou_ceiling"])
        print("pooled semantic ceiling by cloud:")
        for r in sorted(rows, key=lambda r: -r["pooled_miou_ceiling"]):
            print(f"  {r['name']:12s} {r['pooled_miou_ceiling']:.4f}")
        print(f"\n-> use '{best['name']}' for Gold Coast")


if __name__ == "__main__":
    main()
