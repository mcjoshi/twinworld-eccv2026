"""Does extruding facades from roof outlines raise the Gold Coast ceiling?

The semantic score is capped by the fraction of ground truth within 10 cm of a
submitted point, and on Gold Coast that cap is 0.20 with walls at 0.02-0.05 and
windows at 0.01-0.04. This measures whether modelled facades close that gap, on
the two development scenes where it can be checked.

The number that decides it is `miou_ceiling`; the F-score is printed only to
show what is being traded, since Gold Coast is never scored on geometric
precision.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from twinworld.dataio import CLASS_NAMES, NUM_CLASSES, Scene
from twinworld.geometry.facade import extrude_facades
from twinworld.package import submission_cloud
from twinworld.scoring import read_ply_xyz, score_semantics
from twinworld.semantics.features import estimate_up

# Cell size is the dominant parameter: the extruded wall is placed on a cell
# boundary, so a cell of `c` metres puts it within about `c/2` of the true plane.
# The semantic match radius is 10 cm, so 0.30 m cells (median error 0.20 m on a
# synthetic box) are far too coarse to score, and the sweep has to reach 0.10 m.
SWEEP = [
    {"cell": 0.10, "min_building_height": 2.5, "step_threshold": 1.5},
    {"cell": 0.15, "min_building_height": 2.5, "step_threshold": 1.5},
    {"cell": 0.20, "min_building_height": 2.0, "step_threshold": 1.0},
    {"cell": 0.30, "min_building_height": 2.5, "step_threshold": 1.5},
]


def report(name, xyz, gt_xyz, gt_lab, rows):
    c = score_semantics(xyz, np.zeros(len(xyz), np.uint8), gt_xyz, gt_lab)
    cov = c.per_class_coverage
    rows.append({"method": name, "n_points": int(len(xyz)), "miou_ceiling": c.miou_ceiling,
                 "per_class_coverage": {CLASS_NAMES[i]: round(float(cov[i]), 4)
                                        for i in range(NUM_CLASSES)}})
    print(f"  {name:32s} N={len(xyz):9d}  ceiling {c.miou_ceiling:.4f}   "
          + "  ".join(f"{CLASS_NAMES[i]} {cov[i]:.3f}" for i in range(NUM_CLASSES)), flush=True)
    return c.miou_ceiling


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="experiments/2026-08-08-pipeline-v1")
    ap.add_argument("--scenes", nargs="*", default=["gold_coast/scene_009", "gold_coast/scene_010"])
    ap.add_argument("--out", default="experiments/2026-08-08-facade-extrusion")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    all_rows = {}

    for key in args.scenes:
        s = Scene(*key.split("/"))
        ply = Path(args.run) / s.dataset / s.scene_id / "mvs" / "fused.ply"
        if not ply.exists():
            print(f"skip {key}: no cloud")
            continue
        cache = Path(args.run) / s.dataset / s.scene_id / "mvs" / "submission_cloud.npz"
        xyz, _ = submission_cloud(s, ply, cache=cache)
        gt_xyz, gt_lab = read_ply_xyz(s.gt_ply, with_classification=True)
        up = estimate_up(s.train_frames)

        rows: list[dict] = []
        print(f"\n[{s.key}] {len(xyz)} submission points", flush=True)
        base = report("mvs_only", xyz, gt_xyz, gt_lab, rows)

        for params in SWEEP:
            t0 = time.time()
            facades = extrude_facades(xyz, up, **params)
            tag = f"facade_c{params['cell']}_h{params['min_building_height']}_s{params['step_threshold']}"
            print(f"  {tag}: {len(facades)} extruded points ({time.time()-t0:.0f}s)", flush=True)
            if len(facades) == 0:
                continue
            got = report(tag + "_union", np.vstack([xyz, facades]), gt_xyz, gt_lab, rows)
            print(f"      -> ceiling {got - base:+.4f} vs mvs_only", flush=True)
            (out / f"{s.scene_id}.json").write_text(json.dumps(rows, indent=2))

        all_rows[s.key] = rows
        (out / f"{s.scene_id}.json").write_text(json.dumps(rows, indent=2))

    (out / "summary.json").write_text(json.dumps(all_rows, indent=2))
    print("\npooled best per scene:")
    for k, rows in all_rows.items():
        best = max(rows, key=lambda r: r["miou_ceiling"])
        base = next(r for r in rows if r["method"] == "mvs_only")
        print(f"  {k}: {base['miou_ceiling']:.4f} -> {best['miou_ceiling']:.4f} "
              f"({best['method']})")


if __name__ == "__main__":
    main()
