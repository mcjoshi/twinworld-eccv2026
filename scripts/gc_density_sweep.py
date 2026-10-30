"""How dense does a Gold Coast submission actually need to be?

The semantic metric asks only whether a ground-truth point has *any* submitted
point within 10 cm. Coverage is therefore a property of spacing, not of density,
and there is a spacing beyond which extra points buy nothing at all: points on a
plane spaced `s` apart put every location within `s/sqrt(2)` of one of them, so
anything below about 14 cm spacing already saturates the 10 cm radius.

Both of the current clouds ignore this:

- the submitted cloud is voxel-downsampled at 2 cm, the resolution the *geometry*
  metric uses, which is irrelevant on Gold Coast
- the extruded facades are emitted every 4 cm, which is 6x finer than needed

That matters because the Gold Coast clouds are 30 M points each and a full
submission approaches 2 GB. This sweeps both knobs against the only thing that
counts, `miou_ceiling`, and reports the resulting file size.
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
from twinworld.scoring import read_ply_xyz, score_semantics, voxel_downsample
from twinworld.semantics.features import estimate_up

BYTES_PER_POINT = 13  # 3x float32 + uint8 classification


def measure(name, xyz, gt_xyz, gt_lab, rows):
    c = score_semantics(xyz, np.zeros(len(xyz), np.uint8), gt_xyz, gt_lab)
    cov = c.per_class_coverage
    mb = len(xyz) * BYTES_PER_POINT / 1e6
    rows.append({"method": name, "n_points": int(len(xyz)), "approx_mb": round(mb, 1),
                 "miou_ceiling": c.miou_ceiling,
                 "per_class_coverage": {CLASS_NAMES[i]: round(float(cov[i]), 4)
                                        for i in range(NUM_CLASSES)}})
    print(f"  {name:30s} N={len(xyz):9d} ~{mb:7.1f} MB  ceiling {c.miou_ceiling:.4f}   "
          + "  ".join(f"{CLASS_NAMES[i]} {cov[i]:.3f}" for i in range(NUM_CLASSES)), flush=True)
    return c.miou_ceiling


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="experiments/2026-08-08-pipeline-v1")
    ap.add_argument("--scenes", nargs="*", default=["gold_coast/scene_009", "gold_coast/scene_010"])
    ap.add_argument("--out", default="experiments/2026-08-08-gc-density-sweep")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {}

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
        print(f"\n[{s.key}]", flush=True)

        # 1. how far can the MVS cloud alone be thinned before coverage drops?
        base = measure("mvs_voxel0.02", xyz, gt_xyz, gt_lab, rows)
        for v in (0.04, 0.06, 0.08, 0.12):
            measure(f"mvs_voxel{v}", voxel_downsample(xyz, v), gt_xyz, gt_lab, rows)

        # 2. facades at a spacing matched to the metric rather than 6x finer
        for spacing in (0.04, 0.10):
            t0 = time.time()
            fac = extrude_facades(xyz, up, cell=0.10, min_building_height=2.5,
                                  step_threshold=1.5, point_spacing=spacing,
                                  max_points=120_000_000)
            print(f"  facades at {spacing} m spacing: {len(fac)} points "
                  f"({time.time()-t0:.0f}s)", flush=True)
            union = np.vstack([xyz, fac])
            measure(f"union_fac{spacing}", union, gt_xyz, gt_lab, rows)
            for v in (0.06, 0.08):
                measure(f"union_fac{spacing}_voxel{v}", voxel_downsample(union, v),
                        gt_xyz, gt_lab, rows)
            (out / f"{s.scene_id}.json").write_text(json.dumps(rows, indent=2))

        summary[s.key] = rows
        (out / f"{s.scene_id}.json").write_text(json.dumps(rows, indent=2))
        best = max(rows, key=lambda r: r["miou_ceiling"])
        print(f"  best ceiling {best['miou_ceiling']:.4f} at {best['method']} "
              f"({best['approx_mb']} MB); baseline {base:.4f} at "
              f"{rows[0]['approx_mb']} MB", flush=True)

    (out / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
