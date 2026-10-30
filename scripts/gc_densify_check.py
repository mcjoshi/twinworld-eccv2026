"""Can surface completion recover Gold Coast facade coverage?

Measured on scene_009: only 1.7 percent of ground-truth wall points and 0.5
percent of window points have a submitted point within 10 cm, while roofs reach
39.6 percent. About a third of the wall is reconstructed within 50 cm and the
rest is missing entirely, so the deficit is part missing surface and part noisy
surface. Poisson reconstruction addresses both: it interpolates across holes and
averages depth noise into a surface.

What matters here is `miou_ceiling`, the mean per-class coverage, not the
F-score: Gold Coast is never scored on geometric precision.

Poisson grids the bounding box, so the cloud is cropped first. The crop is
percentile-based rather than the distance-from-rig rule used for TUM, because
the Gold Coast clouds stay several hundred metres across even after that rule
and the octree would be too coarse to resolve a facade. Nothing is lost: the
fitted surface is unioned back with the full raw cloud.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from twinworld.dataio import CLASS_NAMES, NUM_CLASSES, Scene
from twinworld.geometry.densify import poisson_densify, read_cloud_with_normals, robust_core
from twinworld.scoring import read_ply_xyz, score_semantics


def report(name: str, xyz: np.ndarray, gt_xyz, gt_lab, rows: list) -> None:
    c = score_semantics(xyz, np.zeros(len(xyz), np.uint8), gt_xyz, gt_lab)
    cov = c.per_class_coverage
    rows.append({
        "method": name, "n_points": int(len(xyz)), "miou_ceiling": c.miou_ceiling,
        "per_class_coverage": {CLASS_NAMES[i]: round(float(cov[i]), 4) for i in range(NUM_CLASSES)},
    })
    print(f"  {name:34s} N={len(xyz):9d}  ceiling {c.miou_ceiling:.4f}   "
          + "  ".join(f"{CLASS_NAMES[i]} {cov[i]:.3f}" for i in range(NUM_CLASSES)), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cloud", required=True)
    ap.add_argument("--scene", default="gold_coast/scene_009")
    ap.add_argument("--out", default="experiments/2026-08-08-gc-densify-check")
    ap.add_argument("--pct", type=float, default=1.0, help="per-axis percentile crop for Poisson")
    args = ap.parse_args()

    s = Scene(*args.scene.split("/"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    xyz, nrm, rgb = read_cloud_with_normals(args.cloud)
    gt_xyz, gt_lab = read_ply_xyz(s.gt_ply, with_classification=True)
    rows: list[dict] = []

    print(f"[{s.key}] {len(xyz)} raw points", flush=True)
    report("mvs_raw", xyz, gt_xyz, gt_lab, rows)

    core = robust_core(xyz, s.train_frames)
    lo = np.percentile(xyz[core], args.pct, axis=0)
    hi = np.percentile(xyz[core], 100 - args.pct, axis=0)
    tight = core & np.all((xyz >= lo) & (xyz <= hi), axis=1)
    print(f"  poisson input: {int(tight.sum())} points, extent "
          f"{(xyz[tight].max(0) - xyz[tight].min(0)).round(1)}", flush=True)

    for trim in (0.30, 1.00):
        t0 = time.time()
        try:
            pts, _, info = poisson_densify(
                xyz[tight],
                nrm[tight] if nrm is not None else None,
                rgb[tight] if rgb is not None else None,
                trim_distance=trim, density_quantile=0.05,
                sample_spacing=0.03, target_cell=0.05,
            )
        except Exception as e:  # noqa: BLE001
            print(f"  poisson trim={trim} FAILED: {e}", flush=True)
            continue
        print(f"  poisson trim={trim}: depth={info['depth']} area={info['mesh_area_m2']} m2, "
              f"{info['kept_after_trim']} kept ({time.time()-t0:.0f}s)", flush=True)
        report(f"poisson_trim{trim}", pts, gt_xyz, gt_lab, rows)
        report(f"poisson_trim{trim}_union_mvs", np.vstack([xyz, pts]), gt_xyz, gt_lab, rows)
        (out / f"{s.scene_id}.json").write_text(json.dumps(rows, indent=2))

    (out / f"{s.scene_id}.json").write_text(json.dumps(rows, indent=2))
    best = max(rows, key=lambda r: r["miou_ceiling"])
    print(f"\nbest ceiling: {best['method']} at {best['miou_ceiling']:.4f} "
          f"(raw was {rows[0]['miou_ceiling']:.4f})")


if __name__ == "__main__":
    main()
