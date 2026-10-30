"""Find the building by density, crop to it, and spend the whole budget filling it.

The situation this addresses. Two teams post mIoU 0.374 and 0.375 on the withheld
scenes against our 0.166, without sacrificing geometry. Our own radius curve
explains what they must be doing: the coverage ceiling is 0.42 at a 10 cm match
radius, 0.71 at 20 cm, 0.82 at 30 cm and 0.90 at 50 cm. Gold Coast charges no
precision penalty, so the ceiling is set by how densely space near the surface is
filled, not by reconstruction quality.

The obstacle has always been budget - dilating 61.7 M points is unaffordable. But
the budget is being wasted, badly: our cloud carries 35.8 M points against 8.8 M
ground-truth points and still covers only 42 percent of them, because ground
truth occupies a 65 x 62 m box inside a 366 x 532 m reconstruction. Roughly
98 percent of the area we pay for contains no ground truth at all.

Height above ground was tested first and failed: our cloud's HAG median is 25 m,
so the whole reconstruction reads as tall and the region contains 99.4 percent of
the points. Density is the alternative signal - the building of interest is the
most photographed and most densely reconstructed part of the scene, and unlike a
bounding box it uses no ground truth, so it transfers to the withheld scenes.

Measured here: for a range of density thresholds, what fraction of ground truth
falls inside the retained region, how many points that costs, and what ceiling a
dense fill of that region reaches. A rule is only usable if it keeps essentially
all ground truth while discarding most of the cloud.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from twinworld.dataio import CLASS_NAMES, NUM_CLASSES, Scene
from twinworld.package import submission_cloud
from twinworld.scoring import read_ply_xyz

V1 = Path("experiments/2026-08-08-pipeline-v1")
OUT = Path("experiments/2026-08-29-density-region")
MATCH_R = 0.10


def coarse_density(xyz, cell=5.0):
    """Points per cell on a 5 m grid, mapped back to every point."""
    key = np.floor(xyz / cell).astype(np.int64)
    _, inv, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    return counts[inv], counts


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for sid in ("scene_009", "scene_010"):
        s = Scene("gold_coast", sid)
        t0 = time.time()
        ply = V1 / s.dataset / s.scene_id / "mvs" / "fused_gs_i.ply"
        xyz, _ = submission_cloud(s, ply, voxel=0.10)
        gt, gl = read_ply_xyz(s.gt_ply, with_classification=True)

        dens, counts = coarse_density(xyz)
        print(f"\n[{s.key}] {len(xyz)/1e6:.1f}M pts, {len(gt)/1e6:.1f}M GT, "
              f"cell counts p50/p90/p99 "
              f"{np.percentile(counts,[50,90,99]).round(0)} ({time.time()-t0:.0f}s)",
              flush=True)

        rec = {"scene": s.key, "cloud_points": int(len(xyz))}
        for q in (50, 70, 80, 90, 95):
            thr = np.percentile(dens, q)
            keep = dens >= thr
            if keep.sum() < 10000:
                continue
            sub = xyz[keep]
            tree = cKDTree(sub)
            # ground truth still reachable at the scored radius after cropping
            d, _ = tree.query(gt, k=1, distance_upper_bound=MATCH_R, workers=-1)
            cov = np.isfinite(d)
            # and within a generous margin, i.e. is the GT even in this region
            d2, _ = tree.query(gt, k=1, distance_upper_bound=5.0, workers=-1)
            near = np.isfinite(d2)
            per_class = {CLASS_NAMES[c]: round(float(cov[gl == c].mean()), 4)
                         for c in range(NUM_CLASSES) if (gl == c).any()}
            ceiling = float(np.mean(list(per_class.values())))
            rec[f"q{q}"] = {"kept_points": int(keep.sum()),
                            "kept_frac": round(float(keep.mean()), 4),
                            "gt_within_5m": round(float(near.mean()), 4),
                            "ceiling_at_10cm": round(ceiling, 4),
                            "per_class": per_class}
            print(f"  keep top {100-q:>2}% dense: {keep.sum()/1e6:6.2f}M pts "
                  f"({keep.mean()*100:5.1f}%)  GT within 5m {near.mean()*100:5.1f}%  "
                  f"ceiling {ceiling:.4f}", flush=True)
        rows.append(rec)
        (OUT / "results.json").write_text(json.dumps(rows, indent=2))

    print("\nreading: a usable rule keeps GT-within-5m near 100 percent while "
          "cutting kept_points hard. The freed budget then funds a dense fill, "
          "which the radius curve says is where the score is.", flush=True)


if __name__ == "__main__":
    main()
