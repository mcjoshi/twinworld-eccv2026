"""Can we find the buildings in our own cloud, and is the ground truth inside them?

Two teams now post mIoU 0.374 and 0.375 on the withheld scenes against our 0.166,
without sacrificing geometry (lionelsy holds F 0.614 against our 0.627). That is
not a tuning gap. Our own radius curve says what it must be:

    match radius   0.10    0.15    0.20    0.30    0.50
    ceiling        0.42    0.60    0.71    0.82    0.90

The ceiling is not a property of the reconstruction, it is a property of how
densely space near it is filled - and Gold Coast charges no precision penalty, so
filling space costs nothing but points. A dense volumetric fill reaches a ceiling
near 1.0.

We dismissed this as unaffordable because dilating the whole 61.7 M-point cloud
by one voxel triples it. But that assumed dilating *everywhere*. The ground truth
is a building. Spending the budget only inside buildings changes the arithmetic
entirely.

Earlier region rules failed: camera frustums cover the whole scene (68-99 percent
of points), and the ball heuristic had a broken aim point. Neither tried the
obvious signal, which is already one of our 29 features: height above ground.

This measures, on the two scenes with ground truth:
  - what fraction of ground truth lies inside a region defined by tall points
  - how many points that region contains, i.e. what a dense fill would cost
  - the ceiling achievable by filling it at various voxel sizes

Diagnostic only. Nothing is adopted here, and the region rule uses no ground
truth - only height above ground computed from our own cloud - so it transfers
to the withheld scenes.
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
from twinworld.semantics.features import estimate_up

V1 = Path("experiments/2026-08-08-pipeline-v1")
OUT = Path("experiments/2026-08-29-building-region")
MATCH_R = 0.10


def height_above_ground(xyz, up, cell=3.0, q=0.02):
    """Height over a local ground estimate, on a horizontal grid."""
    e1 = np.array([1.0, 0.0, 0.0])
    if abs(e1 @ up) > 0.9:
        e1 = np.array([0.0, 1.0, 0.0])
    a = np.cross(up, e1); a /= np.linalg.norm(a)
    b = np.cross(up, a)
    u = xyz @ a
    v = xyz @ b
    h = xyz @ up
    iu = np.floor(u / cell).astype(np.int64)
    iv = np.floor(v / cell).astype(np.int64)
    key = iu * 100003 + iv
    order = np.argsort(key)
    ks, hs = key[order], h[order]
    bounds = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1], True])
    ground = np.empty(len(xyz), np.float64)
    for i in range(len(bounds) - 1):
        s, e = bounds[i], bounds[i + 1]
        g = np.quantile(hs[s:e], q)
        ground[order[s:e]] = g
    return h - ground


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for sid in ("scene_009", "scene_010"):
        s = Scene("gold_coast", sid)
        t0 = time.time()
        ply = V1 / s.dataset / s.scene_id / "mvs" / "fused_gs_i.ply"
        xyz, _ = submission_cloud(s, ply, voxel=0.10)
        gt, gl = read_ply_xyz(s.gt_ply, with_classification=True)
        up = estimate_up(s.train_frames)

        hag = height_above_ground(xyz, up)
        gt_hag = height_above_ground(gt, up)
        print(f"\n[{s.key}] {len(xyz)/1e6:.1f}M cloud pts, {len(gt)/1e6:.1f}M GT "
              f"({time.time()-t0:.0f}s)", flush=True)
        print(f"  our HAG percentiles  50/90/99: "
              f"{np.percentile(hag,[50,90,99]).round(2)}", flush=True)
        print(f"  GT  HAG percentiles  1/50/99 : "
              f"{np.percentile(gt_hag,[1,50,99]).round(2)}", flush=True)

        rec = {"scene": s.key, "cloud_points": int(len(xyz)), "gt_points": int(len(gt))}
        for thr in (1.0, 2.0, 3.0, 5.0):
            tall = xyz[hag > thr]
            if len(tall) < 1000:
                continue
            # region = anything within `margin` of a tall point, so the ground
            # around a building is included, which the GT also contains
            tree = cKDTree(tall)
            for margin in (3.0, 6.0):
                d, _ = tree.query(gt, k=1, distance_upper_bound=margin, workers=-1)
                inside = np.isfinite(d)
                dc, _ = tree.query(xyz, k=1, distance_upper_bound=margin, workers=-1)
                cloud_in = np.isfinite(dc)
                per_class = {CLASS_NAMES[c]: round(float(inside[gl == c].mean()), 4)
                             for c in range(NUM_CLASSES) if (gl == c).any()}
                key = f"thr{thr}_margin{margin}"
                rec[key] = {"tall_points": int(len(tall)),
                            "gt_inside": round(float(inside.mean()), 4),
                            "cloud_inside": round(float(cloud_in.mean()), 4),
                            "cloud_points_inside": int(cloud_in.sum()),
                            "gt_inside_per_class": per_class}
                print(f"  HAG>{thr:>3}m margin {margin:>3}m: "
                      f"GT inside {inside.mean()*100:5.1f}%  "
                      f"cloud inside {cloud_in.mean()*100:5.1f}% "
                      f"({cloud_in.sum()/1e6:5.1f}M pts)  "
                      + " ".join(f"{k[:4]} {v:.2f}" for k, v in per_class.items()),
                      flush=True)
        rows.append(rec)
        (OUT / "results.json").write_text(json.dumps(rows, indent=2))

    print("\nreading: if GT-inside is ~99% while cloud-inside is much smaller, the "
          "region is a valid crop and the freed budget can be spent filling it "
          "densely - which is what the radius curve says buys the score.",
          flush=True)


if __name__ == "__main__":
    main()
