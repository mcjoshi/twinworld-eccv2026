"""Why is only 1.4 percent of the Gold Coast cloud reachable by the scorer?

Two very different explanations, with very different fixes:

  extent   ground truth covers a small part of the area we reconstruct (a few
           scanned buildings inside a whole drone survey). Fix is a spatial
           crop, and the hard part is inferring the region on the final scenes
           where no ground truth exists.

  surface  ground truth covers the same area but is a thin laser-scanned shell,
           while ours is a thick noisy fusion. Fix is surface extraction, and it
           transfers to the final scenes for free.

This measures which one it is: bounding boxes, the fraction of our cloud inside
the ground-truth box, and reachability once already inside that box.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from twinworld.dataio import Scene
from twinworld.package import submission_cloud
from twinworld.scoring import read_ply_xyz

V1 = Path("experiments/2026-08-08-pipeline-v1")
OUT = Path("experiments/2026-08-13-budget-allocation")
MATCH_R = 0.10


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for sid in ("scene_009", "scene_010"):
        s = Scene("gold_coast", sid)
        ply = V1 / s.dataset / s.scene_id / "mvs" / "fused_gs_i.ply"
        xyz, _ = submission_cloud(s, ply, voxel=0.20)
        gt, _ = read_ply_xyz(s.gt_ply)

        lo_o, hi_o = xyz.min(0), xyz.max(0)
        lo_g, hi_g = gt.min(0), gt.max(0)
        ext_o, ext_g = hi_o - lo_o, hi_g - lo_g

        inside = np.all((xyz >= lo_g) & (xyz <= hi_g), axis=1)
        frac_in = float(inside.mean())

        tree = cKDTree(gt)
        d_in, _ = tree.query(xyz[inside], k=1,
                             distance_upper_bound=MATCH_R, workers=-1)
        reach_in = float(np.isfinite(d_in).mean()) if inside.any() else 0.0
        d_all, _ = tree.query(xyz, k=1, distance_upper_bound=MATCH_R, workers=-1)
        reach_all = float(np.isfinite(d_all).mean())

        # how thick is our cloud where it does overlap? distance distribution
        dfull, _ = tree.query(xyz[inside][:2_000_000], k=1, workers=-1)
        q = np.percentile(dfull, [10, 25, 50, 75, 90]).round(3).tolist()

        row = {
            "scene": s.key,
            "our_points": int(len(xyz)), "gt_points": int(len(gt)),
            "our_extent_m": ext_o.round(1).tolist(),
            "gt_extent_m": ext_g.round(1).tolist(),
            "area_ratio": round(float((ext_o[0] * ext_o[1]) /
                                      (ext_g[0] * ext_g[1])), 1),
            "frac_inside_gt_bbox": round(frac_in, 4),
            "reachable_all": round(reach_all, 4),
            "reachable_inside_bbox": round(reach_in, 4),
            "dist_to_gt_pctl_10_25_50_75_90": q,
        }
        rows.append(row)
        print(json.dumps(row, indent=2), flush=True)
        (OUT / "gc_extent.json").write_text(json.dumps(rows, indent=2))

        verdict = ("EXTENT-dominated: crop is the lever"
                   if frac_in < 0.5 and reach_in > 3 * reach_all
                   else "SURFACE-dominated: thinning is the lever")
        print(f"  -> {verdict}\n", flush=True)


if __name__ == "__main__":
    main()
