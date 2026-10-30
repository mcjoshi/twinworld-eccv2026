"""What does submitting TUM at 4 cm instead of 2 cm actually cost?

TUM geometry was dropped from a 2 cm to a 4 cm submission voxel on 21 August to
free points for Gold Coast, when the working assumption was that the envelope
failed somewhere near 108 M points. The journal records the cost as 0.023 of
F-score and the trade as worth it.

Both halves of that assumption are now known to be wrong. There is no size limit
on this competition - the Final phase caps execution time at 6000 s and nothing
else - and a 254 M point submission has already scored. So the 0.023 is worth
buying back, if it is real.

It should be: the scorer voxel-downsamples both clouds to 2 cm before matching,
so a 4 cm submission discards resolution the scorer would have used, while going
finer than 2 cm can only add points the scorer immediately merges.

Measured here on the four TUM scenes that have ground truth, at the same F@5/10/20
the official scorer averages.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from twinworld.dataio import TUM_DEV, Scene
from twinworld.package import submission_cloud
from twinworld.scoring import read_ply_xyz, score_geometry

V1 = Path("experiments/2026-08-08-pipeline-v1")
OUT = Path("experiments/2026-08-30-tum-voxel")
VOXELS = (0.04, 0.02, 0.01)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", default=list(TUM_DEV))
    ap.add_argument("--voxels", type=float, nargs="+", default=list(VOXELS))
    a = ap.parse_args()
    scenes, voxels = a.scenes, tuple(a.voxels)
    OUT.mkdir(parents=True, exist_ok=True)
    per_voxel = {v: {"f": [], "pts": 0} for v in voxels}
    rows = {}
    for sid in scenes:
        s = Scene("tum", sid)
        gt, _ = read_ply_xyz(s.gt_ply)
        ply = V1 / s.dataset / s.scene_id / "mvs" / "fused_union.ply"
        rows[s.key] = {}
        for v in voxels:
            t0 = time.time()
            xyz, _ = submission_cloud(s, ply, voxel=v)
            g = score_geometry(xyz, gt, s.key)
            mean_f = float(np.mean(list(g.f_scores.values())))
            per_voxel[v]["f"].append(mean_f)
            per_voxel[v]["pts"] += len(xyz)
            rows[s.key][v] = {"mean_f": round(mean_f, 4),
                              "points": int(len(xyz)),
                              "per_threshold": {str(k): round(float(x), 4)
                                                for k, x in g.f_scores.items()}}
            print(f"[{s.key}] voxel {v:.2f}: {len(xyz)/1e6:5.2f}M pts  "
                  f"meanF {mean_f:.4f}  ({time.time()-t0:.0f}s)", flush=True)

    print(f"\n{'voxel':>6} {'total Mpts':>11} {'mean F':>8} {'vs 4cm':>8}")
    base = float(np.mean(per_voxel[voxels[0]]["f"]))
    summary = {}
    for v in voxels:
        m = float(np.mean(per_voxel[v]["f"]))
        summary[str(v)] = {"mean_f": round(m, 4),
                           "total_points": per_voxel[v]["pts"],
                           "delta_vs_4cm": round(m - base, 4)}
        print(f"{v:6.2f} {per_voxel[v]['pts']/1e6:11.1f} {m:8.4f} {m-base:+8.4f}")
    (OUT / "results.json").write_text(json.dumps(
        {"summary": summary, "per_scene": rows}, indent=2))
    print(f"\nwrote {OUT/'results.json'}", flush=True)


if __name__ == "__main__":
    main()
