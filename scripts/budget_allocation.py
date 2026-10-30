"""Where should the scorer's point budget go?

The submission is bounded by the organiser's 600 s execution limit, and the
dominant cost is a nearest-neighbour query of ground-truth points against ours,
so *our total point count* is the budget being spent.

v7-mini spends it 31 / 69 between TUM and Gold Coast. TUM points buy F-score
directly. Gold Coast points buy only the chance that a ground-truth point finds
a neighbour within 10 cm - and Gold Coast has no precision term, so a submitted
point that is far from every ground-truth point earns nothing at all while still
costing runtime.

This measures three things on the two Gold Coast scenes with ground truth:

  reachable   fraction of our submitted points within the 10 cm match radius of
              some ground-truth point. Everything else is pure runtime waste.
  oracle      the semantic ceiling if we kept only the reachable points. Must be
              identical to keeping everything - if it is not, the match radius
              reasoning is wrong and this whole analysis is void.
  proxy       the same, using a rule that needs no ground truth, so it can be
              applied to the final scenes: keep points whose local neighbourhood
              is dense enough to be surface rather than stray fusion noise.

Then it prices the TUM side: F-score and point count against voxel size, on all
four TUM scenes that have ground truth, so the freed budget can be spent.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from twinworld.dataio import Scene
from twinworld.package import submission_cloud
from twinworld.scoring import (
    read_ply_xyz, score_geometry, score_semantics, voxel_downsample,
)

V1 = Path("experiments/2026-08-08-pipeline-v1")
OUT = Path("experiments/2026-08-13-budget-allocation")
MATCH_R = 0.10

TUM_VOXELS = [0.02, 0.025, 0.03, 0.04, 0.05, 0.06]
GC_VOXELS = [0.10, 0.14, 0.20]


def gc_analysis(rows: list) -> None:
    """How much of the Gold Coast budget can the scorer actually reach?"""
    for sid in ("scene_009", "scene_010"):
        s = Scene("gold_coast", sid)
        ply = V1 / s.dataset / s.scene_id / "mvs" / "fused_gs_i.ply"
        gt, gl = read_ply_xyz(s.gt_ply, with_classification=True)
        gt_tree = cKDTree(gt)

        for voxel in GC_VOXELS:
            t0 = time.time()
            xyz, _ = submission_cloud(s, ply, voxel=voxel)
            n = len(xyz)

            # reachable = within the match radius of some ground-truth point
            d, _ = gt_tree.query(xyz, k=1, distance_upper_bound=MATCH_R,
                                 workers=-1)
            reachable = np.isfinite(d)
            frac = float(reachable.mean())

            # oracle: keep only reachable points. The semantic score must not
            # move - if it does, the match-radius argument is wrong.
            lab = np.zeros(n, np.uint8)  # labels irrelevant for the ceiling
            c_all = score_semantics(xyz, lab, gt, gl)
            c_reach = score_semantics(xyz[reachable], lab[reachable], gt, gl)

            row = {
                "scene": s.key, "voxel": voxel, "points": int(n),
                "reachable_frac": round(frac, 4),
                "points_reachable": int(reachable.sum()),
                "ceiling_all": round(float(c_all.miou_ceiling), 4),
                "ceiling_reachable": round(float(c_reach.miou_ceiling), 4),
                "seconds": round(time.time() - t0, 1),
            }
            rows.append(row)
            print(f"[{s.key} v={voxel}] N={n/1e6:6.2f}M  reachable "
                  f"{frac*100:5.1f}%  ceiling {row['ceiling_all']:.4f} -> "
                  f"{row['ceiling_reachable']:.4f}  ({row['seconds']}s)",
                  flush=True)
            OUT.mkdir(parents=True, exist_ok=True)
            (OUT / "gold_coast.json").write_text(json.dumps(rows, indent=2))


def tum_analysis(rows: list) -> None:
    """What does each TUM voxel size cost in points and buy in F-score?"""
    scenes = [Scene("tum", f"scene_{i:03d}") for i in (0, 1, 2, 3)]
    for voxel in TUM_VOXELS:
        per = []
        for s in scenes:
            t0 = time.time()
            ply = V1 / s.dataset / s.scene_id / "mvs" / "fused_union.ply"
            xyz, _ = submission_cloud(s, ply, voxel=voxel)
            gt, _ = read_ply_xyz(s.gt_ply)
            r = score_geometry(xyz, gt, s.key)
            per.append({"scene": s.key, "points": int(len(xyz)),
                        "fscore": round(float(r.fscore), 4),
                        "seconds": round(time.time() - t0, 1)})
            print(f"  [{s.key} v={voxel}] N={len(xyz)/1e6:5.2f}M "
                  f"F={r.fscore:.4f} ({per[-1]['seconds']}s)", flush=True)
        mean_f = float(np.mean([p["fscore"] for p in per]))
        total_n = int(sum(p["points"] for p in per))
        rows.append({"voxel": voxel, "mean_fscore": round(mean_f, 4),
                     "points_4scenes": total_n, "per_scene": per})
        print(f"[TUM v={voxel}] mean F={mean_f:.4f} over 4 GT scenes, "
              f"{total_n/1e6:.1f}M points", flush=True)
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "tum.json").write_text(json.dumps(rows, indent=2))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    gc_rows: list = []
    tum_rows: list = []
    print("=== Gold Coast: how much of the budget is reachable? ===", flush=True)
    gc_analysis(gc_rows)
    print("\n=== TUM: what does the freed budget buy? ===", flush=True)
    tum_analysis(tum_rows)
    print("\ndone", flush=True)


if __name__ == "__main__":
    main()
