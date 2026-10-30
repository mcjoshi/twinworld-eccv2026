"""Can splatting depth help TUM if it only fills gaps stereo left empty?

Unioning the whole GS depth cloud into TUM costs F-score (0.5336 -> 0.5193): it
raises recall 0.551 -> 0.618 but drops precision 0.652 -> 0.537, and F-score
penalises both.

But the two failures are separable. Most of the precision loss should come from
GS points landing on surfaces stereo already reconstructed, where they add
duplicate, noisier estimates near a good one. Points landing where stereo found
nothing cannot displace anything, and are the only ones that can add recall.

So: keep a GS point only if no stereo point lies within `gap` metres of it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from twinworld.dataio import Scene
from twinworld.geometry.gs_depth import gs_point_cloud
from twinworld.package import submission_cloud
from twinworld.render.trainer import TrainConfig, train_scene
from twinworld.scoring import read_ply_xyz, score_geometry, voxel_downsample

ITERS = 3000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", default=["tum/scene_000", "tum/scene_002"])
    ap.add_argument("--src", default="experiments/2026-08-08-pipeline-v1")
    ap.add_argument("--work", default="/data2/manish_joshi/twinworld/work/tumgap")
    ap.add_argument("--out", default="experiments/2026-08-10-tum-gapfill")
    args = ap.parse_args()

    from scipy.spatial import cKDTree

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for key in args.scenes:
        s = Scene(*key.split("/"))
        ply = Path(args.src) / s.dataset / s.scene_id / "mvs" / "fused.ply"
        mvs, _ = submission_cloud(s, ply, cache=ply.parent / "submission_cloud.npz")
        gt, _ = read_ply_xyz(s.gt_ply)
        base = score_geometry(mvs, gt, s.key)
        print(f"\n[{s.key}] mvs only: F={base.fscore:.4f} "
              f"(P@10 {base.precision[0.10]:.3f} R@10 {base.recall[0.10]:.3f})", flush=True)
        rows.append({"scene": s.key, "method": "mvs_only", "n": int(len(mvs)),
                     "fscore": base.fscore, "p10": base.precision[0.10],
                     "r10": base.recall[0.10]})

        cfg = TrainConfig(iters=ITERS, refine_stop_iter=int(ITERS * 0.72))
        work = Path(args.work) / s.scene_id
        if not (work / "splats.pt").exists():
            train_scene(s, ply, work, cfg, log_every=ITERS)
        gs, _ = gs_point_cloud(s, work / "splats.pt", cfg, alpha_min=0.7,
                               stride=2, min_views=1)
        d, _ = cKDTree(mvs).query(gs, workers=-1)

        for gap in (0.0, 0.10, 0.20, 0.40):
            add = gs if gap == 0.0 else gs[d > gap]
            x = voxel_downsample(np.vstack([mvs, add]), 0.02)
            r = score_geometry(x, gt, s.key)
            tag = "union_all" if gap == 0.0 else f"gapfill_{gap}"
            rows.append({"scene": s.key, "method": tag, "n": int(len(x)),
                         "added": int(len(add)), "fscore": r.fscore,
                         "p10": r.precision[0.10], "r10": r.recall[0.10]})
            print(f"  {tag:14s} +{len(add):8d} gs pts -> F={r.fscore:.4f} "
                  f"({r.fscore-base.fscore:+.4f})  P@10 {r.precision[0.10]:.3f} "
                  f"R@10 {r.recall[0.10]:.3f}", flush=True)
            (out / "scores.json").write_text(json.dumps(rows, indent=2))

    (out / "scores.json").write_text(json.dumps(rows, indent=2))
    by = {}
    for r in rows:
        by.setdefault(r["method"], []).append(r["fscore"])
    print("\nmean F by method:")
    for k, v in sorted(by.items(), key=lambda kv: -float(np.mean(kv[1]))):
        print(f"  {k:14s} {np.mean(v):.4f}")


if __name__ == "__main__":
    main()
