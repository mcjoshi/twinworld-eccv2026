"""Does the splatting model's own depth recover geometry that stereo missed?

Gold Coast wall coverage at 10 cm is 0.02 from PatchMatch stereo against 0.45
for roofs, yet the facades are visible (median 41 degree incidence) and the
splatting model renders the training views at 25-35 dB. If the radiance field
holds that geometry, rendering its depth should recover it.

Scored on what each dataset is actually judged by: F-score on TUM, semantic
coverage ceiling on Gold Coast.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from twinworld.dataio import CLASS_NAMES, NUM_CLASSES, Scene
from twinworld.geometry.gs_depth import gs_point_cloud
from twinworld.package import submission_cloud
from twinworld.render.trainer import TrainConfig, train_scene
from twinworld.scoring import (
    read_ply_xyz, score_geometry, score_semantics, voxel_downsample,
)

DATASET_CFG = {"tum": {"init": "mvs", "iters": 3000}, "gold_coast": {"init": "sparse", "iters": 7000}}


def evaluate(name, xyz, s, gt, gl, rows, voxel):
    xyz = voxel_downsample(xyz, voxel)
    if s.dataset == "tum":
        r = score_geometry(xyz, gt, s.key)
        rows.append({"method": name, "n": int(len(xyz)), "fscore": r.fscore,
                     "f5": r.f_scores[0.05], "f10": r.f_scores[0.10], "f20": r.f_scores[0.20],
                     "p10": r.precision[0.10], "r10": r.recall[0.10]})
        print(f"  {name:26s} N={len(xyz):9d}  F={r.fscore:.4f} "
              f"(F@5 {r.f_scores[0.05]:.3f} F@10 {r.f_scores[0.10]:.3f} F@20 {r.f_scores[0.20]:.3f}) "
              f"P@10 {r.precision[0.10]:.3f} R@10 {r.recall[0.10]:.3f}", flush=True)
        return r.fscore
    c = score_semantics(xyz, np.zeros(len(xyz), np.uint8), gt, gl)
    cov = c.per_class_coverage
    rows.append({"method": name, "n": int(len(xyz)), "miou_ceiling": c.miou_ceiling,
                 "coverage": {CLASS_NAMES[i]: round(float(cov[i]), 4) for i in range(NUM_CLASSES)}})
    print(f"  {name:26s} N={len(xyz):9d}  ceiling {c.miou_ceiling:.4f}   "
          + "  ".join(f"{CLASS_NAMES[i]} {cov[i]:.3f}" for i in range(NUM_CLASSES)), flush=True)
    return c.miou_ceiling


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--src", default="experiments/2026-08-08-pipeline-v1")
    ap.add_argument("--out", default="experiments/2026-08-10-gs-depth")
    ap.add_argument("--work", default="work/gsdepth")
    ap.add_argument("--stride", type=int, default=2)
    args = ap.parse_args()

    s = Scene(*args.scene.split("/"))
    d = DATASET_CFG[s.dataset]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    voxel = 0.02 if s.dataset == "tum" else 0.06

    mvs_ply = Path(args.src) / s.dataset / s.scene_id / "mvs" / "fused.ply"
    mvs, _ = submission_cloud(s, mvs_ply, cache=mvs_ply.parent / "submission_cloud.npz")
    if s.dataset == "tum":
        gt, gl = read_ply_xyz(s.gt_ply), None
        gt = gt[0]
    else:
        gt, gl = read_ply_xyz(s.gt_ply, with_classification=True)

    rows: list[dict] = []
    print(f"\n[{s.key}]  baseline: MVS only", flush=True)
    base = evaluate("mvs_only", mvs, s, gt, gl, rows, voxel)

    # train once, keeping the checkpoint so depth can be rendered from it
    work = Path(args.work) / s.dataset / s.scene_id
    cfg = TrainConfig(iters=d["iters"], refine_stop_iter=int(d["iters"] * 0.72))
    if not (work / "splats.pt").exists():
        init = s.sparse_ply if d["init"] == "sparse" else mvs_ply
        t0 = time.time()
        train_scene(s, init, work, cfg, log_every=d["iters"])
        print(f"  trained in {time.time()-t0:.0f}s", flush=True)

    for alpha_min, min_views in ((0.7, 2), (0.5, 2), (0.9, 2), (0.7, 3), (0.7, 1)):
        t0 = time.time()
        pts, _ = gs_point_cloud(s, work / "splats.pt", cfg, alpha_min=alpha_min,
                                stride=args.stride, min_views=min_views)
        print(f"  gs depth alpha>={alpha_min} views>={min_views}: {len(pts)} points "
              f"({time.time()-t0:.0f}s)", flush=True)
        if len(pts) == 0:
            continue
        got = evaluate(f"gs_a{alpha_min}_v{min_views}", pts, s, gt, gl, rows, voxel)
        union = evaluate(f"gs_a{alpha_min}_v{min_views}+mvs", np.vstack([mvs, pts]),
                         s, gt, gl, rows, voxel)
        print(f"      gs alone {got-base:+.4f} vs MVS, union {union-base:+.4f}", flush=True)
        (out / f"{s.scene_id}.json").write_text(json.dumps(rows, indent=2))

    (out / f"{s.scene_id}.json").write_text(json.dumps(rows, indent=2))
    key = "fscore" if s.dataset == "tum" else "miou_ceiling"
    best = max(rows, key=lambda r: r[key])
    print(f"\n  best: {best['method']} at {best[key]:.4f} (MVS alone {base:.4f})")


if __name__ == "__main__":
    main()
