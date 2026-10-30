"""Geometry bake-off: run one MVS config over scenes and score it locally.

Usage:
    python scripts/mvs_ablation.py --variant loose --gpu 2 --scenes tum/scene_000
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

from twinworld.dataio import Scene, dev_scenes
from twinworld.geometry.colmap_mvs import MvsConfig, run_mvs
from twinworld.scoring import read_ply_xyz, score_geometry

BASE = MvsConfig()

VARIANTS: dict[str, MvsConfig] = {
    # the first run: a single global depth range over the whole flight
    "fixeddepth": replace(BASE, auto_depth=False),
    # per-image depth ranges from the sparse model
    "auto": replace(BASE, auto_depth=True),
    # per-image depth + permissive fusion, trading precision for recall
    "loose": replace(
        BASE,
        auto_depth=True,
        fusion_min_num_pixels=2,
        fusion_max_reproj_error=3.0,
        fusion_max_depth_error=0.02,
        fusion_max_normal_error=20.0,
        filter_min_ncc=0.06,
        filter_min_num_consistent=1,
    ),
    # `loose` at 2x image resolution: ground sampling distance is 3.4 cm/px
    # against a 2 cm ground-truth scan, so native resolution may be what caps
    # F@5. Costs roughly 4x the runtime.
    "loose_2x": replace(
        BASE,
        auto_depth=True,
        image_scale=2.0,
        fusion_min_num_pixels=2,
        fusion_max_reproj_error=3.0,
        fusion_max_depth_error=0.02,
        fusion_max_normal_error=20.0,
        filter_min_ncc=0.06,
        filter_min_num_consistent=1,
    ),
    # Coverage-maximal, intended for Gold Coast only. Geometric precision is
    # never scored there, but the semantic score is capped by how much ground
    # truth has any submitted point within 10 cm, so surplus and slightly noisy
    # points are close to free while missing points are not.
    "coverage": replace(
        BASE,
        auto_depth=True,
        fusion_min_num_pixels=1,
        fusion_max_reproj_error=4.0,
        fusion_max_depth_error=0.05,
        fusion_max_normal_error=30.0,
        filter_min_ncc=0.06,
        filter_min_num_consistent=1,
        # NOT lowered below 1.0. The first version used 0.5 and it was a
        # mistake: a point triangulated at half a degree has effectively
        # unbounded depth uncertainty, so 1 percent of the resulting Gold Coast
        # cloud landed beyond 3.5 km and the furthest point was 1.2e10 m out.
        # Loosening the photometric and fusion filters adds coverage; loosening
        # the triangulation angle only adds noise at arbitrary depth.
        filter_min_triangulation_angle=1.0,
    ),
    # `coverage` at 2x image resolution. 2x was worth +0.058 mean F on TUM,
    # almost all of it recall, and Gold Coast is bounded by exactly that.
    "coverage_2x": replace(
        BASE,
        auto_depth=True,
        image_scale=2.0,
        fusion_min_num_pixels=1,
        fusion_max_reproj_error=4.0,
        fusion_max_depth_error=0.05,
        fusion_max_normal_error=30.0,
        filter_min_ncc=0.06,
        filter_min_num_consistent=1,
        filter_min_triangulation_angle=1.0,
    ),
    # 4x resolution. 2x was worth +0.058 mean F, but the original images are
    # 1600 px wide and 2x already puts the effective sampling below the 2 cm
    # ground truth, so this tests whether there is any information left to
    # extract or whether 2x is where the sensor runs out. Costs 16x native.
    "loose_4x": replace(
        BASE,
        auto_depth=True,
        image_scale=4.0,
        fusion_min_num_pixels=2,
        fusion_max_reproj_error=3.0,
        fusion_max_depth_error=0.02,
        fusion_max_normal_error=20.0,
        filter_min_ncc=0.06,
        filter_min_num_consistent=1,
    ),
    # as `loose`, plus a smaller matching window for finer structure
    "loose_fine": replace(
        BASE,
        auto_depth=True,
        window_radius=3,
        num_iterations=7,
        fusion_min_num_pixels=2,
        fusion_max_reproj_error=3.0,
        fusion_max_depth_error=0.02,
        fusion_max_normal_error=20.0,
        filter_min_ncc=0.06,
        filter_min_num_consistent=1,
    ),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    ap.add_argument("--gpu", type=int, default=2)
    ap.add_argument("--scenes", nargs="*", default=None, help="keys like tum/scene_000")
    ap.add_argument("--exp", default="experiments/2026-08-08-geometry-bakeoff")
    args = ap.parse_args()

    if args.scenes:
        scenes = [Scene(*k.split("/")) for k in args.scenes]
    else:
        scenes = [s for s in dev_scenes() if s.dataset == "tum"]

    cfg = VARIANTS[args.variant]
    exp = Path(args.exp) / args.variant
    exp.mkdir(parents=True, exist_ok=True)
    (exp / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    rows = []
    for s in scenes:
        out = exp / s.dataset / s.scene_id
        t0 = time.time()
        summary = run_mvs(s, out, cfg, gpu_index=args.gpu, tag=args.variant)
        row = {"variant": args.variant, **summary}
        if s.has_gt_cloud:
            gt, _ = read_ply_xyz(s.gt_ply)
            pred, _ = read_ply_xyz(out / "fused.ply")
            r = score_geometry(pred, gt, s.key)
            row |= {
                "f5": r.f_scores[0.05], "f10": r.f_scores[0.10], "f20": r.f_scores[0.20],
                "fscore": r.fscore, "p10": r.precision[0.10], "r10": r.recall[0.10],
                "n_pred_voxels": r.n_pred, "n_gt_voxels": r.n_gt,
            }
            print(
                f"[{args.variant}] {s.key}  N={summary['n_points']:8d}  "
                f"F@5={r.f_scores[0.05]:.4f} F@10={r.f_scores[0.10]:.4f} "
                f"F@20={r.f_scores[0.20]:.4f}  mean={r.fscore:.4f}  "
                f"P@10={r.precision[0.10]:.3f} R@10={r.recall[0.10]:.3f}  "
                f"({time.time()-t0:.0f}s)",
                flush=True,
            )
        else:
            print(f"[{args.variant}] {s.key}  N={summary['n_points']:8d} (no GT)", flush=True)
        rows.append(row)
        (exp / "scores.json").write_text(json.dumps(rows, indent=2))

    scored = [r for r in rows if "fscore" in r]
    if scored:
        mean = sum(r["fscore"] for r in scored) / len(scored)
        print(f"[{args.variant}] MEAN geometry_score over {len(scored)} scenes = {mean:.4f}")
        (exp / "scores.json").write_text(
            json.dumps({"rows": rows, "mean_fscore": mean}, indent=2)
        )


if __name__ == "__main__":
    main()
