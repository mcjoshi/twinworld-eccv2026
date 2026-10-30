"""Does Poisson surface densification improve the geometry score over raw MVS?

Raw MVS on TUM scene_000 scores F@5 0.18 / F@10 0.55 / F@20 0.72. Precision is
limited by per-point depth noise and recall by unobserved facades. Surface
fitting should help the first and partly the second. This measures it rather
than assuming it, and sweeps the two settings that trade the effects off:
Poisson `depth` (surface resolution) and `trim_distance` (how far from real
observations invented surface is allowed to survive).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from twinworld.dataio import Scene
from twinworld.geometry.densify import (
    confine_to_training_views, poisson_densify, read_cloud_with_normals, robust_core,
)
from twinworld.scoring import read_ply_xyz, score_geometry, write_ply_xyz

SWEEP = [
    {"target_cell": 0.04, "trim_distance": 0.15, "density_quantile": 0.10},
    {"target_cell": 0.04, "trim_distance": 0.08, "density_quantile": 0.10},
    {"target_cell": 0.04, "trim_distance": 0.40, "density_quantile": 0.05},
    {"target_cell": 0.08, "trim_distance": 0.15, "density_quantile": 0.10},
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="auto")
    ap.add_argument("--exp", default="experiments/2026-08-08-geometry-bakeoff")
    ap.add_argument("--scenes", nargs="*", default=["tum/scene_000", "tum/scene_001"])
    ap.add_argument("--out", default="experiments/2026-08-08-densify-ablation")
    ap.add_argument("--sample-spacing", type=float, default=0.02)
    ap.add_argument("--save-best-ply", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []

    for key in args.scenes:
        s = Scene(*key.split("/"))
        src = Path(args.exp) / args.variant / s.dataset / s.scene_id / "fused.ply"
        if not src.exists():
            print(f"skip {key}: {src} missing", flush=True)
            continue
        xyz, nrm, rgb = read_cloud_with_normals(src)
        gt, _ = read_ply_xyz(s.gt_ply)
        core = robust_core(xyz, s.train_frames)
        print(f"[{s.key}] robust core keeps {core.mean():.4f} of {len(xyz)} points, "
              f"extent {(xyz[core].max(0)-xyz[core].min(0)).round(1)} "
              f"(full {(xyz.max(0)-xyz.min(0)).round(1)})", flush=True)

        base = score_geometry(xyz, gt, s.key)
        rows.append({"scene": s.key, "method": "mvs_raw", "n": int(len(xyz)),
                     "f5": base.f_scores[0.05], "f10": base.f_scores[0.10],
                     "f20": base.f_scores[0.20], "fscore": base.fscore,
                     "p10": base.precision[0.10], "r10": base.recall[0.10]})
        print(f"[{s.key}] mvs_raw            N={len(xyz):8d} F={base.fscore:.4f} "
              f"(F@5 {base.f_scores[0.05]:.3f} F@10 {base.f_scores[0.10]:.3f} "
              f"F@20 {base.f_scores[0.20]:.3f}) P@10 {base.precision[0.10]:.3f} "
              f"R@10 {base.recall[0.10]:.3f}", flush=True)

        for params in SWEEP:
            t0 = time.time()
            try:
                pts, cols, info = poisson_densify(
                    xyz[core], nrm[core] if nrm is not None else None,
                    rgb[core] if rgb is not None else None,
                    sample_spacing=args.sample_spacing, **params,
                )
            except Exception as e:  # noqa: BLE001
                print(f"[{s.key}] poisson {params} FAILED: {e}", flush=True)
                continue
            pts_c, _ = confine_to_training_views(s, pts)
            r = score_geometry(pts_c, gt, s.key)
            tag = f"poisson_c{params['target_cell']}_t{params['trim_distance']}"
            rows.append({"scene": s.key, "method": tag, "n": int(len(pts_c)),
                         "f5": r.f_scores[0.05], "f10": r.f_scores[0.10],
                         "f20": r.f_scores[0.20], "fscore": r.fscore,
                         "p10": r.precision[0.10], "r10": r.recall[0.10],
                         "params": params, "info": info,
                         "seconds": round(time.time() - t0, 1)})
            print(f"[{s.key}] {tag:18s} N={len(pts_c):8d} F={r.fscore:.4f} "
                  f"(F@5 {r.f_scores[0.05]:.3f} F@10 {r.f_scores[0.10]:.3f} "
                  f"F@20 {r.f_scores[0.20]:.3f}) P@10 {r.precision[0.10]:.3f} "
                  f"R@10 {r.recall[0.10]:.3f}  ({time.time()-t0:.0f}s)", flush=True)

            # union of raw MVS and the fitted surface: keeps observed detail and
            # adds interpolated coverage
            union = np.vstack([xyz, pts_c])
            ru = score_geometry(union, gt, s.key)
            rows.append({"scene": s.key, "method": tag + "_union", "n": int(len(union)),
                         "f5": ru.f_scores[0.05], "f10": ru.f_scores[0.10],
                         "f20": ru.f_scores[0.20], "fscore": ru.fscore,
                         "p10": ru.precision[0.10], "r10": ru.recall[0.10]})
            print(f"[{s.key}] {tag+'_union':18s} N={len(union):8d} F={ru.fscore:.4f} "
                  f"(F@5 {ru.f_scores[0.05]:.3f} F@10 {ru.f_scores[0.10]:.3f} "
                  f"F@20 {ru.f_scores[0.20]:.3f}) P@10 {ru.precision[0.10]:.3f} "
                  f"R@10 {ru.recall[0.10]:.3f}", flush=True)

            (out / "scores.json").write_text(json.dumps(rows, indent=2))

    (out / "scores.json").write_text(json.dumps(rows, indent=2))
    best = {}
    for r in rows:
        best.setdefault(r["method"], []).append(r["fscore"])
    print("\nmean F-score by method:")
    for k, v in sorted(best.items(), key=lambda kv: -float(np.mean(kv[1]))):
        print(f"  {k:26s} {np.mean(v):.4f}  (n={len(v)})")


if __name__ == "__main__":
    main()
