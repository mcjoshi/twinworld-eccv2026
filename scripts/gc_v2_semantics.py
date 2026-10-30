"""Does adding modelled facades actually raise the *achieved* Gold Coast mIoU?

The ceiling says facades help: 0.1751 -> 0.2167 on scene_009 at full density.
But a ceiling is what a perfect labelling would reach, and the extruded points
are synthetic, with no colour of their own, while roughly half the classifier's
features are colour. If those points get labelled badly they can be worse than
useless, because a wrongly-labelled point sitting closer to a ground-truth point
than a correctly-labelled one actively steals the match.

So this measures the thing that decides it: leave-one-scene-out mIoU on the two
labelled scenes, with and without facades, using the same classifier both times.

Extruded points inherit colour from their nearest reconstructed point, which is
the honest choice: it is the appearance evidence actually available at that
location, and it keeps the feature distribution close to what the classifier was
trained on.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from twinworld.dataio import CLASS_NAMES, NUM_CLASSES, Scene
from twinworld.geometry.facade import extrude_facades
from twinworld.package import submission_cloud
from twinworld.scoring import (
    read_ply_xyz, score_semantics, voxel_downsample_indices,
)
from twinworld.semantics.classifier import ClassifierConfig, fit, predict
from twinworld.semantics.features import compute_features, estimate_up, transfer_labels


def build_cloud(scene: Scene, fused_ply: Path, with_facades: bool, spacing: float,
                cell: float, voxel: float):
    """Submission cloud, optionally with extruded facades merged in."""
    from scipy.spatial import cKDTree

    xyz, rgb = submission_cloud(scene, fused_ply)
    if not with_facades:
        return xyz, rgb

    up = estimate_up(scene.train_frames)
    fac = extrude_facades(xyz, up, cell=cell, min_building_height=2.5,
                          step_threshold=1.5, point_spacing=spacing,
                          max_points=120_000_000)
    if len(fac) == 0:
        return xyz, rgb
    # colour from the nearest reconstructed point
    _, idx = cKDTree(xyz).query(fac, workers=-1)
    fac_rgb = rgb[idx] if rgb is not None else None

    merged = np.vstack([xyz, fac])
    merged_rgb = np.vstack([rgb, fac_rgb]) if rgb is not None else None
    keep = voxel_downsample_indices(merged, voxel)
    return merged[keep], (merged_rgb[keep] if merged_rgb is not None else None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="experiments/2026-08-08-pipeline-v1")
    ap.add_argument("--out", default="experiments/2026-08-08-gc-v2-semantics")
    ap.add_argument("--spacing", type=float, default=0.10)
    ap.add_argument("--cell", type=float, default=0.10)
    ap.add_argument("--voxel", type=float, default=0.06)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = ClassifierConfig()
    scenes = [Scene("gold_coast", s) for s in ("scene_009", "scene_010")]
    results = {}

    for with_fac in (False, True):
        tag = "with_facades" if with_fac else "mvs_only"
        print(f"\n===== {tag} =====", flush=True)
        cache = {}
        for s in scenes:
            t0 = time.time()
            ply = Path(args.run) / s.dataset / s.scene_id / "mvs" / "fused.ply"
            xyz, rgb = build_cloud(s, ply, with_fac, args.spacing, args.cell, args.voxel)
            up = estimate_up(s.train_frames)
            rig = np.stack([fr.center for fr in s.train_frames]).mean(axis=0)
            X, _ = compute_features(xyz, rgb, up, rig=rig)
            gt_xyz, gt_lab = read_ply_xyz(s.gt_ply, with_classification=True)
            y = transfer_labels(gt_xyz, gt_lab, xyz, cfg.label_transfer_radius)
            cache[s.key] = (xyz, X, y, gt_xyz, gt_lab)
            print(f"  [{s.key}] {len(xyz)} pts, {X.shape[1]} features, "
                  f"{int((y < NUM_CLASSES).sum())} labelled ({time.time()-t0:.0f}s)", flush=True)

        folds = []
        for held in scenes:
            t0 = time.time()
            tr = [s for s in scenes if s.key != held.key]
            model = fit(np.concatenate([cache[s.key][1] for s in tr]),
                        np.concatenate([cache[s.key][2] for s in tr]), cfg)
            xyz, X, _, gt_xyz, gt_lab = cache[held.key]
            pred = predict(model, X, n_jobs=cfg.n_jobs)
            c = score_semantics(xyz, pred, gt_xyz, gt_lab)
            iou = c.per_class_iou
            folds.append({"held_out": held.key, "miou": c.miou,
                          "miou_ceiling": c.miou_ceiling,
                          "per_class_iou": {CLASS_NAMES[i]: round(float(iou[i]), 4)
                                            for i in range(NUM_CLASSES)},
                          "n_points": int(len(xyz)),
                          "approx_mb": round(len(xyz) * 13 / 1e6, 1)})
            print(f"  [held out {held.key}] mIoU {c.miou:.4f} of ceiling "
                  f"{c.miou_ceiling:.4f}   "
                  + "  ".join(f"{CLASS_NAMES[i]} {iou[i]:.3f}" for i in range(NUM_CLASSES))
                  + f"   ({time.time()-t0:.0f}s)", flush=True)

        results[tag] = {"folds": folds,
                        "miou_mean": float(np.mean([f["miou"] for f in folds])),
                        "ceiling_mean": float(np.mean([f["miou_ceiling"] for f in folds])),
                        "mb_mean": float(np.mean([f["approx_mb"] for f in folds]))}
        print(f"  {tag}: mean mIoU {results[tag]['miou_mean']:.4f} "
              f"(ceiling {results[tag]['ceiling_mean']:.4f}, "
              f"{results[tag]['mb_mean']:.0f} MB/scene)", flush=True)
        (out / "results.json").write_text(json.dumps(
            {"config": asdict(cfg), "args": vars(args), "results": results}, indent=2))

    if len(results) == 2:
        d = results["with_facades"]["miou_mean"] - results["mvs_only"]["miou_mean"]
        extra = results["with_facades"]["mb_mean"] - results["mvs_only"]["mb_mean"]
        print(f"\nfacades change achieved mIoU by {d:+.4f} for {extra:+.0f} MB per scene")
        print("-> ship facades" if d > 0.005 else "-> not worth it")


if __name__ == "__main__":
    main()
