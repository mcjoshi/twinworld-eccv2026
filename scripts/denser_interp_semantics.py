"""Does the denser-interpolation coverage gain survive into achieved mIoU?

The coverage experiment passed its gate on both dev scenes: mean ceiling
0.445 -> 0.525, with wall +0.051/+0.113 and window +0.048/+0.124. That is the
largest movement in the semantic bottleneck since the packaging discovery, but
a ceiling is an upper bound and has repeatedly failed to convert in this project
(TSDF fusion, tilted views, the 2D-segmentation lift).

So this runs the real pipeline on the union cloud: same features, same
classifier, same leave-one-scene-out protocol, with the k=5 majority-vote
propagation adopted earlier today. The only change is the cloud.

Built in memory rather than via PLY, because the union clouds are ~700 MB each
and writing four of them is pure overhead.

Adoption gate, fixed before running: BOTH folds must beat their own baseline at
10 cm packaging with k=1 propagation (fold 009 > 0.1529, fold 010 > 0.2067,
LOSO > 0.1798). The k=5 baseline is 0.1814, so that is also reported - a result
that beats 0.1798 but not 0.1814 has not beaten the incumbent stack.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from twinworld.dataio import CLASS_NAMES, NUM_CLASSES, Scene
from twinworld.geometry.densify import robust_core
from twinworld.package import submission_cloud
from twinworld.scoring import (
    read_ply_xyz, score_semantics, voxel_downsample_indices,
)
from twinworld.semantics.classifier import ClassifierConfig, fit, predict
from twinworld.semantics.features import compute_features, estimate_up, transfer_labels

V1 = Path("experiments/2026-08-08-pipeline-v1")
INTERP = Path("experiments/2026-08-19-denser-interp")
OUT = Path("experiments/2026-08-19-denser-interp-semantics")
PACK_VOXEL = 0.10
BASE_K1 = {"gold_coast/scene_009": 0.1529, "gold_coast/scene_010": 0.2067}
BASE_LOSO_K1 = 0.1798
BASE_LOSO_K5 = 0.1814
K_PROP = 5


def propagate_k(thin_xyz, thin_labels, full_xyz, k=K_PROP):
    """k-nearest majority vote, the variant adopted on 19 Aug (LOSO 0.1814)."""
    if k == 1:
        _, idx = cKDTree(thin_xyz).query(full_xyz, workers=-1)
        return thin_labels[idx]
    _, idx = cKDTree(thin_xyz).query(full_xyz, k=k, workers=-1)
    lab = thin_labels[idx]
    votes = np.zeros((len(full_xyz), NUM_CLASSES), np.float32)
    for c in range(NUM_CLASSES):
        votes[:, c] = (lab == c).sum(axis=1)
    return np.argmax(votes, axis=1).astype(np.uint8)


def union_cloud(scene: Scene, cfg: ClassifierConfig):
    """Baseline submission cloud unioned with the denser-interp points.

    robust_core is applied to the new points so the far-field tail that
    `submission_cloud` removes from the stereo cloud cannot re-enter here.
    """
    ply = V1 / scene.dataset / scene.scene_id / "mvs" / "fused_gs_i.ply"
    base_xyz, base_rgb = submission_cloud(scene, ply, voxel=PACK_VOXEL)

    d = np.load(INTERP / f"{scene.scene_id}_interp.npz")
    nx, nc = d["xyz"].astype(np.float64), d["rgb"]
    keep = robust_core(nx, scene.train_frames)
    nx, nc = nx[keep], nc[keep]

    xyz = np.vstack([base_xyz, nx])
    rgb = np.vstack([base_rgb, nc]) if base_rgb is not None else None
    idx = voxel_downsample_indices(xyz, PACK_VOXEL)
    xyz = xyz[idx]
    rgb = rgb[idx] if rgb is not None else None
    return xyz, rgb, len(base_xyz)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = ClassifierConfig(max_feature_points=4_000_000)
    scenes = [Scene("gold_coast", s) for s in ("scene_009", "scene_010")]

    cache = {}
    for s in scenes:
        t0 = time.time()
        full, rgb, n_base = union_cloud(s, cfg)
        print(f"[{s.key}] union {len(full)/1e6:.1f}M pts "
              f"(baseline {n_base/1e6:.1f}M) ({time.time()-t0:.0f}s)", flush=True)

        idx = voxel_downsample_indices(full, cfg.feature_voxel)
        if len(idx) > cfg.max_feature_points:
            idx = np.sort(np.random.default_rng(cfg.seed).choice(
                idx, cfg.max_feature_points, replace=False))
        thin = full[idx]
        thin_rgb = rgb[idx] if rgb is not None else None

        up = estimate_up(s.train_frames)
        rig = np.stack([f.center for f in s.train_frames]).mean(axis=0)
        X, _names = compute_features(thin, thin_rgb, up, rig=rig)
        gt, gl = read_ply_xyz(s.gt_ply, with_classification=True)
        y = transfer_labels(gt, gl, thin, cfg.label_transfer_radius)
        cache[s.key] = (thin, X, y, gt, gl, full)
        print(f"[{s.key}] {len(thin)} feature pts, "
              f"{int((y < NUM_CLASSES).sum())} labelled ({time.time()-t0:.0f}s)",
              flush=True)

    fold = {}
    for held in scenes:
        t0 = time.time()
        tr = [s for s in scenes if s.key != held.key][0]
        model = fit(cache[tr.key][1], cache[tr.key][2], cfg)
        thin, X, _, gt, gl, full = cache[held.key]
        pred_thin = predict(model, X, n_jobs=cfg.n_jobs)
        pred = propagate_k(thin, pred_thin, full)
        c = score_semantics(full, pred, gt, gl)
        iou = c.per_class_iou
        fold[held.key] = {"miou": round(float(c.miou), 4),
                          "ceiling": round(float(c.miou_ceiling), 4),
                          "points": int(len(full)),
                          "per_class": {CLASS_NAMES[i]: round(float(iou[i]), 4)
                                        for i in range(NUM_CLASSES)},
                          "seconds": round(time.time() - t0, 1)}
        print(f"[held {held.key}] mIoU {c.miou:.4f} "
              f"(base {BASE_K1[held.key]:.4f}, {c.miou-BASE_K1[held.key]:+.4f}) "
              f"ceiling {c.miou_ceiling:.4f}  "
              + "  ".join(f"{CLASS_NAMES[i]} {iou[i]:.3f}"
                          for i in range(NUM_CLASSES))
              + f"  ({fold[held.key]['seconds']}s)", flush=True)

    loso = float(np.mean([f["miou"] for f in fold.values()]))
    both = all(fold[k]["miou"] > BASE_K1[k] for k in BASE_K1)
    row = {"cloud": "fused_gs_i + denser interp (per_pair=5, max_pairs=200)",
           "propagation": f"k={K_PROP} majority",
           "loso_miou": round(loso, 4), "both_folds": both,
           "vs_k1_baseline": round(loso - BASE_LOSO_K1, 4),
           "vs_k5_incumbent": round(loso - BASE_LOSO_K5, 4),
           "verdict": "ADOPT" if both and loso > BASE_LOSO_K5 else "REJECT",
           "folds": fold}
    print(f"\nLOSO {loso:.4f}  vs k1 baseline 0.1798 ({loso-BASE_LOSO_K1:+.4f})  "
          f"vs k5 incumbent 0.1814 ({loso-BASE_LOSO_K5:+.4f})  "
          f"both folds: {both} -> {row['verdict']}", flush=True)
    (OUT / "results.json").write_text(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
