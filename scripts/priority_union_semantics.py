"""Priority union: add coverage without evicting accurate points.

The denser-interpolation experiment raised the mean coverage ceiling from 0.445
to 0.525 and *lowered* leave-one-scene-out mIoU from 0.1798 to 0.1715. The
per-class breakdown named the mechanism:

    wall  0.121 -> 0.155   (+0.034, coverage paid off exactly as predicted)
    roof  0.443 -> 0.412   (-0.031)
    other 0.039 -> 0.025   (-0.014)

`voxel_downsample` keeps one point per occupied cell - the point nearest that
cell's centre. Unioning the stereo cloud with splat-rendered points and then
downsampling therefore lets a noisy rendered point *evict* an accurate stereo
point whenever it happens to sit closer to the cell centre. The rendered depth
is less accurate than MVS, so in cells where both exist the union was trading
precision for nothing: the cell was already covered.

Priority union fixes exactly that. Voxelise the stereo cloud first and keep all
of it; then add rendered points only in cells the stereo cloud does not occupy.
Coverage can only rise, and no accurate point is ever displaced.

Variants:
  priority        stereo wins every contested cell
  priority_8M     same, with the feature budget raised to 8 M - the denser cloud
                  diluted supervision at a fixed 4 M budget (labelled points fell
                  from 57 952 to 52 971), which may itself explain part of the loss

Adoption gate, fixed before running: BOTH folds must beat their own baseline
(fold 009 > 0.1529, fold 010 > 0.2067) and LOSO must beat the k=5 incumbent of
0.1814, not merely the 0.1798 k=1 figure.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from twinworld.dataio import CLASS_NAMES, NUM_CLASSES, Scene
from twinworld.geometry.densify import robust_core
from twinworld.package import submission_cloud
from twinworld.scoring import read_ply_xyz, score_semantics, voxel_downsample_indices
from twinworld.semantics.classifier import ClassifierConfig, fit, predict
from twinworld.semantics.features import compute_features, estimate_up, transfer_labels

V1 = Path("experiments/2026-08-08-pipeline-v1")
INTERP = Path("experiments/2026-08-19-denser-interp")
OUT = Path("experiments/2026-08-19-priority-union")
PACK_VOXEL = 0.10
BASE_K1 = {"gold_coast/scene_009": 0.1529, "gold_coast/scene_010": 0.2067}
BASE_LOSO_K5 = 0.1814
K_PROP = 5


def propagate_k(thin_xyz, thin_labels, full_xyz, k=K_PROP):
    _, idx = cKDTree(thin_xyz).query(full_xyz, k=k, workers=-1)
    lab = thin_labels[idx]
    votes = np.zeros((len(full_xyz), NUM_CLASSES), np.float32)
    for c in range(NUM_CLASSES):
        votes[:, c] = (lab == c).sum(axis=1)
    return np.argmax(votes, axis=1).astype(np.uint8)


def priority_union(scene: Scene):
    """Stereo cloud kept whole; rendered points only in cells it does not own."""
    ply = V1 / scene.dataset / scene.scene_id / "mvs" / "fused_gs_i.ply"
    base_xyz, base_rgb = submission_cloud(scene, ply, voxel=PACK_VOXEL)

    d = np.load(INTERP / f"{scene.scene_id}_interp.npz")
    nx, nc = d["xyz"].astype(np.float64), d["rgb"]
    keep = robust_core(nx, scene.train_frames)
    nx, nc = nx[keep], nc[keep]

    # thin the rendered points to the same grid first, so at most one candidate
    # competes for each empty cell
    idx = voxel_downsample_indices(nx, PACK_VOXEL)
    nx, nc = nx[idx], nc[idx]

    def keys(p):
        return np.floor(p / PACK_VOXEL).astype(np.int64)

    base_keys = keys(base_xyz)
    new_keys = keys(nx)
    # structured view so np.isin works on whole coordinate triples
    bview = np.ascontiguousarray(base_keys).view(
        np.dtype((np.void, base_keys.dtype.itemsize * 3))).ravel()
    nview = np.ascontiguousarray(new_keys).view(
        np.dtype((np.void, new_keys.dtype.itemsize * 3))).ravel()
    free = ~np.isin(nview, bview)

    xyz = np.vstack([base_xyz, nx[free]])
    rgb = np.vstack([base_rgb, nc[free]]) if base_rgb is not None else None
    return xyz, rgb, len(base_xyz), int(free.sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-feature-points", type=int, default=4_000_000)
    ap.add_argument("--tag", default="priority")
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    cfg = ClassifierConfig(max_feature_points=a.max_feature_points)
    scenes = [Scene("gold_coast", s) for s in ("scene_009", "scene_010")]

    cache = {}
    for s in scenes:
        t0 = time.time()
        full, rgb, n_base, n_added = priority_union(s)
        print(f"[{a.tag}][{s.key}] {len(full)/1e6:.1f}M pts = {n_base/1e6:.1f}M stereo "
              f"+ {n_added/1e6:.1f}M rendered into empty cells "
              f"(no eviction) ({time.time()-t0:.0f}s)", flush=True)

        idx = voxel_downsample_indices(full, cfg.feature_voxel)
        if len(idx) > cfg.max_feature_points:
            idx = np.sort(np.random.default_rng(cfg.seed).choice(
                idx, cfg.max_feature_points, replace=False))
        thin = full[idx]
        thin_rgb = rgb[idx] if rgb is not None else None
        up = estimate_up(s.train_frames)
        rig = np.stack([f.center for f in s.train_frames]).mean(axis=0)
        X, _ = compute_features(thin, thin_rgb, up, rig=rig)
        gt, gl = read_ply_xyz(s.gt_ply, with_classification=True)
        y = transfer_labels(gt, gl, thin, cfg.label_transfer_radius)
        cache[s.key] = (thin, X, y, gt, gl, full)
        print(f"[{a.tag}][{s.key}] {len(thin)} feature pts, "
              f"{int((y < NUM_CLASSES).sum())} labelled ({time.time()-t0:.0f}s)",
              flush=True)

    fold = {}
    for held in scenes:
        t0 = time.time()
        tr = [s for s in scenes if s.key != held.key][0]
        model = fit(cache[tr.key][1], cache[tr.key][2], cfg)
        thin, X, _, gt, gl, full = cache[held.key]
        pred = propagate_k(thin, predict(model, X, n_jobs=cfg.n_jobs), full)
        c = score_semantics(full, pred, gt, gl)
        iou = c.per_class_iou
        fold[held.key] = {"miou": round(float(c.miou), 4),
                          "ceiling": round(float(c.miou_ceiling), 4),
                          "points": int(len(full)),
                          "per_class": {CLASS_NAMES[i]: round(float(iou[i]), 4)
                                        for i in range(NUM_CLASSES)},
                          "seconds": round(time.time() - t0, 1)}
        print(f"[{a.tag}][held {held.key}] mIoU {c.miou:.4f} "
              f"(base {BASE_K1[held.key]:.4f}, {c.miou-BASE_K1[held.key]:+.4f}) "
              f"ceiling {c.miou_ceiling:.4f}  "
              + "  ".join(f"{CLASS_NAMES[i]} {iou[i]:.3f}"
                          for i in range(NUM_CLASSES)), flush=True)

    loso = float(np.mean([f["miou"] for f in fold.values()]))
    both = all(fold[k]["miou"] > BASE_K1[k] for k in BASE_K1)
    row = {"tag": a.tag, "max_feature_points": a.max_feature_points,
           "propagation": f"k={K_PROP}", "loso_miou": round(loso, 4),
           "both_folds": both, "vs_k5_incumbent": round(loso - BASE_LOSO_K5, 4),
           "verdict": "ADOPT" if both and loso > BASE_LOSO_K5 else "REJECT",
           "folds": fold}
    print(f"\n[{a.tag}] LOSO {loso:.4f} vs k5 incumbent {BASE_LOSO_K5:.4f} "
          f"({loso-BASE_LOSO_K5:+.4f}) both folds: {both} -> {row['verdict']}",
          flush=True)
    (OUT / f"{a.tag}.json").write_text(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
