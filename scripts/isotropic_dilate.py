"""Isotropic facade dilation: test a normal-free uncertainty shell.

The scorer assigns every ground-truth point the nearest submitted label within
10 cm and charges no precision penalty. Facade ground truth is a median 18 cm
from the submitted wall surface, but local PCA normals are noisy on sparse
facades. Rather than choose one possibly wrong normal direction, emit an
axis-aligned octahedral shell around every facade point.

Each selected point gets six extra points: +/- radius along x, y, and z. The
facade selection, source clouds, packaging voxel, classifier, propagation, and
leave-one-scene-out evaluation are intentionally identical to offset_sheets.py.

Pre-registered adoption gate, before running: fold 009 must beat 0.1529, fold
010 must beat 0.2067, and LOSO must exceed 0.1931. Mean-only is REJECT.
Every configuration reports its scored point count, multiplier, and
score-per-point-multiplier; 108.6M points failed server packaging while 55M
passed, so a high-cost winner is not useful.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from twinworld.dataio import CLASS_NAMES, NUM_CLASSES, Scene
from twinworld.package import submission_cloud
from twinworld.scoring import read_ply_xyz, score_semantics, voxel_downsample_indices
from twinworld.semantics.classifier import (
    ClassifierConfig, fit, predict, thinned_features, transfer_labels,
)
from twinworld.semantics.features import estimate_up

V1 = Path("experiments/2026-08-08-pipeline-v1")
OUT = Path("experiments/2026-08-20-isotropic")
PACK_VOXEL = 0.10
MATCH_R = 0.10
BASE = {"gold_coast/scene_009": 0.1529, "gold_coast/scene_010": 0.2067}
INCUMBENT = 0.1931          # mesh densification, adopted 20 Aug
K_PROP = 5


def propagate_k(thin_xyz, thin_labels, full_xyz, k=K_PROP):
    _, idx = cKDTree(thin_xyz).query(full_xyz, k=k, workers=-1)
    lab = thin_labels[idx]
    votes = np.zeros((len(full_xyz), NUM_CLASSES), np.float32)
    for c in range(NUM_CLASSES):
        votes[:, c] = (lab == c).sum(axis=1)
    return np.argmax(votes, axis=1).astype(np.uint8)


def per_class_coverage(sub, gt, gl):
    d, _ = cKDTree(sub).query(gt, k=1, distance_upper_bound=MATCH_R, workers=-1)
    m = np.isfinite(d)
    out = {CLASS_NAMES[c]: round(float(m[gl == c].mean()), 4)
           for c in range(NUM_CLASSES) if (gl == c).any()}
    out["ceiling"] = round(float(np.mean(list(out.values()))), 4)
    return out


def normals(xyz, k=24):
    """Local PCA normals. The smallest-eigenvalue direction is the surface normal."""
    tree = cKDTree(xyz)
    n = np.empty_like(xyz)
    CH = 400_000
    for a in range(0, len(xyz), CH):
        b = min(a + CH, len(xyz))
        _, idx = tree.query(xyz[a:b], k=k, workers=-1)
        nb = xyz[idx]                                  # (m, k, 3)
        nb = nb - nb.mean(axis=1, keepdims=True)
        cov = np.einsum("mki,mkj->mij", nb, nb) / k
        w, v = np.linalg.eigh(cov)
        n[a:b] = v[:, :, 0]
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius", type=float, default=0.15,
                    help="octahedral-shell radius in metres")
    ap.add_argument("--vertical-thresh", type=float, default=0.5,
                    help="|normal . up| below this counts as a facade")
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--base-voxel", type=float, default=PACK_VOXEL,
                    help="voxel for the base cloud before sheets are added; a\n"
                         "coarser base frees budget for the sheets that do the\n"
                         "actual covering")
    ap.add_argument("--tag", default="dilate015")
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    cfg = ClassifierConfig(max_feature_points=4_000_000, learning_rate=a.lr)
    scenes = [Scene("gold_coast", s) for s in ("scene_009", "scene_010")]
    ply = {s.key: V1 / s.dataset / s.scene_id / "mvs" / "fused_gs_i.ply"
           for s in scenes}

    cache = {}
    for s in scenes:
        t0 = time.time()
        base, _ = submission_cloud(s, ply[s.key], voxel=a.base_voxel)
        gt, gl = read_ply_xyz(s.gt_ply, with_classification=True)
        cov0 = per_class_coverage(base, gt, gl)
        print(f"[{a.tag}][{s.key}] base {len(base)/1e6:.1f}M  "
              + "  ".join(f"{k} {v:.3f}" for k, v in cov0.items()), flush=True)

        nrm = normals(base)
        up = estimate_up(s.train_frames)
        vertical = np.abs(nrm @ up) < a.vertical_thresh
        print(f"[{a.tag}][{s.key}] {vertical.mean()*100:.1f}% of points are "
              f"near-vertical (facade candidates) ({time.time()-t0:.0f}s)",
              flush=True)

        sheets = [base]
        fx = base[vertical]
        for axis in np.eye(3, dtype=base.dtype):
            sheets.append(fx + axis * a.radius)
            sheets.append(fx - axis * a.radius)
        full = np.vstack(sheets)
        idx = voxel_downsample_indices(full, a.base_voxel)
        full = full[idx]
        cov1 = per_class_coverage(full, gt, gl)
        print(f"[{a.tag}][{s.key}] with shell {len(full)/1e6:.1f}M pts "
              f"({len(full)/len(base):.2f}x)  "
              + "  ".join(f"{k} {cov1[k]:.3f} ({cov1[k]-cov0[k]:+.3f})"
                          for k in cov0), flush=True)
        cache[s.key] = (full, gt, gl, cov0, cov1, len(base))

    feat = {}
    for s in scenes:
        thin, X, _, _ = thinned_features(
            s, ply[s.key], ClassifierConfig(max_feature_points=4_000_000))
        gt, gl = cache[s.key][1], cache[s.key][2]
        y = transfer_labels(gt, gl, thin, cfg.label_transfer_radius)
        feat[s.key] = (thin, X, y)
        print(f"[{a.tag}][{s.key}] {len(thin)} feature pts, "
              f"{int((y < NUM_CLASSES).sum())} labelled", flush=True)

    fold = {}
    for held in scenes:
        t0 = time.time()
        tr = [s for s in scenes if s.key != held.key][0]
        model = fit(feat[tr.key][1], feat[tr.key][2], cfg)
        thin, X, _ = feat[held.key]
        full, gt, gl, cov0, cov1, n_base = cache[held.key]
        pred = propagate_k(thin, predict(model, X, n_jobs=cfg.n_jobs), full)
        c = score_semantics(full, pred, gt, gl)
        iou = c.per_class_iou
        fold[held.key] = {"miou": round(float(c.miou), 4),
                          "ceiling": round(float(c.miou_ceiling), 4),
                          "points": int(len(full)),
                          "point_multiplier": round(len(full) / n_base, 3),
                          "per_class": {CLASS_NAMES[i]: round(float(iou[i]), 4)
                                        for i in range(NUM_CLASSES)},
                          "seconds": round(time.time() - t0, 1)}
        print(f"[{a.tag}][held {held.key}] mIoU {c.miou:.4f} "
              f"(base {BASE[held.key]:.4f}, {c.miou-BASE[held.key]:+.4f}) "
              f"ceiling {c.miou_ceiling:.4f}  "
              + "  ".join(f"{CLASS_NAMES[i]} {iou[i]:.3f}"
                          for i in range(NUM_CLASSES)), flush=True)

    loso = float(np.mean([f["miou"] for f in fold.values()]))
    both = all(fold[k]["miou"] > BASE[k] for k in BASE)
    point_multiplier = float(np.mean([f["point_multiplier"] for f in fold.values()]))
    row = {"tag": a.tag, "radius": a.radius, "base_voxel": a.base_voxel,
           "vertical_thresh": a.vertical_thresh, "lr": a.lr,
           "loso_miou": round(loso, 4), "both_folds": both,
           "vs_incumbent": round(loso - INCUMBENT, 4),
           "mean_point_multiplier": round(point_multiplier, 3),
           "score_per_point_multiplier": round(loso / point_multiplier, 4),
           "verdict": "ADOPT" if both and loso > INCUMBENT else "REJECT",
           "folds": fold}
    print(f"\n[{a.tag}] LOSO {loso:.4f} vs incumbent {INCUMBENT:.4f} "
          f"({loso-INCUMBENT:+.4f}) both folds: {both} -> {row['verdict']}",
          flush=True)
    (OUT / f"{a.tag}.json").write_text(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
