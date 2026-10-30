"""How labels get from the classified thinning onto the submitted cloud.

`propagate` gives every submitted point the label of its single nearest
classified point. At 20 cm packaging that was harmless - the submitted points
were sparser than the classified ones, so the nearest classified point was
usually the only sensible answer. At 10 cm the submitted cloud is denser than
the 15 cm feature thinning, so one noisy classified point now stamps its label
onto several submitted points, and any classifier error is amplified rather
than averaged.

Variants tested, all on identical cached features so the only thing changing is
the transfer:

  k1        current behaviour, nearest classified point
  k3, k5    majority vote over the k nearest classified points
  k5_dist   inverse-distance weighted vote over 5 neighbours

Adoption gate, fixed before running: BOTH folds must beat their own baseline at
10 cm packaging (fold 009 > 0.1529, fold 010 > 0.2067, LOSO > 0.1798).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from twinworld.dataio import CLASS_NAMES, NUM_CLASSES, Scene
from twinworld.package import submission_cloud
from twinworld.scoring import read_ply_xyz, score_semantics
from twinworld.semantics.classifier import (
    ClassifierConfig, fit, predict, thinned_features, transfer_labels,
)

V1 = Path("experiments/2026-08-08-pipeline-v1")
OUT = Path("experiments/2026-08-14-propagation")
PACK_VOXEL = 0.10
BASE = {"gold_coast/scene_009": 0.1529, "gold_coast/scene_010": 0.2067}
BASE_LOSO = 0.1798


def propagate_k(thin_xyz, thin_labels, full_xyz, k: int, weighted: bool):
    if k == 1:
        _, idx = cKDTree(thin_xyz).query(full_xyz, workers=-1)
        return thin_labels[idx]
    d, idx = cKDTree(thin_xyz).query(full_xyz, k=k, workers=-1)
    lab = thin_labels[idx]                       # (N, k)
    w = 1.0 / np.maximum(d, 1e-6) if weighted else np.ones_like(d)
    votes = np.zeros((len(full_xyz), NUM_CLASSES), np.float32)
    for c in range(NUM_CLASSES):
        votes[:, c] = (w * (lab == c)).sum(axis=1)
    return np.argmax(votes, axis=1).astype(np.uint8)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = ClassifierConfig(max_feature_points=4_000_000)   # cached features
    scenes = [Scene("gold_coast", s) for s in ("scene_009", "scene_010")]
    ply = {s.key: V1 / s.dataset / s.scene_id / "mvs" / "fused_gs_i.ply"
           for s in scenes}

    cache, models = {}, {}
    for s in scenes:
        t0 = time.time()
        thin, X, _, _ = thinned_features(s, ply[s.key], cfg)
        gt, gl = read_ply_xyz(s.gt_ply, with_classification=True)
        y = transfer_labels(gt, gl, thin, cfg.label_transfer_radius)
        cache[s.key] = (thin, X, y, gt, gl)
        print(f"[{s.key}] {len(thin)} feature pts ({time.time()-t0:.0f}s)", flush=True)
    for s in scenes:
        models[s.key] = fit(cache[s.key][1], cache[s.key][2], cfg)
    print("models fitted", flush=True)

    # predictions and the packaged cloud are identical across variants
    pred_thin, full = {}, {}
    for held in scenes:
        tr = [s for s in scenes if s.key != held.key][0]
        pred_thin[held.key] = predict(models[tr.key], cache[held.key][1],
                                      n_jobs=cfg.n_jobs)
        full[held.key], _ = submission_cloud(held, ply[held.key], voxel=PACK_VOXEL)
        print(f"[{held.key}] packaged {len(full[held.key])/1e6:.1f}M pts", flush=True)

    rows = []
    for tag, k, wt in (("k1", 1, False), ("k3", 3, False),
                       ("k5", 5, False), ("k5_dist", 5, True)):
        fold = {}
        for held in scenes:
            t0 = time.time()
            thin, _, _, gt, gl = cache[held.key]
            lab = propagate_k(thin, pred_thin[held.key], full[held.key], k, wt)
            c = score_semantics(full[held.key], lab, gt, gl)
            iou = c.per_class_iou
            fold[held.key] = {"miou": round(float(c.miou), 4),
                              "per_class": {CLASS_NAMES[i]: round(float(iou[i]), 4)
                                            for i in range(NUM_CLASSES)},
                              "seconds": round(time.time() - t0, 1)}
            print(f"  [{tag} held {held.key}] mIoU {c.miou:.4f} "
                  f"(base {BASE[held.key]:.4f}, {c.miou-BASE[held.key]:+.4f})"
                  f"  ({fold[held.key]['seconds']}s)", flush=True)
        loso = float(np.mean([f["miou"] for f in fold.values()]))
        both = all(fold[kk]["miou"] > BASE[kk] for kk in BASE)
        rows.append({"tag": tag, "k": k, "weighted": wt,
                     "loso_miou": round(loso, 4), "both_folds": both,
                     "verdict": "ADOPT" if both and loso > BASE_LOSO else "REJECT",
                     "folds": fold})
        print(f"[{tag}] LOSO {loso:.4f} ({loso-BASE_LOSO:+.4f} vs base) "
              f"both folds: {both} -> {rows[-1]['verdict']}\n", flush=True)
        (OUT / "results.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
