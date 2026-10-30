"""Do today's two adoptions compose?

Two changes passed the both-folds gate on 19 Aug, and they act at different
stages, so they should stack rather than overlap:

  lr-0.05   training-side: learning rate 0.05 instead of 0.10 at the same 200
            trees. LOSO 0.1798 -> 0.1834. The capacity sweep showed *more*
            capacity consistently hurts (trees-600 -0.0045, trees-1500 -0.0066,
            leaves-127 -0.0058) while a slower rate at fixed capacity helps,
            which is ordinary small-data regularisation.
  k=5       inference-side: majority vote over the 5 nearest classified points
            instead of the single nearest. LOSO 0.1798 -> 0.1814.

Everything else is the incumbent v8 configuration: the stereo+interpolated cloud
`fused_gs_i.ply`, 4 M feature budget, 15 cm feature voxel, 10 cm packaging. The
denser-cloud variants were all rejected today and are not used here.

Adoption gate, fixed before running: BOTH folds must beat their own baseline
(009 > 0.1529, 010 > 0.2067) and LOSO must beat the better single change, 0.1834.
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from twinworld.dataio import CLASS_NAMES, NUM_CLASSES, Scene
from twinworld.package import submission_cloud
from twinworld.scoring import read_ply_xyz, score_semantics
from twinworld.semantics.classifier import (ClassifierConfig, fit, predict,
                                            thinned_features, transfer_labels)

V1 = Path("experiments/2026-08-08-pipeline-v1")
OUT = Path("experiments/2026-08-19-combined")
PACK_VOXEL = 0.10
BASE = {"gold_coast/scene_009": 0.1529, "gold_coast/scene_010": 0.2067}
BEST_SINGLE = 0.1834


def propagate_k(thin_xyz, thin_labels, full_xyz, k):
    if k == 1:
        _, idx = cKDTree(thin_xyz).query(full_xyz, workers=-1)
        return thin_labels[idx]
    _, idx = cKDTree(thin_xyz).query(full_xyz, k=k, workers=-1)
    lab = thin_labels[idx]
    votes = np.zeros((len(full_xyz), NUM_CLASSES), np.float32)
    for c in range(NUM_CLASSES):
        votes[:, c] = (lab == c).sum(axis=1)
    return np.argmax(votes, axis=1).astype(np.uint8)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    scenes = [Scene("gold_coast", s) for s in ("scene_009", "scene_010")]
    ply = {s.key: V1 / s.dataset / s.scene_id / "mvs" / "fused_gs_i.ply" for s in scenes}
    feat_cfg = ClassifierConfig(max_feature_points=4_000_000)

    cache = {}
    for s in scenes:
        thin, X, _, _ = thinned_features(s, ply[s.key], feat_cfg)
        gt, gl = read_ply_xyz(s.gt_ply, with_classification=True)
        y = transfer_labels(gt, gl, thin, feat_cfg.label_transfer_radius)
        full, _ = submission_cloud(s, ply[s.key], voxel=PACK_VOXEL)
        cache[s.key] = (thin, X, y, gt, gl, full)
        print(f"[{s.key}] ready: {len(thin)} feature pts, {len(full)/1e6:.1f}M packaged", flush=True)

    rows = []
    for tag, lr, k in (("lr010_k1_baseline", 0.10, 1), ("lr005_k1", 0.05, 1),
                       ("lr010_k5", 0.10, 5), ("lr005_k5", 0.05, 5),
                       ("lr005_k3", 0.05, 3)):
        cfg = ClassifierConfig(max_feature_points=4_000_000, learning_rate=lr)
        models = {s.key: fit(cache[s.key][1], cache[s.key][2], cfg) for s in scenes}
        fold = {}
        for held in scenes:
            t0 = time.time()
            tr = [s for s in scenes if s.key != held.key][0]
            thin, X, _, gt, gl, full = cache[held.key]
            pred = propagate_k(thin, predict(models[tr.key], X, n_jobs=cfg.n_jobs), full, k)
            c = score_semantics(full, pred, gt, gl)
            iou = c.per_class_iou
            fold[held.key] = {"miou": round(float(c.miou), 4),
                              "per_class": {CLASS_NAMES[i]: round(float(iou[i]), 4)
                                            for i in range(NUM_CLASSES)}}
            print(f"  [{tag} held {held.key}] mIoU {c.miou:.4f} "
                  f"({c.miou-BASE[held.key]:+.4f})  "
                  + "  ".join(f"{CLASS_NAMES[i]} {iou[i]:.3f}" for i in range(NUM_CLASSES))
                  + f"  ({time.time()-t0:.0f}s)", flush=True)
        loso = float(np.mean([f["miou"] for f in fold.values()]))
        both = all(fold[kk]["miou"] > BASE[kk] for kk in BASE)
        rows.append({"tag": tag, "lr": lr, "k": k, "loso_miou": round(loso, 4),
                     "both_folds": both,
                     "verdict": "ADOPT" if both and loso > BEST_SINGLE else "REJECT",
                     "folds": fold})
        print(f"[{tag}] LOSO {loso:.4f} (vs best single {BEST_SINGLE:.4f}, "
              f"{loso-BEST_SINGLE:+.4f}) both folds: {both} -> {rows[-1]['verdict']}\n", flush=True)
        (OUT / "results.json").write_text(json.dumps(rows, indent=2))

    best = max(rows, key=lambda r: r["loso_miou"])
    print(f"best: {best['tag']} LOSO {best['loso_miou']:.4f} -> {best['verdict']}")


if __name__ == "__main__":
    main()
