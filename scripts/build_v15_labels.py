"""Build v15 labels for both slab tiers in one pass.

Segmenters and the fused tree are shared; only the propagation target differs
between tiers, so votes and predictions are computed once per scene.
"""
import time
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree
from twinworld.dataio import NUM_CLASSES, CLASS_NAMES, Scene
from twinworld.package import submission_cloud
from twinworld.scoring import read_ply_xyz
from twinworld.semantics.classifier import (ClassifierConfig, fit, predict,
                                            thinned_features, transfer_labels)
from seg2d_lift import train_segmenter, predict_views
from seg2d_fusion import vote_features

V1 = Path("experiments/2026-08-08-pipeline-v1")
SLAB = Path("experiments/2026-08-27-slab-clouds")
LB = ("scene_009", "scene_010")
ALLGC = ("scene_009", "scene_010", "scene_011", "scene_012")
dev = torch.device("cuda:2")
cfg = ClassifierConfig(max_feature_points=4_000_000, learning_rate=0.05)

models = []
for sid in LB:
    s = Scene("gold_coast", sid)
    m, mu, sd = train_segmenter(s, dev, 1250, 512, 6e-5, "nvidia/mit-b0")
    models.append((s, m, mu, sd))
print("segmenters ready", flush=True)

Xs, ys = [], []
for sid in LB:
    s = Scene("gold_coast", sid)
    ply = V1/s.dataset/sid/"mvs"/"fused_gs_i.ply"
    thin, X, _, _ = thinned_features(s, ply, ClassifierConfig(max_feature_points=4_000_000))
    gt, gl = read_ply_xyz(s.gt_ply, with_classification=True)
    y = transfer_labels(gt, gl, thin, cfg.label_transfer_radius)
    other = [mm for mm in models if mm[0].scene_id != sid][0]
    vf = vote_features(s, thin, predict_views(other[1], s, dev, other[2], other[3]))
    Xs.append(np.hstack([X, vf])); ys.append(y)
    print(f"[{sid}] train features ready", flush=True)
tree = fit(np.vstack(Xs), np.concatenate(ys), cfg)
print("tree fitted", flush=True)

for sid in ALLGC:
    s = Scene("gold_coast", sid)
    t0 = time.time()
    ply = V1/s.dataset/sid/"mvs"/"fused_gs_i.ply"
    thin, X, _, _ = thinned_features(s, ply, ClassifierConfig(max_feature_points=4_000_000))
    vfs = [vote_features(s, thin, predict_views(m, s, dev, mu, sd)) for _, m, mu, sd in models]
    vf = np.mean(vfs, axis=0).astype(np.float32)
    pred_thin = predict(tree, np.hstack([X, vf]), n_jobs=1)
    kt = cKDTree(thin)
    for tier in ("safe", "push"):
        full = np.load(SLAB/f"{sid}_{tier}.npz")["xyz"].astype(np.float64)
        _, nn = kt.query(full, k=5, workers=-1)
        lab = pred_thin[nn]
        votes = np.zeros((len(full), NUM_CLASSES), np.float32)
        for c in range(NUM_CLASSES): votes[:, c] = (lab == c).sum(1)
        labels = np.argmax(votes, 1).astype(np.uint8)
        np.savez_compressed(SLAB/f"{sid}_{tier}_labeled.npz",
                            xyz=full.astype(np.float32), labels=labels)
        print(f"[{sid}][{tier}] {len(full)/1e6:.1f}M labelled ({time.time()-t0:.0f}s)", flush=True)
