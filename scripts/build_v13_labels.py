"""Build v13 Gold Coast labels: geometric features fused with 2D segmentation votes.

Gated on 25 Aug as fusion_mitb0: LOSO 0.2103 against v12's 0.1915, both folds
improving (0.1913 and 0.2294 against baselines of 0.1529 and 0.2067), at zero
extra points. The gain concentrates in the classes geometry cannot represent -
"other" +0.065 and window +0.037 - while wall, which geometry already handles,
is unchanged.

For the submission the segmenter trains on BOTH labelled scenes (009 and 010),
since 011 and 012 carry no labels and there is nothing to leak. The
leave-one-scene-out discipline that validated the method is preserved in the
experiment, not here: here we use all available supervision, which is the correct
thing to do once the method itself has been gated.

Point clouds are unchanged from v13's predecessor - the offset-sheet clouds built
for v12 - so the submission's size and runtime profile are identical and the
89.6 M envelope already reasoned about still applies.

Writes fused_gs_i_v13.npz per Gold Coast scene (xyz + labels).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from twinworld.dataio import CLASS_NAMES, NUM_CLASSES, Scene
from twinworld.package import submission_cloud
from twinworld.semantics.classifier import (
    ClassifierConfig, fit, predict, thinned_features, transfer_labels,
)
from twinworld.scoring import read_ply_xyz
from seg2d_lift import predict_views, train_segmenter
from seg2d_fusion import vote_features

V1 = Path("experiments/2026-08-08-pipeline-v1")
PACK = 0.10
LABELLED = ("scene_009", "scene_010")
ALL_GC = ("scene_009", "scene_010", "scene_011", "scene_012")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=3)
    ap.add_argument("--steps", type=int, default=2500)
    a = ap.parse_args()
    dev = torch.device(f"cuda:{a.gpu}")

    # one segmenter, trained on every labelled view we have
    print("training segmenter on both labelled scenes ...", flush=True)
    t0 = time.time()
    models = []
    for sid in LABELLED:
        s = Scene("gold_coast", sid)
        m, mean, std = train_segmenter(s, dev, a.steps // 2, 512, 6e-5, "nvidia/mit-b0")
        models.append((s, m, mean, std))
    print(f"  segmenters ready ({time.time()-t0:.0f}s)", flush=True)

    # training set for the tree: geometric features + votes, on labelled scenes
    cfg = ClassifierConfig(max_feature_points=4_000_000, learning_rate=0.05)
    Xs, ys = [], []
    for sid in LABELLED:
        s = Scene("gold_coast", sid)
        ply = V1 / s.dataset / s.scene_id / "mvs" / "fused_gs_i.ply"
        thin, X, _, _ = thinned_features(
            s, ply, ClassifierConfig(max_feature_points=4_000_000))
        gt, gl = read_ply_xyz(s.gt_ply, with_classification=True)
        y = transfer_labels(gt, gl, thin, cfg.label_transfer_radius)
        # votes from the segmenter trained on the OTHER labelled scene, so the
        # training features carry the same domain shift the test scenes will
        other = [(sc, m, mu, sd) for sc, m, mu, sd in models if sc.scene_id != sid][0]
        pm = predict_views(other[1], s, dev, other[2], other[3])
        vf = vote_features(s, thin, pm)
        Xs.append(np.hstack([X, vf]))
        ys.append(y)
        print(f"  [{s.key}] {len(thin)} train pts, "
              f"{100*vf[:,-1].mean():.1f}% seen", flush=True)

    model = fit(np.vstack(Xs), np.concatenate(ys), cfg)
    print("tree fitted on both labelled scenes", flush=True)

    # apply to every Gold Coast scene, including the withheld 011 and 012
    for sid in ALL_GC:
        s = Scene("gold_coast", sid)
        t1 = time.time()
        ply = V1 / s.dataset / s.scene_id / "mvs" / "fused_gs_i.ply"
        thin, X, _, _ = thinned_features(
            s, ply, ClassifierConfig(max_feature_points=4_000_000))
        # average the two segmenters' votes for the unlabelled scenes
        vfs = [vote_features(s, thin, predict_views(m, s, dev, mu, sd))
               for _, m, mu, sd in models]
        vf = np.mean(vfs, axis=0).astype(np.float32)
        pred_thin = predict(model, np.hstack([X, vf]), n_jobs=cfg.n_jobs)

        # v13 keeps v12's offset-sheet cloud; only the labels change
        v12 = V1 / s.dataset / s.scene_id / "mvs" / "fused_gs_i_v12.npz"
        full = np.load(v12)["xyz"].astype(np.float64)
        _, nn = cKDTree(thin).query(full, k=5, workers=-1)
        lab = pred_thin[nn]
        votes = np.zeros((len(full), NUM_CLASSES), np.float32)
        for c in range(NUM_CLASSES):
            votes[:, c] = (lab == c).sum(1)
        labels = np.argmax(votes, 1).astype(np.uint8)

        out = V1 / s.dataset / s.scene_id / "mvs" / "fused_gs_i_v13.npz"
        np.savez_compressed(out, xyz=full.astype(np.float32), labels=labels)
        share = {CLASS_NAMES[c]: round(float((labels == c).mean()), 3)
                 for c in range(NUM_CLASSES)}
        print(f"[{s.key}] {len(full)/1e6:.1f}M pts -> {out.name}  {share}  "
              f"({time.time()-t1:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
