"""A learned point segmenter, as a baseline against LightGBM on hand features.

Two things this is for. First, score: the system is now classification-bound
(ceiling ~0.44, achieved 0.180), so the model class is finally worth
interrogating. Second, the paper: a reviewer will ask why a gradient-boosted
tree over 29 hand-designed features is the segmenter in 2026, and "we tried a
learned one" is only an answer if it is true.

Architecture is EdgeConv (DGCNN), written directly in PyTorch because
MinkowskiEngine and torch_geometric are not installed and building them is a
poor use of the window. A fixed k-nearest-neighbour graph is precomputed once
on CPU (the point set is static), and each layer applies
max_j MLP([x_i, x_j - x_i]) over the neighbourhood, which is permutation
invariant and sees local geometry the tree cannot.

Deliberately one-hop with sampled centres: a two-hop formulation over 1.5 M
points at K=16 and width 128 allocates ~25 GB in the second layer, so depth is
spent on the edge MLP rather than on graph hops, and training samples centres
per step instead of running full batch.

Inputs are the same 29 features the tree sees, so the comparison isolates the
model class rather than the representation.

Honest expectation: two labelled scenes and ~70k labelled points is a small-data
regime where a boosted tree is genuinely strong and a deep model is prone to
overfitting. A negative result here is a real result and belongs in the paper.

Adoption gate, fixed before running: BOTH folds must beat their own LightGBM
baseline at v8 packaging (fold 009 > 0.1529, fold 010 > 0.2067, LOSO > 0.1798).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import cKDTree

from twinworld.dataio import CLASS_NAMES, NUM_CLASSES, Scene
from twinworld.package import submission_cloud
from twinworld.scoring import read_ply_xyz, score_semantics
from twinworld.semantics.classifier import (
    ClassifierConfig, propagate, thinned_features, transfer_labels,
)

V1 = Path("experiments/2026-08-08-pipeline-v1")
OUT = Path("experiments/2026-08-16-learned-segmenter")
PACK_VOXEL = 0.10
BASE = {"gold_coast/scene_009": 0.1529, "gold_coast/scene_010": 0.2067}
BASE_LOSO = 0.1798
K = 16


class Net(nn.Module):
    """One-hop EdgeConv with depth in the edge MLP, then a per-point head."""

    def __init__(self, c_in: int, width: int = 128):
        super().__init__()
        self.edge1 = nn.Sequential(
            nn.Linear(2 * c_in, width), nn.GELU(),
            nn.Linear(width, width), nn.GELU())
        self.edge2 = nn.Sequential(
            nn.Linear(width, width), nn.GELU(),
            nn.Linear(width, width), nn.GELU())
        self.head = nn.Sequential(
            nn.Linear(c_in + 2 * width, width), nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(width, width // 2), nn.GELU(),
            nn.Linear(width // 2, NUM_CLASSES))

    def forward(self, x_c, x_n):
        # x_c: (B, C) centres; x_n: (B, K, C) their neighbours
        centre = x_c.unsqueeze(1).expand_as(x_n)
        e = self.edge1(torch.cat([centre, x_n - centre], dim=-1))   # (B, K, W)
        p1 = e.max(dim=1).values
        p2 = self.edge2(e).max(dim=1).values
        return self.head(torch.cat([x_c, p1, p2], dim=-1))


def knn_index(xyz: np.ndarray, k: int) -> np.ndarray:
    _, idx = cKDTree(xyz).query(xyz, k=k + 1, workers=-1)
    return idx[:, 1:].astype(np.int64)          # drop self


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=2)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--feature-voxel", type=float, default=0.15)
    ap.add_argument("--max-feature-points", type=int, default=1_500_000)
    ap.add_argument("--tag", default="edgeconv")
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device(f"cuda:{a.gpu}")
    cfg = ClassifierConfig(feature_voxel=a.feature_voxel,
                           max_feature_points=a.max_feature_points)
    scenes = [Scene("gold_coast", s) for s in ("scene_009", "scene_010")]
    ply = {s.key: V1 / s.dataset / s.scene_id / "mvs" / "fused_gs_i.ply"
           for s in scenes}

    cache = {}
    for s in scenes:
        t0 = time.time()
        thin, X, _, _ = thinned_features(s, ply[s.key], cfg)
        gt, gl = read_ply_xyz(s.gt_ply, with_classification=True)
        y = transfer_labels(gt, gl, thin, cfg.label_transfer_radius)
        nn_idx = knn_index(thin, K)
        cache[s.key] = {"thin": thin, "X": X.astype(np.float32), "y": y,
                        "gt": gt, "gl": gl, "idx": nn_idx}
        print(f"[{s.key}] {len(thin)} pts, {int((y < NUM_CLASSES).sum())} labelled, "
              f"kNN built ({time.time()-t0:.0f}s)", flush=True)

    fold = {}
    for held in scenes:
        t0 = time.time()
        tr = [s for s in scenes if s.key != held.key][0]
        d_tr, d_h = cache[tr.key], cache[held.key]

        # normalisation comes from the TRAINING scene only
        mu = d_tr["X"].mean(0)
        sd = d_tr["X"].std(0) + 1e-6

        Xtr = torch.from_numpy((d_tr["X"] - mu) / sd).to(dev)
        itr = torch.from_numpy(d_tr["idx"]).to(dev)
        ytr = torch.from_numpy(d_tr["y"].astype(np.int64)).to(dev)
        lab_idx = torch.from_numpy(
            np.flatnonzero(d_tr["y"] < NUM_CLASSES)).to(dev)

        counts = np.bincount(d_tr["y"][d_tr["y"] < NUM_CLASSES],
                             minlength=NUM_CLASSES).astype(np.float64)
        cw = (counts.sum() / np.maximum(counts, 1)) ** 0.5
        cw = torch.from_numpy((cw / cw.mean()).astype(np.float32)).to(dev)
        print(f"  [{held.key}] train on {tr.key}; labelled {len(lab_idx)}; "
              f"class weights {np.round(cw.cpu().numpy(), 2).tolist()}", flush=True)

        torch.manual_seed(0)
        net = Net(Xtr.shape[1]).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.steps)
        net.train()
        for step in range(a.steps):
            sel = lab_idx[torch.randint(len(lab_idx), (a.batch,), device=dev)]
            xc = Xtr[sel]
            xn = Xtr[itr[sel]]
            loss = F.cross_entropy(net(xc, xn), ytr[sel], weight=cw,
                                   label_smoothing=0.05)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            if step % 500 == 0 or step == a.steps - 1:
                print(f"    step {step:5d}  loss {loss.item():.4f}", flush=True)

        # inference over the held-out scene, chunked
        Xh = torch.from_numpy((d_h["X"] - mu) / sd).to(dev)
        ih = torch.from_numpy(d_h["idx"]).to(dev)
        net.eval()
        preds = np.empty(len(Xh), np.uint8)
        CH = 200_000
        with torch.no_grad():
            for beg in range(0, len(Xh), CH):
                end = min(beg + CH, len(Xh))
                out = net(Xh[beg:end], Xh[ih[beg:end]])
                preds[beg:end] = out.argmax(1).cpu().numpy().astype(np.uint8)

        full, _ = submission_cloud(held, ply[held.key], voxel=PACK_VOXEL)
        pred = propagate(d_h["thin"], preds, full)
        c = score_semantics(full, pred, d_h["gt"], d_h["gl"])
        iou = c.per_class_iou
        fold[held.key] = {"miou": round(float(c.miou), 4),
                          "per_class": {CLASS_NAMES[i]: round(float(iou[i]), 4)
                                        for i in range(NUM_CLASSES)},
                          "seconds": round(time.time() - t0, 1)}
        print(f"  [held {held.key}] mIoU {c.miou:.4f} "
              f"(LightGBM {BASE[held.key]:.4f}, {c.miou-BASE[held.key]:+.4f})  "
              + "  ".join(f"{CLASS_NAMES[i]} {iou[i]:.3f}"
                          for i in range(NUM_CLASSES))
              + f"  ({fold[held.key]['seconds']}s)", flush=True)
        del Xtr, Xh, itr, ih
        torch.cuda.empty_cache()

    loso = float(np.mean([f["miou"] for f in fold.values()]))
    both = all(fold[k]["miou"] > BASE[k] for k in BASE)
    row = {"tag": a.tag, "arch": "EdgeConv/DGCNN one-hop", "k": K,
           "steps": a.steps, "batch": a.batch, "lr": a.lr,
           "feature_voxel": a.feature_voxel,
           "loso_miou": round(loso, 4), "both_folds": both,
           "verdict": "ADOPT" if both and loso > BASE_LOSO else "REJECT",
           "folds": fold}
    print(f"\n[{a.tag}] LOSO {loso:.4f} vs LightGBM {BASE_LOSO:.4f} "
          f"({loso-BASE_LOSO:+.4f}) both folds: {both} -> {row['verdict']}",
          flush=True)
    (OUT / f"{a.tag}.json").write_text(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
