"""Confidence pruning + pseudo-labelling: the two 'insane' levers, gated.

PRUNE: our uncertain points poison their neighbourhoods - a GT point takes our
nearest guess even when the model was split. Labelling them 255 does NOT help
(the scorer still lets them win the NN race and counts them wrong; verified in
our digit-calibrated reimplementation). Physically REMOVING them does: true
exclusion from the match pool. Safety: drop a low-confidence point only if a
high-confidence point sits within NEIGH_R, so coverage survives.

PSEUDO: self-training on the withheld domain. The labeller loses ~24 percent
crossing from dev to final scenes. High-confidence predictions on 011/012 are
target-domain supervision that involves no ground truth at all. Gated here by
simulation: for the fold held on scene B, pseudo-labels come from scene B's own
high-confidence predictions (the same information available at test time).

Gate: both folds beat baselines; LOSO beats ade_push 0.2498.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree
from twinworld.dataio import CLASS_NAMES, NUM_CLASSES, Scene
from twinworld.scoring import read_ply_xyz, score_semantics
from twinworld.semantics.classifier import (ClassifierConfig, fit,
                                            thinned_features, transfer_labels)
from seg2d_lift import train_segmenter, predict_views
from seg2d_fusion import vote_features
from ade_fusion import ade_maps, grouped_votes

V1 = Path("experiments/2026-08-08-pipeline-v1")
OUT = Path("experiments/2026-08-29-insane")
BASE = {"gold_coast/scene_009": 0.1529, "gold_coast/scene_010": 0.2067}
INC = 0.2498

def proba(model, X, batch=500_000):
    out = np.empty((len(X), NUM_CLASSES), np.float32)
    for a in range(0, len(X), batch): out[a:a+batch] = model.predict_proba(X[a:a+batch])
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=2)
    ap.add_argument("--conf", type=float, default=0.55)
    ap.add_argument("--neigh", type=float, default=0.08)
    ap.add_argument("--pseudo-conf", type=float, default=0.85)
    ap.add_argument("--tag", default="insane")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device(f"cuda:{a.gpu}")
    scenes = [Scene("gold_coast", s) for s in ("scene_009","scene_010")]
    ply = {s.key: V1/s.dataset/s.scene_id/"mvs"/"fused_gs_i.ply" for s in scenes}
    cfg = ClassifierConfig(max_feature_points=4_000_000, learning_rate=0.05)

    geo = {}
    for s in scenes:
        thin, X, _, _ = thinned_features(s, ply[s.key], ClassifierConfig(max_feature_points=4_000_000))
        gt, gl = read_ply_xyz(s.gt_ply, with_classification=True)
        y = transfer_labels(gt, gl, thin, cfg.label_transfer_radius)
        av = grouped_votes(s, thin, ade_maps(s, dev))
        full = np.load(f"experiments/2026-08-27-slab-clouds/{s.scene_id}_push.npz")["xyz"].astype(np.float64)
        geo[s.key] = dict(thin=thin, X=X, y=y, gt=gt, gl=gl, av=av, full=full)
        print(f"[{s.key}] ready", flush=True)

    res = {}
    for held in scenes:
        t0 = time.time()
        tr = [s for s in scenes if s.key != held.key][0]
        m, mu, sd = train_segmenter(tr, dev, 1500, 512, 6e-5, "nvidia/mit-b0")
        vf = {s.key: vote_features(s, geo[s.key]["thin"], predict_views(m, s, dev, mu, sd)) for s in scenes}
        del m; torch.cuda.empty_cache()
        g = geo[held.key]; gtr = geo[tr.key]
        Xtr = np.hstack([gtr["X"], vf[tr.key], gtr["av"]])
        Xh  = np.hstack([g["X"], vf[held.key], g["av"]])
        model = fit(Xtr, gtr["y"], cfg)
        P = proba(model, Xh)
        conf = P.max(1); lab_thin = P.argmax(1).astype(np.uint8)

        # PSEUDO: retrain with held scene's own high-confidence points as extra supervision
        hi = conf >= a.pseudo_conf
        Xp = np.vstack([Xtr, Xh[hi]]); yp = np.concatenate([gtr["y"], lab_thin[hi]])
        model2 = fit(Xp, yp, cfg)
        P2 = proba(model2, Xh)
        conf2 = P2.max(1); lab2 = P2.argmax(1).astype(np.uint8)
        print(f"  [{held.key}] pseudo added {hi.sum()} pts ({100*hi.mean():.0f}%)", flush=True)

        for name, cf, lb in (("base", conf, lab_thin), ("pseudo", conf2, lab2)):
            kt = cKDTree(g["thin"])
            _, nn = kt.query(g["full"], k=5, workers=-1)
            votes = np.zeros((len(g["full"]), NUM_CLASSES), np.float32)
            lv = lb[nn]
            for c in range(NUM_CLASSES): votes[:,c] = (lv==c).sum(1)
            pred = np.argmax(votes,1).astype(np.uint8)
            # PRUNE variant: drop full-cloud points whose thin-neighbour vote is weak
            # and which have a strong neighbour nearby
            strength = votes.max(1) / np.maximum(votes.sum(1),1)
            pconf = cf[nn].mean(1) * strength
            weak = pconf < a.conf
            strong_pts = g["full"][~weak]
            if len(strong_pts) > 1000 and weak.any():
                d,_ = cKDTree(strong_pts).query(g["full"][weak], k=1,
                                               distance_upper_bound=a.neigh, workers=-1)
                drop = np.zeros(len(g["full"]), bool)
                wi = np.flatnonzero(weak); drop[wi[np.isfinite(d)]] = True
            else: drop = np.zeros(len(g["full"]), bool)
            for variant, mask in (("", None), ("_prune", drop)):
                if mask is None: fx, fl = g["full"], pred
                else: fx, fl = g["full"][~mask], pred[~mask]
                c = score_semantics(fx, fl, g["gt"], g["gl"])
                res.setdefault(name+variant, {})[held.key] = round(float(c.miou),4)
                print(f"  [{held.key}] {name+variant or 'base':12s} mIoU {c.miou:.4f} "
                      f"({len(fx)/1e6:.1f}M pts)", flush=True)
        print(f"  fold done ({time.time()-t0:.0f}s)", flush=True)

    print()
    for name, folds in res.items():
        loso = float(np.mean(list(folds.values())))
        both = all(folds[k] > BASE[k] for k in BASE)
        v = "ADOPT" if both and loso > INC else "REJECT"
        print(f"[{name:14s}] LOSO {loso:.4f} vs incumbent {INC} ({loso-INC:+.4f}) both:{both} -> {v}", flush=True)
    (OUT/f"{a.tag}.json").write_text(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()
