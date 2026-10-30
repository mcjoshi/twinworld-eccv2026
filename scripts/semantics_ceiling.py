"""How much of the semantic score is reachable with perfect geometry?

Substitutes the ground-truth cloud (voxel-downsampled to the submission
resolution) for our reconstruction, so coverage is near-perfect and the only
error left is classification. The resulting mIoU is the ceiling the real
pipeline is working towards, and the gap between this and the MVS-based number
tells us whether to invest in geometry or in the classifier.

Runs on CPU only, so it can be used while the GPUs are busy with MVS.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from twinworld.dataio import CLASS_NAMES, NUM_CLASSES, Scene
from twinworld.scoring import read_ply_xyz, score_semantics, voxel_downsample
from twinworld.semantics.classifier import ClassifierConfig, fit, predict
from twinworld.semantics.features import compute_features, estimate_up


def load_scene_cloud(scene: Scene, voxel: float):
    from plyfile import PlyData

    v = PlyData.read(str(scene.gt_ply))["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    rgb = np.stack([v["red"], v["green"], v["blue"]], 1).astype(np.uint8)
    lab = np.asarray(v["classification"]).astype(np.uint8)
    # carry colour and label through the downsample by keeping the chosen indices
    keys = np.floor(xyz / voxel).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    first.sort()
    return xyz[first], rgb[first], lab[first]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--voxel", type=float, default=0.05)
    ap.add_argument("--out", default="experiments/2026-08-08-semantics-ceiling")
    ap.add_argument("--class-weighting", default="balanced")
    ap.add_argument("--n-estimators", type=int, default=300)
    ap.add_argument("--max-train-points", type=int, default=800_000)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = ClassifierConfig(class_weighting=args.class_weighting,
                           n_estimators=args.n_estimators,
                           max_train_points=args.max_train_points)
    scenes = [Scene("gold_coast", s) for s in ("scene_009", "scene_010")]

    cache = {}
    for s in scenes:
        t0 = time.time()
        xyz, rgb, lab = load_scene_cloud(s, args.voxel)
        rig = np.stack([fr.center for fr in s.train_frames]).mean(axis=0)
        X, names = compute_features(xyz, rgb, estimate_up(s.train_frames), rig=rig)
        cache[s.key] = (xyz, X, lab)
        hist = np.bincount(lab, minlength=256)
        print(
            f"[{s.key}] {len(xyz):,} pts at {args.voxel} m, {X.shape[1]} features, "
            f"class counts {hist[:5].tolist()} ignore={int(hist[255])} "
            f"({time.time()-t0:.0f}s)",
            flush=True,
        )

    folds = []
    for held in scenes:
        t0 = time.time()
        tr = [s for s in scenes if s.key != held.key]
        Xtr = np.concatenate([cache[s.key][1] for s in tr])
        ytr = np.concatenate([cache[s.key][2] for s in tr])
        model = fit(Xtr, ytr, cfg)

        xyz, X, _ = cache[held.key]
        pred = predict(model, X)
        gt_xyz, gt_lab = read_ply_xyz(held.gt_ply, with_classification=True)
        c = score_semantics(xyz, pred, gt_xyz, gt_lab)
        iou = c.per_class_iou
        folds.append(
            {
                "held_out": held.key,
                "miou": c.miou,
                "per_class_iou": {CLASS_NAMES[i]: (None if np.isnan(iou[i]) else round(float(iou[i]), 4))
                                  for i in range(NUM_CLASSES)},
                "unmatched_frac": round(c.n_unmatched / max(c.n_total, 1), 5),
                "seconds": round(time.time() - t0, 1),
            }
        )
        print(
            f"[held out {held.key}] mIoU {c.miou:.4f}  "
            + "  ".join(f"{CLASS_NAMES[i]} {iou[i]:.3f}" for i in range(NUM_CLASSES))
            + f"  unmatched {folds[-1]['unmatched_frac']:.4f}  ({time.time()-t0:.0f}s)",
            flush=True,
        )

    mious = [f["miou"] for f in folds]
    res = {
        "what": "semantic ceiling on GT geometry, leave-one-scene-out",
        "voxel": args.voxel,
        "config": asdict(cfg),
        "folds": folds,
        "miou_mean": float(np.mean(mious)),
        "miou_spread": float(np.max(mious) - np.min(mious)),
    }
    (out / f"ceiling_{args.class_weighting}_v{args.voxel}.json").write_text(json.dumps(res, indent=2))
    print(f"\nCEILING mIoU = {res['miou_mean']:.4f}  (fold spread {res['miou_spread']:.4f})")


if __name__ == "__main__":
    main()
