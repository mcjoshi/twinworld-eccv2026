"""LightGBM capacity sweep for Gold Coast semantic LOSO evaluation.

Everything except the requested LightGBM capacity settings is fixed: features
use a 15 cm voxel with a four-million-point budget, predictions are packaged at
10 cm, and each scene is evaluated only with a model fitted on the other scene.

Pre-registered adoption gate: baseline fold scores are gold_coast/scene_009 =
0.1529 and gold_coast/scene_010 = 0.2067, with LOSO mean = 0.1798. A config is
ADOPTED only if BOTH folds beat their own baseline AND the LOSO mean > 0.1798.
Improving the mean alone is a REJECT.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from twinworld.dataio import CLASS_NAMES, NUM_CLASSES, Scene
from twinworld.package import submission_cloud
from twinworld.scoring import read_ply_xyz, score_semantics
from twinworld.semantics.classifier import (
    ClassifierConfig, fit, predict, propagate, thinned_features, transfer_labels,
)

V1 = Path("experiments/2026-08-08-pipeline-v1")
OUT = Path("experiments/2026-08-19-capacity")
PACK_VOXEL = 0.10
BASE = {"gold_coast/scene_009": 0.1529, "gold_coast/scene_010": 0.2067}
BASE_LOSO = 0.1798

CONFIGS = (
    {"tag": "baseline", "n_estimators": 200, "num_leaves": 31,
     "min_child_samples": 60, "learning_rate": 0.10},
    {"tag": "trees-600", "n_estimators": 600, "num_leaves": 31,
     "min_child_samples": 60, "learning_rate": 0.10},
    {"tag": "trees-1500", "n_estimators": 1500, "num_leaves": 31,
     "min_child_samples": 60, "learning_rate": 0.10},
    {"tag": "leaves-127", "n_estimators": 200, "num_leaves": 127,
     "min_child_samples": 60, "learning_rate": 0.10},
    {"tag": "child-20", "n_estimators": 200, "num_leaves": 31,
     "min_child_samples": 20, "learning_rate": 0.10},
    {"tag": "lr-005", "n_estimators": 200, "num_leaves": 31,
     "min_child_samples": 60, "learning_rate": 0.05},
    {"tag": "medium-capacity", "n_estimators": 600, "num_leaves": 127,
     "min_child_samples": 20, "learning_rate": 0.10},
    {"tag": "high-capacity", "n_estimators": 1500, "num_leaves": 127,
     "min_child_samples": 20, "learning_rate": 0.05},
)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    scenes = [Scene("gold_coast", s) for s in ("scene_009", "scene_010")]
    ply = {s.key: V1 / s.dataset / s.scene_id / "mvs" / "fused_gs_i.ply"
           for s in scenes}
    base_cfg = ClassifierConfig(max_feature_points=4_000_000, n_jobs=1)

    cache = {}
    for s in scenes:
        t0 = time.time()
        thin, X, _, _ = thinned_features(s, ply[s.key], base_cfg)
        gt, gl = read_ply_xyz(s.gt_ply, with_classification=True)
        y = transfer_labels(gt, gl, thin, base_cfg.label_transfer_radius)
        full, _ = submission_cloud(s, ply[s.key], voxel=PACK_VOXEL)
        cache[s.key] = (thin, X, y, gt, gl, full)
        print(f"[{s.key}] {len(thin)} feature pts, {len(full)/1e6:.1f}M packaged "
              f"({time.time() - t0:.0f}s)", flush=True)

    results = []
    for settings in CONFIGS:
        cfg = ClassifierConfig(max_feature_points=4_000_000, n_jobs=1,
                               n_estimators=settings["n_estimators"],
                               num_leaves=settings["num_leaves"],
                               min_child_samples=settings["min_child_samples"],
                               learning_rate=settings["learning_rate"])
        models = {s.key: fit(cache[s.key][1], cache[s.key][2], cfg) for s in scenes}
        fold = {}
        for held in scenes:
            t0 = time.time()
            tr = [s for s in scenes if s.key != held.key][0]
            thin, X, _, gt, gl, full = cache[held.key]
            pred_thin = predict(models[tr.key], X, n_jobs=cfg.n_jobs)
            pred = propagate(thin, pred_thin, full)
            c = score_semantics(full, pred, gt, gl)
            iou = c.per_class_iou
            fold[held.key] = {
                "miou": round(float(c.miou), 4),
                "per_class": {CLASS_NAMES[i]: round(float(iou[i]), 4)
                              for i in range(NUM_CLASSES)},
                "seconds": round(time.time() - t0, 1),
            }
            print(f"  [{settings['tag']} held {held.key}] mIoU {c.miou:.4f} "
                  f"(base {BASE[held.key]:.4f}, {c.miou - BASE[held.key]:+.4f})  "
                  + "  ".join(f"{CLASS_NAMES[i]} {iou[i]:.3f}"
                              for i in range(NUM_CLASSES)), flush=True)
        loso = float(np.mean([f["miou"] for f in fold.values()]))
        both = all(fold[k]["miou"] > BASE[k] for k in BASE)
        results.append({
            **settings,
            "n_jobs": cfg.n_jobs,
            "loso_miou": round(loso, 4),
            "both_folds": both,
            "verdict": "ADOPT" if both and loso > BASE_LOSO else "REJECT",
            "folds": fold,
        })
        print(f"[{settings['tag']}] LOSO {loso:.4f} ({loso - BASE_LOSO:+.4f}) "
              f"both folds: {both} -> {results[-1]['verdict']}\n", flush=True)
        (OUT / "results.json").write_text(json.dumps(results, indent=2))

    print("\nconfig                 fold009  fold010  LOSO    verdict")
    for result in results:
        f9 = result["folds"]["gold_coast/scene_009"]["miou"]
        f10 = result["folds"]["gold_coast/scene_010"]["miou"]
        print(f"{result['tag']:22s} {f9:.4f}   {f10:.4f}   "
              f"{result['loso_miou']:.4f}  {result['verdict']}")


if __name__ == "__main__":
    main()
