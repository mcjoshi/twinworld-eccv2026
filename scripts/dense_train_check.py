"""Does label-dense training sampling beat uniform sampling?

Same features, same model, same inference path. The only change is which points
contribute training gradient: uniform sampling yields ~26 000 labelled points
per scene, label-dense sampling yields up to 1.5 M drawn from the region within
the label-transfer radius of ground truth.

Adoption rule, fixed before the run: held-out LOSO must beat 0.1717 with both
folds improving (baselines 0.1449 / 0.1985 on the same v6 clouds).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from twinworld.dataio import CLASS_NAMES, NUM_CLASSES, Scene
from twinworld.scoring import read_ply_xyz, score_semantics
from twinworld.semantics.classifier import (
    ClassifierConfig, fit, predict, propagate, thinned_features, training_features,
)


def main() -> None:
    cfg = ClassifierConfig()
    run = Path("experiments/2026-08-10-pipeline-v6")
    out = Path("experiments/2026-08-11-dense-train")
    out.mkdir(parents=True, exist_ok=True)
    scenes = [Scene("gold_coast", sid) for sid in ("scene_009", "scene_010")]
    cloud = {s.key: Path("experiments/2026-08-08-pipeline-v1") / s.dataset / s.scene_id
             / "mvs" / "fused_gs_i.ply" for s in scenes}

    train_sets, infer_sets = {}, {}
    for s in scenes:
        t0 = time.time()
        X, y = training_features(s, cloud[s.key], cfg)
        n_lab = int((y < NUM_CLASSES).sum())
        train_sets[s.key] = (X, y)
        print(f"[{s.key}] training set: {len(X)} pts, {n_lab} labelled "
              f"(uniform baseline had ~26k) ({time.time()-t0:.0f}s)", flush=True)
        thin, Xi, _, full = thinned_features(s, cloud[s.key], cfg)
        infer_sets[s.key] = (thin, Xi, full)

    results = {}
    for held in scenes:
        t0 = time.time()
        tr = [s for s in scenes if s.key != held.key][0]
        model = fit(*train_sets[tr.key], cfg)
        thin, Xi, full = infer_sets[held.key]
        pred = propagate(thin, predict(model, Xi, n_jobs=cfg.n_jobs), full)
        gt, gl = read_ply_xyz(held.gt_ply, with_classification=True)
        c = score_semantics(full, pred, gt, gl)
        iou = c.per_class_iou
        results[held.key] = {"miou": c.miou,
                             "per_class": {CLASS_NAMES[i]: round(float(iou[i]), 4)
                                           for i in range(NUM_CLASSES)}}
        print(f"[held out {held.key}] dense-train mIoU {c.miou:.4f}   "
              + "  ".join(f"{CLASS_NAMES[i]} {iou[i]:.3f}" for i in range(NUM_CLASSES))
              + f"   ({time.time()-t0:.0f}s)", flush=True)

    (out / "results.json").write_text(json.dumps(results, indent=2))
    mean = float(np.mean([r["miou"] for r in results.values()]))
    baselines = {"gold_coast/scene_009": 0.1449, "gold_coast/scene_010": 0.1985}
    both = all(results[k]["miou"] > v for k, v in baselines.items())
    print(f"\ndense-train LOSO mean {mean:.4f} (uniform baseline 0.1717, "
          f"folds {baselines})")
    print("both folds improve:", both, "-> ADOPT" if both and mean > 0.1717 else "-> keep uniform")


if __name__ == "__main__":
    main()
