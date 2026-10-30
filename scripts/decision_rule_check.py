"""Class weighting x decision rule: fixing the rare-class operating point.

The survey's leverage table says "other" is the largest single lever in the
competition (+0.108) and converts only 4 percent of its reachable coverage.
Diagnosis: with `balanced` class weights, a 3 percent wall-to-other error rate
creates as many false "other" points as the class has true points, because wall
is 56 percent of the scene and other is under 3. The classifier is not weak so
much as operated at an IoU-hostile point.

Two knobs, swept jointly on cached features:
  weighting  none | sqrt | balanced   (training-time)
  rule       argmax | per-class bias vector tuned for pooled mIoU (post-hoc)

The bias vector multiplies predicted probabilities before argmax and is tuned
by coordinate ascent ON THE TRAINING SCENE ONLY (against its own transferred
labels), then applied frozen to the held-out scene, so LOSO stays honest.

Gate: beat 0.1846 (uniform-4M baseline) with both folds improving.
"""

from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

import numpy as np

from twinworld.dataio import CLASS_NAMES, NUM_CLASSES, Scene
from twinworld.scoring import read_ply_xyz, score_semantics
from twinworld.semantics.classifier import (
    ClassifierConfig, fit, propagate, thinned_features, transfer_labels,
)


def predict_proba(model, X, batch=500_000, n_jobs=1):
    try:
        model.booster_.params["num_threads"] = n_jobs
        model.set_params(n_jobs=n_jobs)
    except Exception:  # noqa: BLE001
        pass
    out = np.empty((len(X), NUM_CLASSES), np.float32)
    for a in range(0, len(X), batch):
        out[a:a + batch] = model.predict_proba(X[a:a + batch])
    return out


def masked_miou(pred, y):
    """mIoU over labelled points only - the tuning objective on the train scene."""
    m = y < NUM_CLASSES
    p, t = pred[m], y[m]
    ious = []
    for c in range(NUM_CLASSES):
        inter = ((p == c) & (t == c)).sum()
        union = ((p == c) | (t == c)).sum()
        if union:
            ious.append(inter / union)
    return float(np.mean(ious)) if ious else 0.0


def tune_bias(proba, y, iters=3):
    """Coordinate ascent on per-class log-bias, maximising labelled-point mIoU."""
    bias = np.zeros(NUM_CLASSES, np.float32)
    grid = np.linspace(-2.5, 2.5, 21)
    best = masked_miou(np.argmax(proba, 1), y)
    for _ in range(iters):
        for c in range(NUM_CLASSES):
            vals = []
            for g in grid:
                b = bias.copy()
                b[c] = g
                pred = np.argmax(proba + b, 1)  # log-domain bias on log-ish scores
                vals.append((masked_miou(pred, y), g))
            v, g = max(vals)
            if v > best:
                best, bias[c] = v, g
    return bias, best


def main() -> None:
    out = Path("experiments/2026-08-11-decision-rule")
    out.mkdir(parents=True, exist_ok=True)
    scenes = [Scene("gold_coast", sid) for sid in ("scene_009", "scene_010")]
    cloud = {s.key: Path("experiments/2026-08-08-pipeline-v1") / s.dataset / s.scene_id
             / "mvs" / "fused_gs_i.ply" for s in scenes}

    base_cfg = ClassifierConfig(max_feature_points=4_000_000)
    cache = {}
    for s in scenes:
        thin, X, _, full = thinned_features(s, cloud[s.key], base_cfg)
        gt, gl = read_ply_xyz(s.gt_ply, with_classification=True)
        y = transfer_labels(gt, gl, thin, base_cfg.label_transfer_radius)
        cache[s.key] = (thin, X, y, full, gt, gl)
        print(f"[{s.key}] features loaded: {len(thin)} pts", flush=True)

    results = []
    for weighting in ("none", "sqrt", "balanced"):
        cfg = ClassifierConfig(max_feature_points=4_000_000, class_weighting=weighting)
        models, probas = {}, {}
        for s in scenes:
            models[s.key] = fit(cache[s.key][1], cache[s.key][2], cfg)
        for rule in ("argmax", "iou_bias"):
            fold = {}
            for held in scenes:
                tr = [s for s in scenes if s.key != held.key][0]
                m = models[tr.key]
                # tune on the training scene's own predictions
                if rule == "iou_bias":
                    if (tr.key, weighting) not in probas:
                        probas[(tr.key, weighting)] = predict_proba(m, cache[tr.key][1])
                    # log-domain: use log proba for additive bias
                    lp_tr = np.log(np.clip(probas[(tr.key, weighting)], 1e-9, 1))
                    bias, tuned = tune_bias(lp_tr, cache[tr.key][2])
                else:
                    bias = np.zeros(NUM_CLASSES, np.float32)
                thin, X, _, full, gt, gl = cache[held.key]
                lp = np.log(np.clip(predict_proba(m, X), 1e-9, 1))
                pred_thin = np.argmax(lp + bias, 1).astype(np.uint8)
                pred = propagate(thin, pred_thin, full)
                c = score_semantics(full, pred, gt, gl)
                fold[held.key] = c
            mean = float(np.mean([c.miou for c in fold.values()]))
            row = {"weighting": weighting, "rule": rule, "miou_mean": mean,
                   "folds": {k: c.miou for k, c in fold.items()},
                   "per_class_009": [round(float(v), 3)
                                     for v in fold["gold_coast/scene_009"].per_class_iou]}
            results.append(row)
            f9, f10 = fold["gold_coast/scene_009"].miou, fold["gold_coast/scene_010"].miou
            print(f"  {weighting:9s} + {rule:8s}: LOSO {mean:.4f} "
                  f"(folds {f9:.4f} / {f10:.4f})", flush=True)
            (out / "results.json").write_text(json.dumps(results, indent=2))

    best = max(results, key=lambda r: r["miou_mean"])
    ok = (best["folds"]["gold_coast/scene_009"] > 0.1567
          and best["folds"]["gold_coast/scene_010"] > 0.2125)
    print(f"\nbest: {best['weighting']}+{best['rule']} at {best['miou_mean']:.4f} "
          f"(baseline 0.1846); both folds beat uniform-4M: {ok}",
          "-> ADOPT" if ok and best["miou_mean"] > 0.1846 else "-> keep uniform-4M argmax")


if __name__ == "__main__":
    main()
