"""The rare-class operating point, re-examined at v8/v9 packaging.

mIoU averages over five classes and does not weight by frequency, so a rare
class handled badly costs exactly as much as a common one. The per-class
diagnostic makes the priority obvious in hindsight:

    class    coverage   IoU    conversion
    other      0.545   0.049       9%
    ground     0.429   0.183      43%
    window     0.253   0.061      24%

"other" is reached by the scorer more than half the time and converts at nine
percent. Lifting it alone from 0.049 to 0.25 is worth +0.040 mIoU - the whole
gap to the current leader.

A decision-rule sweep was run on 11 August and rejected. That rejection is stale:
it was measured at 20 cm packaging, where the class balance *among reachable
points* was completely different (the coverage ceiling was 0.26, not 0.44), and
the tuning was done against a baseline that has since moved. The experiment is
therefore re-run rather than assumed.

Two knobs:
  weighting   none | sqrt | balanced | heavy   (training time)
  rule        argmax | bias vector tuned for pooled mIoU (post hoc)

The bias vector is tuned by coordinate ascent ON THE TRAINING SCENE ONLY,
against its own transferred labels, then applied frozen to the held-out scene,
so the leave-one-scene-out protocol stays honest.

Adoption gate, fixed before running: BOTH folds must beat their own baseline at
10 cm packaging (fold 009 > 0.1529, fold 010 > 0.2067, LOSO > 0.1798).
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
    ClassifierConfig, fit, propagate, thinned_features, transfer_labels,
)

V1 = Path("experiments/2026-08-08-pipeline-v1")
OUT = Path("experiments/2026-08-19-rare-class")
PACK_VOXEL = 0.10
BASE = {"gold_coast/scene_009": 0.1529, "gold_coast/scene_010": 0.2067}
BASE_LOSO = 0.1798


def predict_proba(model, X, batch=500_000):
    out = np.empty((len(X), NUM_CLASSES), np.float32)
    for a in range(0, len(X), batch):
        out[a:a + batch] = model.predict_proba(X[a:a + batch])
    return out


def masked_miou(pred, y):
    m = y < NUM_CLASSES
    p, t = pred[m], y[m]
    ious = []
    for c in range(NUM_CLASSES):
        inter = ((p == c) & (t == c)).sum()
        union = ((p == c) | (t == c)).sum()
        if union:
            ious.append(inter / union)
    return float(np.mean(ious)) if ious else 0.0


def tune_bias(log_proba, y, iters=3):
    """Coordinate ascent on per-class log-bias, maximising labelled mIoU."""
    bias = np.zeros(NUM_CLASSES, np.float32)
    grid = np.linspace(-3.0, 3.0, 25)
    best = masked_miou(np.argmax(log_proba, 1), y)
    for _ in range(iters):
        for c in range(NUM_CLASSES):
            cur = bias[c]
            vals = []
            for g in grid:
                b = bias.copy()
                b[c] = g
                vals.append((masked_miou(np.argmax(log_proba + b, 1), y), g))
            v, g = max(vals)
            bias[c] = g if v > best else cur
            best = max(best, v)
    return bias, best


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    scenes = [Scene("gold_coast", s) for s in ("scene_009", "scene_010")]
    ply = {s.key: V1 / s.dataset / s.scene_id / "mvs" / "fused_gs_i.ply"
           for s in scenes}
    base_cfg = ClassifierConfig(max_feature_points=4_000_000)

    cache = {}
    for s in scenes:
        t0 = time.time()
        thin, X, _, _ = thinned_features(s, ply[s.key], base_cfg)
        gt, gl = read_ply_xyz(s.gt_ply, with_classification=True)
        y = transfer_labels(gt, gl, thin, base_cfg.label_transfer_radius)
        full, _ = submission_cloud(s, ply[s.key], voxel=PACK_VOXEL)
        cache[s.key] = (thin, X, y, gt, gl, full)
        print(f"[{s.key}] {len(thin)} feature pts, {len(full)/1e6:.1f}M packaged "
              f"({time.time()-t0:.0f}s)", flush=True)

    results = []
    for weighting in ("balanced", "sqrt", "heavy", "none"):
        w = "balanced" if weighting == "heavy" else weighting
        cfg = ClassifierConfig(max_feature_points=4_000_000, class_weighting=w)
        models = {s.key: fit(cache[s.key][1], cache[s.key][2], cfg) for s in scenes}
        proba = {s.key: np.log(np.clip(predict_proba(models[s.key], cache[s.key][1]),
                                       1e-9, 1)) for s in scenes}
        for rule in ("argmax", "iou_bias"):
            fold = {}
            for held in scenes:
                t0 = time.time()
                tr = [s for s in scenes if s.key != held.key][0]
                if rule == "iou_bias":
                    bias, _ = tune_bias(proba[tr.key], cache[tr.key][2])
                else:
                    bias = np.zeros(NUM_CLASSES, np.float32)
                thin, X, _, gt, gl, full = cache[held.key]
                lp = np.log(np.clip(predict_proba(models[tr.key], X), 1e-9, 1))
                pred_thin = np.argmax(lp + bias, 1).astype(np.uint8)
                pred = propagate(thin, pred_thin, full)
                c = score_semantics(full, pred, gt, gl)
                iou = c.per_class_iou
                fold[held.key] = {"miou": round(float(c.miou), 4),
                                  "bias": [round(float(b), 2) for b in bias],
                                  "per_class": {CLASS_NAMES[i]: round(float(iou[i]), 4)
                                                for i in range(NUM_CLASSES)},
                                  "seconds": round(time.time() - t0, 1)}
                print(f"  [{weighting}+{rule} held {held.key}] mIoU {c.miou:.4f} "
                      f"(base {BASE[held.key]:.4f}, {c.miou-BASE[held.key]:+.4f})  "
                      + "  ".join(f"{CLASS_NAMES[i]} {iou[i]:.3f}"
                                  for i in range(NUM_CLASSES)), flush=True)
            loso = float(np.mean([f["miou"] for f in fold.values()]))
            both = all(fold[k]["miou"] > BASE[k] for k in BASE)
            results.append({"weighting": weighting, "rule": rule,
                            "loso_miou": round(loso, 4), "both_folds": both,
                            "verdict": "ADOPT" if both and loso > BASE_LOSO else "REJECT",
                            "folds": fold})
            print(f"[{weighting}+{rule}] LOSO {loso:.4f} ({loso-BASE_LOSO:+.4f}) "
                  f"both folds: {both} -> {results[-1]['verdict']}\n", flush=True)
            (OUT / "results.json").write_text(json.dumps(results, indent=2))

    best = max(results, key=lambda r: r["loso_miou"])
    print(f"best: {best['weighting']}+{best['rule']} LOSO {best['loso_miou']:.4f} "
          f"-> {best['verdict']}")


if __name__ == "__main__":
    main()
