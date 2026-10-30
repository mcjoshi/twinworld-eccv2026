"""Feature granularity against packaging granularity.

Until 13 Aug the submission voxel for Gold Coast was 20 cm and the feature voxel
was 15 cm, so features were finer than the cloud they labelled and the mismatch
was invisible. v8 ships at 10 cm, which inverts the relationship: labels are now
decided on a 15 cm thinning and propagated onto a 10 cm cloud, so points further
apart than the scorer's own 10 cm match radius are forced to share a label.

That is a direct cap on the conversion rate - we reach a ceiling of ~0.44 and
achieve 0.180, a 41 percent conversion - and it costs nothing but compute to
test.

Swept: feature_voxel against the uniform feature budget that keeps the thinned
cloud representative. Everything else is held fixed.

Adoption gate, fixed before running: BOTH folds must beat their own baseline at
the same packaging (fold 009 > 0.1529, fold 010 > 0.2067, LOSO > 0.1798).
"""

from __future__ import annotations

import argparse
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
OUT = Path("experiments/2026-08-14-feature-voxel")
PACK_VOXEL = 0.10          # the v8 Gold Coast packaging voxel
BASE = {"gold_coast/scene_009": 0.1529, "gold_coast/scene_010": 0.2067}
BASE_LOSO = 0.1798


def run(feature_voxel: float, max_feature_points: int, tag: str,
        label_radius: float = 0.15) -> dict:
    cfg = ClassifierConfig(feature_voxel=feature_voxel,
                           max_feature_points=max_feature_points,
                           label_transfer_radius=label_radius)
    scenes = [Scene("gold_coast", s) for s in ("scene_009", "scene_010")]
    ply = {s.key: V1 / s.dataset / s.scene_id / "mvs" / "fused_gs_i.ply"
           for s in scenes}

    cache = {}
    for s in scenes:
        t0 = time.time()
        thin, X, _, _ = thinned_features(s, ply[s.key], cfg)
        gt, gl = read_ply_xyz(s.gt_ply, with_classification=True)
        y = transfer_labels(gt, gl, thin, cfg.label_transfer_radius)
        cache[s.key] = (thin, X, y, gt, gl)
        print(f"[{tag}][{s.key}] {len(thin)} feature pts, "
              f"{int((y < NUM_CLASSES).sum())} labelled ({time.time()-t0:.0f}s)",
              flush=True)

    fold = {}
    for held in scenes:
        t0 = time.time()
        tr = [s for s in scenes if s.key != held.key][0]
        model = fit(cache[tr.key][1], cache[tr.key][2], cfg)
        thin, X, _, gt, gl = cache[held.key]
        full, _ = submission_cloud(held, ply[held.key], voxel=PACK_VOXEL)
        pred = propagate(thin, predict(model, X, n_jobs=cfg.n_jobs), full)
        c = score_semantics(full, pred, gt, gl)
        iou = c.per_class_iou
        fold[held.key] = {"miou": round(float(c.miou), 4),
                          "ceiling": round(float(c.miou_ceiling), 4),
                          "per_class": {CLASS_NAMES[i]: round(float(iou[i]), 4)
                                        for i in range(NUM_CLASSES)},
                          "seconds": round(time.time() - t0, 1)}
        print(f"[{tag}][held {held.key}] mIoU {c.miou:.4f} "
              f"(base {BASE[held.key]:.4f}, {c.miou-BASE[held.key]:+.4f})  "
              + "  ".join(f"{CLASS_NAMES[i]} {iou[i]:.3f}"
                          for i in range(NUM_CLASSES))
              + f"  ({fold[held.key]['seconds']}s)", flush=True)

    loso = float(np.mean([f["miou"] for f in fold.values()]))
    both = all(fold[k]["miou"] > BASE[k] for k in BASE)
    row = {"tag": tag, "feature_voxel": feature_voxel,
           "max_feature_points": max_feature_points,
           "label_radius": label_radius,
           "loso_miou": round(loso, 4), "both_folds": both,
           "verdict": "ADOPT" if both and loso > BASE_LOSO else "REJECT",
           "folds": fold}
    print(f"[{tag}] LOSO {loso:.4f} (base {BASE_LOSO:.4f}, {loso-BASE_LOSO:+.4f}) "
          f"both folds improve: {both} -> {row['verdict']}\n", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{tag}.json").write_text(json.dumps(row, indent=2))
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-voxel", type=float, required=True)
    ap.add_argument("--max-feature-points", type=int, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--label-radius", type=float, default=0.15)
    a = ap.parse_args()
    run(a.feature_voxel, a.max_feature_points, a.tag, a.label_radius)


if __name__ == "__main__":
    main()
