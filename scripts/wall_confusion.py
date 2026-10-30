"""Why does wall score 0.121 while being 56 percent of the scene?

IoU alone cannot distinguish three very different failures:

  under-prediction   wall points exist in ground truth but we call them
                     something else -> low recall, and the fix is features or
                     class weighting
  over-prediction    we call other things wall -> low precision, and the fix is
                     the decision rule
  unreachable        the wall points are not covered by our cloud at all -> the
                     fix is reconstruction, and no classifier work will help

The coverage ceiling separates the third from the first two; a confusion matrix
over covered points separates the first from the second. This computes both, on
the held-out fold, at the v8 packaging.

Diagnostic only - nothing is adopted or rejected here.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from twinworld.dataio import CLASS_NAMES, NUM_CLASSES, Scene
from twinworld.package import submission_cloud
from twinworld.scoring import read_ply_xyz, score_semantics
from twinworld.semantics.classifier import (
    ClassifierConfig, fit, predict, propagate, thinned_features, transfer_labels,
)

V1 = Path("experiments/2026-08-08-pipeline-v1")
OUT = Path("experiments/2026-08-16-wall-confusion")
PACK_VOXEL = 0.10
MATCH_R = 0.10


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = ClassifierConfig(max_feature_points=4_000_000)
    scenes = [Scene("gold_coast", s) for s in ("scene_009", "scene_010")]
    ply = {s.key: V1 / s.dataset / s.scene_id / "mvs" / "fused_gs_i.ply"
           for s in scenes}

    cache = {}
    for s in scenes:
        thin, X, _, _ = thinned_features(s, ply[s.key], cfg)
        gt, gl = read_ply_xyz(s.gt_ply, with_classification=True)
        y = transfer_labels(gt, gl, thin, cfg.label_transfer_radius)
        cache[s.key] = (thin, X, y, gt, gl)
        print(f"[{s.key}] features ready: {len(thin)} pts", flush=True)

    out = {}
    for held in scenes:
        t0 = time.time()
        tr = [s for s in scenes if s.key != held.key][0]
        model = fit(cache[tr.key][1], cache[tr.key][2], cfg)
        thin, X, _, gt, gl = cache[held.key]
        full, _ = submission_cloud(held, ply[held.key], voxel=PACK_VOXEL)
        pred = propagate(thin, predict(model, X, n_jobs=cfg.n_jobs), full)

        # what the scorer sees: each GT point takes its nearest submitted label
        d, idx = cKDTree(full).query(gt, k=1, distance_upper_bound=MATCH_R,
                                     workers=-1)
        covered = np.isfinite(d)
        gt_lab = gl[covered]
        pr_lab = pred[idx[covered]]

        # per-class coverage: the reconstruction-side failure
        cov = {}
        for c in range(NUM_CLASSES):
            m = gl == c
            cov[CLASS_NAMES[c]] = round(float(covered[m].mean()), 4) if m.any() else None

        # confusion over covered points only: the classification-side failure
        conf = np.zeros((NUM_CLASSES, NUM_CLASSES), np.int64)
        for t in range(NUM_CLASSES):
            m = gt_lab == t
            if not m.any():
                continue
            for p in range(NUM_CLASSES):
                conf[t, p] = int((pr_lab[m] == p).sum())

        c_all = score_semantics(full, pred, gt, gl)
        rec = {CLASS_NAMES[t]: round(float(conf[t, t] / max(conf[t].sum(), 1)), 4)
               for t in range(NUM_CLASSES)}
        prec = {CLASS_NAMES[p]: round(float(conf[p, p] / max(conf[:, p].sum(), 1)), 4)
                for p in range(NUM_CLASSES)}

        out[held.key] = {"miou": round(float(c_all.miou), 4),
                         "per_class_iou": {CLASS_NAMES[i]: round(float(c_all.per_class_iou[i]), 4)
                                           for i in range(NUM_CLASSES)},
                         "coverage": cov,
                         "recall_given_covered": rec,
                         "precision_given_covered": prec,
                         "confusion": conf.tolist(),
                         "gt_class_share": {CLASS_NAMES[c]: round(float((gl == c).mean()), 4)
                                            for c in range(NUM_CLASSES)}}

        print(f"\n=== held out {held.key} (mIoU {c_all.miou:.4f}, "
              f"{time.time()-t0:.0f}s) ===", flush=True)
        print(f"{'class':8s} {'GT share':>9} {'coverage':>9} {'recall|cov':>11} "
              f"{'prec|cov':>9} {'IoU':>7}")
        for c in range(NUM_CLASSES):
            n = CLASS_NAMES[c]
            print(f"{n:8s} {out[held.key]['gt_class_share'][n]:9.3f} "
                  f"{cov[n] if cov[n] is not None else float('nan'):9.3f} "
                  f"{rec[n]:11.3f} {prec[n]:9.3f} "
                  f"{out[held.key]['per_class_iou'][n]:7.3f}")
        print("\nconfusion (rows = ground truth, cols = predicted), covered points:")
        print(f"{'':8s}" + "".join(f"{CLASS_NAMES[p]:>10s}" for p in range(NUM_CLASSES)))
        for t in range(NUM_CLASSES):
            tot = max(conf[t].sum(), 1)
            print(f"{CLASS_NAMES[t]:8s}" +
                  "".join(f"{conf[t, p]/tot:10.3f}" for p in range(NUM_CLASSES)))
        (OUT / "results.json").write_text(json.dumps(out, indent=2))

    (OUT / "results.json").write_text(json.dumps(out, indent=2))
    print("\ndone", flush=True)


if __name__ == "__main__":
    main()
