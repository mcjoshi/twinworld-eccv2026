"""What does the Gold Coast submission voxel buy in ACHIEVED mIoU?

The ceiling analysis says 20 cm -> 10 cm nearly doubles the coverage ceiling
(0.258 -> 0.421 on scene_009, 0.279 -> 0.469 on scene_010). A ceiling is not a
score: it is an upper bound that assumes every reachable point is also labelled
correctly. Our measured conversion rate was about 60 percent, but it was
measured on 14 cm clouds and there is no reason it holds at 10 cm, where the
newly reachable points are the harder ones near class boundaries.

So this runs the real thing: same classifier, same features, same leave-one-
scene-out protocol, varying only the voxel of the cloud the labels are
propagated onto - which is the only thing the packaging decision changes.

Adoption gate, fixed before running: a finer voxel is adopted only if BOTH folds
beat their own 20 cm result. Point counts are recorded alongside, because the
gain has to be paid for out of the scorer's runtime budget and the exchange rate
against TUM F-score is the actual decision.
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
OUT = Path("experiments/2026-08-13-gc-voxel-achieved")
VOXELS = [0.20, 0.14, 0.10, 0.06]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = ClassifierConfig(max_feature_points=4_000_000)
    scenes = [Scene("gold_coast", s) for s in ("scene_009", "scene_010")]
    ply = {s.key: V1 / s.dataset / s.scene_id / "mvs" / "fused_gs_i.ply"
           for s in scenes}

    # features and labels once per scene; only the propagation target varies
    cache = {}
    for s in scenes:
        t0 = time.time()
        thin, X, _, _ = thinned_features(s, ply[s.key], cfg)
        gt, gl = read_ply_xyz(s.gt_ply, with_classification=True)
        y = transfer_labels(gt, gl, thin, cfg.label_transfer_radius)
        cache[s.key] = (thin, X, y, gt, gl)
        print(f"[{s.key}] {len(thin)} feature pts, "
              f"{int((y < NUM_CLASSES).sum())} labelled ({time.time()-t0:.0f}s)",
              flush=True)

    models = {s.key: fit(cache[s.key][1], cache[s.key][2], cfg) for s in scenes}
    print("models fitted", flush=True)

    rows = []
    for voxel in VOXELS:
        fold = {}
        for held in scenes:
            t0 = time.time()
            tr = [s for s in scenes if s.key != held.key][0]
            thin, X, _, gt, gl = cache[held.key]
            full, _ = submission_cloud(held, ply[held.key], voxel=voxel)
            pred_thin = predict(models[tr.key], X, n_jobs=cfg.n_jobs)
            pred = propagate(thin, pred_thin, full)
            c = score_semantics(full, pred, gt, gl)
            iou = c.per_class_iou
            fold[held.key] = {
                "miou": round(float(c.miou), 4),
                "ceiling": round(float(c.miou_ceiling), 4),
                "points": int(len(full)),
                "per_class": {CLASS_NAMES[i]: round(float(iou[i]), 4)
                              for i in range(NUM_CLASSES)},
                "seconds": round(time.time() - t0, 1),
            }
            print(f"  [v={voxel} held {held.key}] mIoU {c.miou:.4f} "
                  f"(ceiling {c.miou_ceiling:.4f}, {len(full)/1e6:.1f}M pts)  "
                  + "  ".join(f"{CLASS_NAMES[i]} {iou[i]:.3f}"
                              for i in range(NUM_CLASSES))
                  + f"  ({fold[held.key]['seconds']}s)", flush=True)
        mean = float(np.mean([f["miou"] for f in fold.values()]))
        total = int(sum(f["points"] for f in fold.values()))
        rows.append({"voxel": voxel, "loso_miou": round(mean, 4),
                     "points_2scenes": total, "folds": fold})
        print(f"[v={voxel}] LOSO mIoU {mean:.4f} over {total/1e6:.1f}M pts\n",
              flush=True)
        (OUT / "results.json").write_text(json.dumps(rows, indent=2))

    base = next(r for r in rows if r["voxel"] == 0.20)
    print("=== gate: both folds must beat their own 20 cm result ===")
    for r in rows:
        if r["voxel"] == 0.20:
            continue
        both = all(r["folds"][k]["miou"] > base["folds"][k]["miou"] for k in base["folds"])
        extra = r["points_2scenes"] - base["points_2scenes"]
        gain = r["loso_miou"] - base["loso_miou"]
        print(f"  v={r['voxel']}: LOSO {base['loso_miou']:.4f} -> {r['loso_miou']:.4f} "
              f"({gain:+.4f}) for {extra/1e6:+.1f}M pts  both folds improve: {both}"
              f"  {'-> ADOPT' if both and gain > 0 else '-> REJECT'}")


if __name__ == "__main__":
    main()
