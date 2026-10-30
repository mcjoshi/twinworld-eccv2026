"""Uniform 4M feature budget: more supervision without distribution shift.

Dense-train failed because training features came from a differently-sampled
cloud than inference features. This keeps sampling identical on both sides and
simply raises the shared uniform budget from 1.5M to 4M points, which lifts the
labelled subset from ~26k to ~70k. Same gate: LOSO > 0.1717, both folds improve.
"""
import json, time
from pathlib import Path
import numpy as np
from twinworld.dataio import CLASS_NAMES, NUM_CLASSES, Scene
from twinworld.scoring import read_ply_xyz, score_semantics
from twinworld.semantics.classifier import (ClassifierConfig, fit, predict, propagate,
                                            thinned_features, transfer_labels)

cfg = ClassifierConfig(max_feature_points=4_000_000)
scenes = [Scene("gold_coast", s) for s in ("scene_009", "scene_010")]
cloud = {s.key: Path("experiments/2026-08-08-pipeline-v1")/s.dataset/s.scene_id/"mvs"/"fused_gs_i.ply"
         for s in scenes}
cache = {}
for s in scenes:
    t0=time.time()
    thin, X, _, full = thinned_features(s, cloud[s.key], cfg)
    gt, gl = read_ply_xyz(s.gt_ply, with_classification=True)
    y = transfer_labels(gt, gl, thin, cfg.label_transfer_radius)
    cache[s.key]=(thin,X,y,full,gt,gl)
    print(f"[{s.key}] {len(thin)} uniform pts, {int((y<NUM_CLASSES).sum())} labelled ({time.time()-t0:.0f}s)", flush=True)
res={}
for held in scenes:
    t0=time.time()
    tr=[s for s in scenes if s.key!=held.key][0]
    model=fit(cache[tr.key][1], cache[tr.key][2], cfg)
    thin,X,_,full,gt,gl=cache[held.key]
    pred=propagate(thin, predict(model,X,n_jobs=cfg.n_jobs), full)
    c=score_semantics(full,pred,gt,gl); iou=c.per_class_iou
    res[held.key]=c.miou
    print(f"[held out {held.key}] mIoU {c.miou:.4f}   "
          +"  ".join(f"{CLASS_NAMES[i]} {iou[i]:.3f}" for i in range(NUM_CLASSES))
          +f"  ({time.time()-t0:.0f}s)", flush=True)
mean=float(np.mean(list(res.values())))
base={"gold_coast/scene_009":0.1449,"gold_coast/scene_010":0.1985}
both=all(res[k]>v for k,v in base.items())
Path("experiments/2026-08-11-uniform4m").mkdir(parents=True, exist_ok=True)
Path("experiments/2026-08-11-uniform4m/results.json").write_text(json.dumps(res,indent=2))
print(f"\nuniform-4M LOSO mean {mean:.4f} (baseline 0.1717); both folds improve: {both}",
      "-> ADOPT" if both and mean>0.1717 else "-> keep 1.5M")
