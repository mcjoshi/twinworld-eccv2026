"""Did splat-depth fusion make the Gold Coast cloud less ACCURATE?

The arithmetic that motivates this. Covering a 65 x 62 x 40 m volume so that any
point is within 10 cm needs ~105 M lattice points per scene - volumetric fill
does not fit the budget. But covering the building SURFACE needs under 1 M points
at 14 cm spacing. We submit 35.8 M and cover 42 percent of ground truth. The
points are not missing, they are misplaced: the median distance from ground truth
to our nearest submitted point is 18 cm, just outside the 10 cm match radius.

So the binding problem may be surface ACCURACY, not coverage strategy - and there
is an obvious suspect. Gold Coast ships `fused_gs_i.ply`, which is stereo unioned
with depth rendered from the splat model at interpolated viewpoints. That union
was adopted in August because it raised coverage. Splat-rendered depth is noisier
than stereo, and nobody checked whether adding it moved the surface away from
ground truth.

This compares, per cloud, the distance distribution from ground truth to the
nearest submitted point, and the coverage at the scored radius:
    fused.ply        stereo only
    fused_gs.ply     stereo + splat depth at training views
    fused_gs_i.ply   stereo + splat depth at interpolated views  (shipped)
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from twinworld.dataio import CLASS_NAMES, NUM_CLASSES, Scene
from twinworld.package import submission_cloud
from twinworld.scoring import read_ply_xyz

V1 = Path("experiments/2026-08-08-pipeline-v1")
OUT = Path("experiments/2026-08-29-cloud-accuracy")
OUT.mkdir(parents=True, exist_ok=True)
rows=[]
for sid in ("scene_009","scene_010"):
    s = Scene("gold_coast", sid)
    gt, gl = read_ply_xyz(s.gt_ply, with_classification=True)
    print(f"\n[{s.key}] GT {len(gt)/1e6:.1f}M", flush=True)
    for name in ("fused.ply","fused_gs.ply","fused_gs_i.ply"):
        p = V1/s.dataset/s.scene_id/"mvs"/name
        if not p.exists():
            print(f"  {name:16s} missing"); continue
        xyz,_ = submission_cloud(s, p, voxel=0.10)
        d,_ = cKDTree(xyz).query(gt, k=1, workers=-1)
        cov = {CLASS_NAMES[c]: round(float((d[gl==c]<=0.10).mean()),4)
               for c in range(NUM_CLASSES) if (gl==c).any()}
        ceil = float(np.mean(list(cov.values())))
        q = np.percentile(d,[25,50,75]).round(3)
        rows.append({"scene":s.key,"cloud":name,"points":int(len(xyz)),
                     "ceiling":round(ceil,4),"p25":float(q[0]),"p50":float(q[1]),
                     "p75":float(q[2]),"per_class":cov})
        print(f"  {name:16s} {len(xyz)/1e6:6.1f}M  ceiling {ceil:.4f}  "
              f"dist p25/p50/p75 {q[0]:.3f}/{q[1]:.3f}/{q[2]:.3f}", flush=True)
        (OUT/"results.json").write_text(json.dumps(rows,indent=2))
print("\nreading: if fused.ply has a SMALLER median distance and a HIGHER ceiling "
      "than fused_gs_i.ply, the splat fusion we adopted for coverage cost us "
      "accuracy, and accuracy is what the 10cm radius actually rewards.", flush=True)
