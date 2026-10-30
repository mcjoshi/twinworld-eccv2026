"""Graded voxel by rig distance: can it fit interp-max into the budget?

The interp-max cloud reaches ceiling 0.67/0.78 but costs 191/155 M points per
scene - about 5x budget - and uniform coarsening to fit would displace points
past the 10 cm match radius. The cameras orbit the building of interest, so
distance from the rig centroid is a test-time-legal density signal. Grading
(fine near, coarse far) bounds the downside: a wrong radius degrades resolution
somewhere, it never deletes coverage.

Measures: where the points and the ground truth actually sit relative to the
rig, then the ceiling/point cost of fine-within-R, coarse-outside schemes.
"""
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from twinworld.dataio import CLASS_NAMES, NUM_CLASSES, Scene
from twinworld.package import submission_cloud
from twinworld.geometry.densify import robust_core
from twinworld.scoring import read_ply_xyz, voxel_downsample_indices

OUT = Path("experiments/2026-08-26-rig-graded"); OUT.mkdir(parents=True, exist_ok=True)
rows = []
for sid in ("scene_009", "scene_010"):
    s = Scene("gold_coast", sid)
    base, _ = submission_cloud(s, Path(f"experiments/2026-08-08-pipeline-v1/gold_coast/{sid}/mvs/fused_gs_i.ply"), voxel=0.10)
    nx = np.load(f"experiments/2026-08-26-interp-max/{sid}_interp.npz")["xyz"].astype(np.float64)
    nx = nx[robust_core(nx, s.train_frames)]
    union = np.vstack([base, nx])
    gt, gl = read_ply_xyz(s.gt_ply, with_classification=True)
    rig = np.stack([f.center for f in s.train_frames]).mean(axis=0)
    du = np.linalg.norm(union - rig, axis=1)
    dg = np.linalg.norm(gt - rig, axis=1)
    cam_r = float(np.percentile(np.linalg.norm(
        np.stack([f.center for f in s.train_frames]) - rig, axis=1), 90))
    print(f"\n[{s.key}] union {len(union)/1e6:.0f}M  cam-orbit r90 {cam_r:.0f}m  "
          f"GT dist p50/p99 {np.percentile(dg,[50,99]).round(0)}  "
          f"pts dist p50/p90 {np.percentile(du,[50,90]).round(0)}", flush=True)
    for R in (cam_r*1.2, cam_r*1.6, cam_r*2.2):
        gt_in = float((dg <= R).mean())
        for fine, coarse in ((0.11, 0.30), (0.12, 0.35)):
            near = union[du <= R]; far = union[du > R]
            nearv = near[voxel_downsample_indices(near, fine)]
            farv = far[voxel_downsample_indices(far, coarse)] if len(far) else far
            sub = np.vstack([nearv, farv]) if len(farv) else nearv
            d, _ = cKDTree(sub).query(gt, k=1, distance_upper_bound=0.10, workers=-1)
            cov = np.isfinite(d)
            pc = {CLASS_NAMES[c]: round(float(cov[gl==c].mean()),4) for c in range(NUM_CLASSES) if (gl==c).any()}
            ceil = float(np.mean(list(pc.values())))
            rows.append({"scene": s.key, "R": round(R,1), "fine": fine, "coarse": coarse,
                         "gt_inside_R": round(gt_in,4), "points": int(len(sub)),
                         "ceiling": round(ceil,4), "per_class": pc})
            print(f"  R={R:5.0f}m fine={fine} coarse={coarse}: {len(sub)/1e6:6.1f}M  "
                  f"GT-in-R {gt_in*100:5.1f}%  ceiling {ceil:.4f}", flush=True)
            (OUT/"results.json").write_text(json.dumps(rows, indent=2))
