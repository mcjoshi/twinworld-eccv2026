"""Radial grading around the optical-axis convergence point, not the rig centroid.

The rig-centroid grading failed because camera POSITIONS sit off to one side:
ground truth lies at p50 101 m from the centroid, co-located with the point bulk.
But the cameras LOOK AT the building. The least-squares intersection of their
optical axes estimates the true target centre from poses alone - test-time legal.
"""
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from twinworld.dataio import CLASS_NAMES, NUM_CLASSES, Scene
from twinworld.package import submission_cloud
from twinworld.geometry.densify import robust_core
from twinworld.scoring import read_ply_xyz, voxel_downsample_indices

def convergence_point(frames):
    A = np.zeros((3,3)); b = np.zeros(3)
    for f in frames:
        d = f.R.T @ np.array([0.0,0.0,1.0]); d /= np.linalg.norm(d)
        P = np.eye(3) - np.outer(d,d)
        A += P; b += P @ f.center
    return np.linalg.solve(A, b)

OUT = Path("experiments/2026-08-27-conv-graded"); OUT.mkdir(parents=True, exist_ok=True)
rows=[]
for sid in ("scene_009","scene_010"):
    s = Scene("gold_coast", sid)
    base,_ = submission_cloud(s, Path(f"experiments/2026-08-08-pipeline-v1/gold_coast/{sid}/mvs/fused_gs_i.ply"), voxel=0.10)
    nx = np.load(f"experiments/2026-08-26-interp-max/{sid}_interp.npz")["xyz"].astype(np.float64)
    nx = nx[robust_core(nx, s.train_frames)]
    union = np.vstack([base, nx])
    gt, gl = read_ply_xyz(s.gt_ply, with_classification=True)
    c = convergence_point(s.train_frames)
    du = np.linalg.norm(union - c, axis=1); dg = np.linalg.norm(gt - c, axis=1)
    print(f"\n[{s.key}] conv-point GT dist p50/p99 {np.percentile(dg,[50,99]).round(0)}  "
          f"pts p50/p90 {np.percentile(du,[50,90]).round(0)}", flush=True)
    for R in (float(np.percentile(dg,99))*0+60, 80, 100, 120):
        gt_in = float((dg<=R).mean())
        for fine, coarse in ((0.11,0.35),(0.13,0.40)):
            near = union[du<=R]; far = union[du>R]
            nv = near[voxel_downsample_indices(near, fine)]
            fv = far[voxel_downsample_indices(far, coarse)] if len(far) else far
            sub = np.vstack([nv, fv]) if len(fv) else nv
            d,_ = cKDTree(sub).query(gt, k=1, distance_upper_bound=0.10, workers=-1)
            cov = np.isfinite(d)
            pc = {CLASS_NAMES[k]: round(float(cov[gl==k].mean()),4) for k in range(NUM_CLASSES) if (gl==k).any()}
            ceil = float(np.mean(list(pc.values())))
            rows.append({"scene":s.key,"R":R,"fine":fine,"coarse":coarse,"gt_in":round(gt_in,4),
                         "points":int(len(sub)),"ceiling":round(ceil,4)})
            print(f"  R={R:4.0f} fine={fine} coarse={coarse}: {len(sub)/1e6:6.1f}M  "
                  f"GT-in {gt_in*100:5.1f}%  ceiling {ceil:.4f}", flush=True)
            (OUT/"results.json").write_text(json.dumps(rows,indent=2))
