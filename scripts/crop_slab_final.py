"""R-crop + slab + fine voxel: the configuration that could actually fit.

Convergence-point radius keeps 100.0 percent of dev ground truth at R=120 m with
a 2x margin over GT p99 (50/69 m). The capture protocol orbits one target
building, so the radius transfers by construction of the challenge itself.
Cropping the far-field entirely - base included - is what frees the budget the
band needs. Saves the resulting cloud per scene for the fusion gate.
"""
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from twinworld.dataio import CLASS_NAMES, NUM_CLASSES, Scene
from twinworld.package import submission_cloud
from twinworld.geometry.densify import robust_core
from twinworld.scoring import read_ply_xyz, voxel_downsample_indices

def conv_point(frames):
    A=np.zeros((3,3)); b=np.zeros(3)
    for f in frames:
        d=f.R.T@np.array([0.,0.,1.]); d/=np.linalg.norm(d)
        P=np.eye(3)-np.outer(d,d); A+=P; b+=P@f.center
    return np.linalg.solve(A,b)

OUT=Path("experiments/2026-08-27-crop-slab"); OUT.mkdir(parents=True, exist_ok=True)
rows=[]
for sid in ("scene_009","scene_010"):
    s=Scene("gold_coast",sid)
    base,_=submission_cloud(s, Path(f"experiments/2026-08-08-pipeline-v1/gold_coast/{sid}/mvs/fused_gs_i.ply"), voxel=0.10)
    nx=np.load(f"experiments/2026-08-26-interp-max/{sid}_interp.npz")["xyz"].astype(np.float64)
    nx=nx[robust_core(nx,s.train_frames)]
    gt,gl=read_ply_xyz(s.gt_ply,with_classification=True)
    c=conv_point(s.train_frames)
    d2b,_=cKDTree(base).query(nx,k=1,distance_upper_bound=0.5,workers=-1)
    for R in (120.0, 150.0):
        gt_in=float((np.linalg.norm(gt-c,axis=1)<=R).mean())
        bin_=base[np.linalg.norm(base-c,axis=1)<=R]
        for slab,v in ((0.20,0.12),(0.20,0.13),(0.30,0.13)):
            band=nx[np.isfinite(d2b)&(d2b<=slab)]
            band=band[np.linalg.norm(band-c,axis=1)<=R]
            union=np.vstack([bin_,band])
            sub=union[voxel_downsample_indices(union,v)]
            d,_=cKDTree(sub).query(gt,k=1,distance_upper_bound=0.10,workers=-1)
            cov=np.isfinite(d)
            pc={CLASS_NAMES[k]:round(float(cov[gl==k].mean()),4) for k in range(NUM_CLASSES) if (gl==k).any()}
            ceil=float(np.mean(list(pc.values())))
            rows.append({"scene":s.key,"R":R,"slab":slab,"voxel":v,"gt_in":round(gt_in,5),
                         "points":int(len(sub)),"ceiling":round(ceil,4)})
            print(f"[{s.key}] R{R:.0f} slab{slab} v{v}: {len(sub)/1e6:6.1f}M  "
                  f"GT-in {gt_in*100:6.2f}%  ceiling {ceil:.4f}  "
                  +" ".join(f"{k[:4]} {x:.2f}" for k,x in pc.items()), flush=True)
            (OUT/"results.json").write_text(json.dumps(rows,indent=2))
            if R==120.0 and slab==0.20 and v==0.12:
                np.savez_compressed(OUT/f"{sid}_cloud.npz", xyz=sub.astype(np.float32))
                print(f"  saved {sid}_cloud.npz", flush=True)
