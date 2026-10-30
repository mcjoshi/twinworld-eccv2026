"""Find the slab configuration that actually FITS: coarser voxels + radial crop
of the interp band (base always kept whole, so downside stays bounded)."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from twinworld.dataio import CLASS_NAMES, NUM_CLASSES, Scene
from twinworld.package import submission_cloud
from twinworld.geometry.densify import robust_core
from twinworld.scoring import read_ply_xyz, voxel_downsample_indices

def convergence_point(frames):
    A=np.zeros((3,3)); b=np.zeros(3)
    for f in frames:
        d=f.R.T@np.array([0.,0.,1.]); d/=np.linalg.norm(d)
        P=np.eye(3)-np.outer(d,d); A+=P; b+=P@f.center
    return np.linalg.solve(A,b)

OUT=Path("experiments/2026-08-27-slab-fit"); OUT.mkdir(parents=True, exist_ok=True)
rows=[]
for sid in ("scene_009","scene_010"):
    s=Scene("gold_coast",sid)
    base,_=submission_cloud(s, Path(f"experiments/2026-08-08-pipeline-v1/gold_coast/{sid}/mvs/fused_gs_i.ply"), voxel=0.10)
    nx=np.load(f"experiments/2026-08-26-interp-max/{sid}_interp.npz")["xyz"].astype(np.float64)
    nx=nx[robust_core(nx,s.train_frames)]
    gt,gl=read_ply_xyz(s.gt_ply,with_classification=True)
    bt=cKDTree(base)
    d2b,_=bt.query(nx,k=1,distance_upper_bound=0.5,workers=-1)
    c=convergence_point(s.train_frames)
    dr=np.linalg.norm(nx-c,axis=1)
    for slab,R,v in ((0.20,80,0.15),(0.20,80,0.17),(0.15,70,0.13),(0.15,70,0.15),(0.30,80,0.17)):
        band=nx[np.isfinite(d2b)&(d2b<=slab)&(dr<=R)]
        union=np.vstack([base,band])
        sub=union[voxel_downsample_indices(union,v)]
        d,_=cKDTree(sub).query(gt,k=1,distance_upper_bound=0.10,workers=-1)
        cov=np.isfinite(d)
        pc={CLASS_NAMES[k]:round(float(cov[gl==k].mean()),4) for k in range(NUM_CLASSES) if (gl==k).any()}
        ceil=float(np.mean(list(pc.values())))
        rows.append({"scene":s.key,"slab":slab,"R":R,"voxel":v,"points":int(len(sub)),"ceiling":round(ceil,4)})
        print(f"[{s.key}] slab{slab} R{R} v{v}: {len(sub)/1e6:6.1f}M  ceiling {ceil:.4f}  "
              +" ".join(f"{k[:4]} {x:.2f}" for k,x in pc.items()), flush=True)
        (OUT/"results.json").write_text(json.dumps(rows,indent=2))
