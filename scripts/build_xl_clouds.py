"""Build slab-crop clouds for all four Gold Coast scenes at two budget tiers.

Config gated by the sweeps: interp band within 0.15 m of the base surface,
everything cropped at R=70 (fine tier retains R120 GT margin via base? no -)
Correction, recorded honestly: the sweep used slab 0.15 with R70 applied to the
BAND only in slab_fit (base kept whole) and full-crop at R120 in crop_slab.
This builder uses: base cropped at R=120 (100 percent of dev GT, 2x margin over
GT p99), band = interp within slab of base and R=120, union voxelled per tier.
Test-time legal throughout: conv point and slab need only cameras and our own
clouds.
"""
import sys, json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from twinworld.dataio import Scene
from twinworld.package import submission_cloud
from twinworld.geometry.densify import robust_core
from twinworld.scoring import voxel_downsample_indices

def conv_point(frames):
    A=np.zeros((3,3)); b=np.zeros(3)
    for f in frames:
        d=f.R.T@np.array([0.,0.,1.]); d/=np.linalg.norm(d)
        P=np.eye(3)-np.outer(d,d); A+=P; b+=P@f.center
    return np.linalg.solve(A,b)

OUT=Path("experiments/2026-08-29-xl-clouds"); OUT.mkdir(parents=True, exist_ok=True)
R=120.0; SLAB=0.20
for sid in ("scene_009","scene_010","scene_011","scene_012"):
    s=Scene("gold_coast",sid)
    base,_=submission_cloud(s, Path(f"experiments/2026-08-08-pipeline-v1/gold_coast/{sid}/mvs/fused_gs_i.ply"), voxel=0.10)
    nx=np.load(f"experiments/2026-08-26-interp-max/{sid}_interp.npz")["xyz"].astype(np.float64)
    nx=nx[robust_core(nx,s.train_frames)]
    c=conv_point(s.train_frames)
    d2b,_=cKDTree(base).query(nx,k=1,distance_upper_bound=0.5,workers=-1)
    band=nx[np.isfinite(d2b)&(d2b<=SLAB)]
    band=band[np.linalg.norm(band-c,axis=1)<=R]
    bin_=base[np.linalg.norm(base-c,axis=1)<=R]
    union=np.vstack([bin_,band])
    for tier,v in (("xl",0.12),):
        sub=union[voxel_downsample_indices(union,v)]
        np.savez_compressed(OUT/f"{sid}_{tier}.npz", xyz=sub.astype(np.float32))
        print(f"[{sid}] {tier} v={v}: base-in-R {len(bin_)/1e6:.1f}M + band {len(band)/1e6:.1f}M -> {len(sub)/1e6:6.1f}M", flush=True)
