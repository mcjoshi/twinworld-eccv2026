"""Per-scene union of APD-MVS and loose_2x for every TUM scene -> fused_union.ply."""
import numpy as np, json, time
from pathlib import Path
from twinworld.dataio import Scene
from twinworld.package import submission_cloud
from twinworld.scoring import read_ply_xyz, voxel_downsample, write_ply_xyz
for sid in [f"scene_{i:03d}" for i in range(9)]:
    s=Scene("tum",sid); t0=time.time()
    out=Path(f"experiments/2026-08-08-pipeline-v1/tum/{sid}/mvs/fused_union.ply")
    if out.exists(): print(f"{sid}: exists"); continue
    apd,_=read_ply_xyz(f"/data2/manish_joshi/twinworld/work/apdmvs/tum/{sid}/conv/APD/APD.ply")
    base=out.parent/"fused.ply"
    loose,_=submission_cloud(s, base, cache=out.parent/"submission_cloud.npz")
    u=voxel_downsample(np.vstack([apd,loose]),0.02)
    write_ply_xyz(out,u)
    print(f"{sid}: apd {len(apd)} + loose {len(loose)} -> union {len(u)} ({time.time()-t0:.0f}s)", flush=True)
