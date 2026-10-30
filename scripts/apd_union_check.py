"""Is APD union loose_2x better than APD alone? Complementarity check."""
import numpy as np
from pathlib import Path
from twinworld.dataio import Scene
from twinworld.package import submission_cloud
from twinworld.scoring import read_ply_xyz, score_geometry, voxel_downsample
for sid in ("scene_000","scene_001","scene_002","scene_003"):
    s=Scene("tum",sid)
    apd,_=read_ply_xyz(f"/data2/manish_joshi/twinworld/work/apdmvs/tum/{sid}/conv/APD/APD.ply")
    base_ply=Path(f"experiments/2026-08-08-pipeline-v1/tum/{sid}/mvs/fused.ply")
    loose,_=submission_cloud(s, base_ply, cache=base_ply.parent/"submission_cloud.npz")
    gt,_=read_ply_xyz(s.gt_ply)
    a=score_geometry(apd,gt,s.key); l=score_geometry(loose,gt,s.key)
    u=score_geometry(voxel_downsample(np.vstack([apd,loose]),0.02),gt,s.key)
    print(f"{sid}: loose {l.fscore:.4f}  apd {a.fscore:.4f}  union {u.fscore:.4f} "
          f"(P@10 {u.precision[0.10]:.3f} R@10 {u.recall[0.10]:.3f})", flush=True)
