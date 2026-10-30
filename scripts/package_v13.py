"""Package v12: TUM at 4 cm, Gold Coast from the offset-sheet clouds.

TUM at 4 cm is the packaging that measured F 0.627 on the final scenes on
20 August, so its geometry is observed rather than projected. Gold Coast uses the
outward-offset thinned clouds gated at LOSO 0.1915, against 0.1798 for the plain
10 cm cloud v11 ships - the first offset configuration cheap enough to fit.
"""
from pathlib import Path
import numpy as np
from twinworld.dataio import ALL_SCENES, Scene
from twinworld.package import SceneArtefacts, build_submission, submission_cloud

V1 = Path("experiments/2026-08-08-pipeline-v1")
V7 = Path("experiments/2026-08-11-pipeline-v7")
TUM_VOXEL = 0.04

arts = []
for dataset, sid in ALL_SCENES:
    s = Scene(dataset, sid)
    rgb = V7 / dataset / sid / "gs" / "rgb"
    if s.needs_semantics:
        d = np.load(V1 / dataset / sid / "mvs" / "fused_gs_i_v13.npz")
        xyz, labels = d["xyz"].astype(np.float64), d["labels"]
    else:
        ply = V1 / dataset / sid / "mvs" / "fused_union.ply"
        xyz, _ = submission_cloud(s, ply, voxel=TUM_VOXEL)
        labels = None
    print(f"  {s.key}: {len(xyz)/1e6:.1f}M points", flush=True)
    arts.append(SceneArtefacts(scene=s, rgb_dir=rgb, xyz=xyz, labels=labels))

rep = build_submission(arts, Path("submissions/v13-full.zip"), downsample=None)
print("VALID" if rep.ok else "INVALID", rep.errors[:3])
print("zip_mb", rep.stats.get("zip_mb"))
