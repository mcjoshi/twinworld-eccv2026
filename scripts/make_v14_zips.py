"""Package v14 (dense+fusion, f=0.11, 97.6M) and a safe f=0.12 variant."""
from pathlib import Path
import numpy as np
from twinworld.dataio import ALL_SCENES, Scene
from twinworld.package import SceneArtefacts, build_submission, submission_cloud
from twinworld.scoring import voxel_downsample_indices

V1 = Path("experiments/2026-08-08-pipeline-v1")
V7 = Path("experiments/2026-08-11-pipeline-v7")
for tag, revox in (("v14", None), ("v14safe", 0.12)):
    arts = []
    for dataset, sid in ALL_SCENES:
        s = Scene(dataset, sid)
        rgb = V7 / dataset / sid / "gs" / "rgb"
        if s.needs_semantics:
            d = np.load(V1 / dataset / sid / "mvs" / "fused_gs_i_v14.npz")
            xyz, labels = d["xyz"].astype(np.float64), d["labels"]
            if revox:
                keep = voxel_downsample_indices(xyz, revox)
                xyz, labels = xyz[keep], labels[keep]
        else:
            xyz, _ = submission_cloud(s, V1 / dataset / sid / "mvs" / "fused_union.ply", voxel=0.04)
            labels = None
        arts.append(SceneArtefacts(scene=s, rgb_dir=rgb, xyz=xyz, labels=labels))
        print(f"  [{tag}] {s.key}: {len(xyz)/1e6:.1f}M", flush=True)
    rep = build_submission(arts, Path(f"submissions/{tag}-full.zip"), downsample=None)
    print(tag, "VALID" if rep.ok else rep.errors[:2], rep.stats.get("zip_mb"), flush=True)
