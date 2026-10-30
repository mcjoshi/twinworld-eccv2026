"""Package both v15 tiers from the labelled slab clouds."""
from pathlib import Path
import numpy as np
from twinworld.dataio import ALL_SCENES, Scene
from twinworld.package import SceneArtefacts, build_submission, submission_cloud

V1 = Path("experiments/2026-08-08-pipeline-v1")
V7 = Path("experiments/2026-08-11-pipeline-v7")
SLAB = Path("experiments/2026-08-27-slab-clouds")
for tier in ("safe", "push"):
    arts = []
    for dataset, sid in ALL_SCENES:
        s = Scene(dataset, sid)
        rgb = V7 / dataset / sid / "gs" / "rgb"
        if s.needs_semantics:
            d = np.load(SLAB / f"{sid}_{tier}_labeled.npz")
            xyz, labels = d["xyz"].astype(np.float64), d["labels"]
        else:
            xyz, _ = submission_cloud(s, V1 / dataset / sid / "mvs" / "fused_union.ply", voxel=0.04)
            labels = None
        arts.append(SceneArtefacts(scene=s, rgb_dir=rgb, xyz=xyz, labels=labels))
        print(f"  [v15{tier}] {s.key}: {len(xyz)/1e6:.1f}M", flush=True)
    rep = build_submission(arts, Path(f"submissions/v15{tier}-full.zip"), downsample=None)
    print(f"v15{tier}", "VALID" if rep.ok else rep.errors[:2], rep.stats.get("zip_mb"), flush=True)
