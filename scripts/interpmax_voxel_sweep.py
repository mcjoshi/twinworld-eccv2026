"""How much of the interp-max ceiling survives voxel-trimming to budget?

The interp-max clouds reach ceilings 0.666 and 0.784 but cost 191 M and 155 M
points per scene at the 10 cm voxel. The budget is ~90-100 M total. Coarser
voxels shed points but push survivors away from ground truth - at 20 cm the
old clouds' ceiling collapsed. This measures the actual curve so the packaging
choice is a measurement, not a guess.
"""
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from twinworld.dataio import CLASS_NAMES, NUM_CLASSES, Scene
from twinworld.package import submission_cloud
from twinworld.geometry.densify import robust_core
from twinworld.scoring import read_ply_xyz, voxel_downsample_indices

OUT = Path("experiments/2026-08-26-interpmax-voxel"); OUT.mkdir(parents=True, exist_ok=True)
rows = []
for sid in ("scene_009", "scene_010"):
    s = Scene("gold_coast", sid)
    base, _ = submission_cloud(s, Path(f"experiments/2026-08-08-pipeline-v1/gold_coast/{sid}/mvs/fused_gs_i.ply"), voxel=0.10)
    nx = np.load(f"experiments/2026-08-26-interp-max/{sid}_interp.npz")["xyz"].astype(np.float64)
    nx = nx[robust_core(nx, s.train_frames)]
    union = np.vstack([base, nx])
    gt, gl = read_ply_xyz(s.gt_ply, with_classification=True)
    for v in (0.10, 0.12, 0.14, 0.16, 0.20):
        sub = union[voxel_downsample_indices(union, v)]
        d, _ = cKDTree(sub).query(gt, k=1, distance_upper_bound=0.10, workers=-1)
        cov = np.isfinite(d)
        pc = {CLASS_NAMES[c]: round(float(cov[gl==c].mean()),4) for c in range(NUM_CLASSES) if (gl==c).any()}
        ceil = float(np.mean(list(pc.values())))
        rows.append({"scene": s.key, "voxel": v, "points": int(len(sub)), "ceiling": round(ceil,4), "per_class": pc})
        print(f"[{s.key} v={v}] {len(sub)/1e6:6.1f}M pts  ceiling {ceil:.4f}  " + " ".join(f"{k[:4]} {x:.2f}" for k,x in pc.items()), flush=True)
        (OUT/"results.json").write_text(json.dumps(rows, indent=2))
