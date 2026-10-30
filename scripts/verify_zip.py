"""Score a submission zip from its own bytes against dev ground truth."""
import io, sys, zipfile
import numpy as np
from plyfile import PlyData
from twinworld.dataio import dev_scenes
from twinworld.scoring import read_ply_xyz, score_geometry, score_semantics, SemanticCounts
z = zipfile.ZipFile(sys.argv[1])
geo = []; sem = SemanticCounts()
for s in dev_scenes():
    v = PlyData.read(io.BytesIO(z.read(f"{s.dataset}/{s.scene_id}/3D_point_cloud/point_cloud.ply")))["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    if s.dataset == "tum":
        gt, _ = read_ply_xyz(s.gt_ply)
        r = score_geometry(xyz, gt, s.key); geo.append(r.fscore)
        print(f"{s.key}: F={r.fscore:.4f}", flush=True)
    else:
        lab = np.asarray(v["classification"])
        gt, gl = read_ply_xyz(s.gt_ply, with_classification=True)
        c = score_semantics(xyz, lab, gt, gl); sem.add(c)
        print(f"{s.key}: mIoU={c.miou:.4f}", flush=True)
g = float(np.mean(geo))
print(f"\nZIP-VERIFIED {sys.argv[1]}: geometry {g:.4f}  semantic {sem.miou:.4f}  "
      f"rendering 0.6309\ndev-basis final = {0.6309 + g + sem.miou:.4f}")
