"""Cache surface normals for the packed Gold Coast clouds.

Normals are what the sheet build offsets along, and they do not depend on which
offset configuration the sweep selects, so they are computed once here and
reused. Fitted on the clean 10 cm surface cloud and inherited by the packed
cloud's points from their nearest surface point: the packed cloud carries a
20 cm interpolation band whose neighbourhoods are volumetric, and a normal
fitted inside that band points nowhere in particular.

Orientation is towards the camera rig, so a positive offset always means
"outward, towards the cameras" and a one-sided configuration is meaningful.

Writes {scene}_xl_normals.npz alongside the clouds.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from twinworld.dataio import Scene
from twinworld.package import submission_cloud

V1 = Path("experiments/2026-08-08-pipeline-v1")
XL = Path("experiments/2026-08-29-xl-clouds")
NORMAL_K = 16
CHUNK = 400_000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", default=["scene_011", "scene_012"])
    a = ap.parse_args()

    for sid in a.scenes:
        out = XL / f"{sid}_xl_normals.npz"
        if out.exists():
            print(f"[{sid}] cached, skipping", flush=True)
            continue
        s = Scene("gold_coast", sid)
        t0 = time.time()
        base, _ = submission_cloud(
            s, V1 / s.dataset / s.scene_id / "mvs" / "fused_gs_i.ply", voxel=0.10)
        rig = np.mean([f.center for f in s.train_frames], axis=0)
        print(f"[{sid}] base {len(base)/1e6:.1f}M pts, fitting normals", flush=True)

        tree = cKDTree(base)
        bn = np.empty((len(base), 3), np.float32)
        for i in range(0, len(base), CHUNK):
            j = min(i + CHUNK, len(base))
            _, idx = tree.query(base[i:j], k=NORMAL_K, workers=-1)
            nb = base[idx]
            P = nb - nb.mean(axis=1)[:, None, :]
            C = np.einsum("mki,mkj->mij", P, P) / NORMAL_K
            _, V = np.linalg.eigh(C)
            nrm = V[:, :, 0]
            flip = np.sign(np.einsum("mi,mi->m", rig - base[i:j], nrm))
            flip[flip == 0] = 1.0
            bn[i:j] = (nrm * flip[:, None]).astype(np.float32)
            if i % (CHUNK * 20) == 0:
                print(f"  {i/1e6:5.1f}M / {len(base)/1e6:.1f}M "
                      f"({time.time()-t0:.0f}s)", flush=True)

        xyz = np.load(XL / f"{sid}_xl_v18labeled.npz")["xyz"].astype(np.float64)
        print(f"[{sid}] inheriting onto {len(xyz)/1e6:.1f}M packed pts "
              f"({time.time()-t0:.0f}s)", flush=True)
        nrm = np.empty((len(xyz), 3), np.float32)
        for i in range(0, len(xyz), CHUNK * 5):
            j = min(i + CHUNK * 5, len(xyz))
            _, k1 = tree.query(xyz[i:j], k=1, workers=-1)
            nrm[i:j] = bn[k1]
        np.savez_compressed(out, normals=nrm)
        print(f"[{sid}] wrote {out.name} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
