"""Build the v12 Gold Coast clouds: outward offset sheets on a thinned base.

Gated on 23 Aug as off015_out_thin025: LOSO 0.1915 against v11's plain 0.1798,
both folds improving, at a point multiplier of 1.06 - the first offset
configuration that pays for itself. Every earlier variant scored higher (up to
0.2319) and needed 1.4x to 2.9x the points, which does not fit a budget where
108.6 M scored points failed on the server.

Why this configuration is cheap. A sheet is emitted per surviving base point, so
thinning the base halves the sheets too and the ratio barely moves - that is why
`--facade-thin` alone failed. The lever that actually reduces sheets-per-point is
emitting ONE side rather than two. Outward-only at thin=0.25 lands at 1.06x.

Applies to all four Gold Coast scenes: 009/010 so the result can be verified
against ground truth, and 011/012 because those are what the final phase scores.
Labels come from the existing v7 semantic run, transferred by k=5 majority vote -
the propagation adopted on 19 August.

Writes fused_gs_i_v12.npz per scene (xyz + labels), consumed by the packaging
stage.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from twinworld.dataio import NUM_CLASSES, Scene
from twinworld.package import submission_cloud
from twinworld.scoring import voxel_downsample_indices
from twinworld.semantics.features import estimate_up

V1 = Path("experiments/2026-08-08-pipeline-v1")
V7 = Path("experiments/2026-08-11-pipeline-v7")
PACK_VOXEL = 0.10
OFFSET = 0.15
THIN = 0.25
VERT = 0.5


def normals(xyz, k=24):
    tree = cKDTree(xyz)
    n = np.empty_like(xyz)
    for a in range(0, len(xyz), 400_000):
        b = min(a + 400_000, len(xyz))
        _, idx = tree.query(xyz[a:b], k=k, workers=-1)
        nb = xyz[idx]
        nb = nb - nb.mean(axis=1, keepdims=True)
        cov = np.einsum("mki,mkj->mij", nb, nb) / k
        _, v = np.linalg.eigh(cov)
        n[a:b] = v[:, :, 0]
    return n


def propagate_k(thin_xyz, thin_lab, full_xyz, k=5):
    _, idx = cKDTree(thin_xyz).query(full_xyz, k=k, workers=-1)
    lab = thin_lab[idx]
    votes = np.zeros((len(full_xyz), NUM_CLASSES), np.float32)
    for c in range(NUM_CLASSES):
        votes[:, c] = (lab == c).sum(axis=1)
    return np.argmax(votes, axis=1).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+",
                    default=[f"gold_coast/scene_{i}" for i in
                             ("009", "010", "011", "012")])
    ap.add_argument("--offset", type=float, default=OFFSET)
    ap.add_argument("--thin", type=float, default=THIN)
    a = ap.parse_args()

    for key in a.scenes:
        s = Scene(*key.split("/"))
        t0 = time.time()
        ply = V1 / s.dataset / s.scene_id / "mvs" / "fused_gs_i.ply"
        out = V1 / s.dataset / s.scene_id / "mvs" / "fused_gs_i_v12.npz"

        base, _ = submission_cloud(s, ply, voxel=PACK_VOXEL)
        nrm = normals(base)
        rig = np.stack([f.center for f in s.train_frames]).mean(axis=0)
        flip = np.einsum("ij,ij->i", nrm, base - rig) < 0
        nrm[flip] *= -1.0                      # consistent outward orientation
        up = estimate_up(s.train_frames)
        vertical = np.abs(nrm @ up) < VERT

        rng = np.random.default_rng(0)
        drop = vertical & (rng.random(len(base)) > a.thin)
        kept = base[~drop]
        sheet = base[vertical] + nrm[vertical] * a.offset   # outward only
        full = np.vstack([kept, sheet])
        full = full[voxel_downsample_indices(full, PACK_VOXEL)]

        # labels from the v7 semantic run, carried by the adopted k=5 vote
        lab_npz = V7 / s.dataset / s.scene_id / "semantics" / "labels.npz"
        d = np.load(lab_npz)
        lab = propagate_k(d["xyz"].astype(np.float64), d["labels"], full)

        np.savez_compressed(out, xyz=full.astype(np.float32), labels=lab)
        print(f"[{s.key}] base {len(base)/1e6:.1f}M -> v12 {len(full)/1e6:.1f}M "
              f"({len(full)/len(base):.2f}x)  vertical {vertical.mean()*100:.0f}%  "
              f"-> {out.name}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
