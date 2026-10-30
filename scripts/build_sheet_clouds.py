"""Materialise the sheeted Gold Coast clouds selected by the sweep.

Emits, for every parent point, a copy at each requested offset along the parent's
surface normal, carrying the parent's label. This is the operation ranked in
scripts/sheet_probe.py, reproduced exactly so that the measured configuration is
the one that ships.

Two rules this build obeys, both of which have teeth:

Only scenes 011 and 012 are worth sheeting. They are the Gold Coast scenes the
final phase scores; 009 and 010 are decimated to a 1 m voxel by the packaging
probe and contribute nothing but bytes.

Nothing may voxel-downsample the result. Every offset is smaller than the 12 cm
parent voxel, so a downsample applied after sheeting would merge each sheet back
into its parent and silently delete the entire gain. To shrink the payload,
coarsen the parents with --parent-voxel *before* sheeting, which is what the
budget-matched configurations in the sweep measure.

Writes {scene}_{tag}.npz with xyz and labels.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from twinworld.dataio import CLASS_NAMES, NUM_CLASSES
from twinworld.scoring import voxel_downsample_indices

XL = Path("experiments/2026-08-29-xl-clouds")
OUT = Path("experiments/2026-08-30-sheets")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offsets", type=float, nargs="+", required=True,
                    help="metres along the outward normal, include 0.0 to keep parents")
    ap.add_argument("--parent-voxel", type=float, default=0.0,
                    help="coarsen parents before sheeting; 0 keeps the packed cloud")
    ap.add_argument("--scenes", nargs="+", default=["scene_011", "scene_012"])
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    total = 0
    for sid in a.scenes:
        t0 = time.time()
        d = np.load(XL / f"{sid}_xl_v18labeled.npz")
        xyz = d["xyz"].astype(np.float32)
        labels = d["labels"]
        nrm = np.load(XL / f"{sid}_xl_normals.npz")["normals"]
        if len(nrm) != len(xyz):
            raise SystemExit(f"{sid}: {len(nrm)} normals for {len(xyz)} points")

        if a.parent_voxel > 0:
            keep = voxel_downsample_indices(xyz.astype(np.float64), a.parent_voxel)
            xyz, labels, nrm = xyz[keep], labels[keep], nrm[keep]
            print(f"[{sid}] parents coarsened to {a.parent_voxel:.2f} m: "
                  f"{len(xyz)/1e6:.1f}M", flush=True)

        parts = [(xyz + np.float32(o) * nrm) if o else xyz for o in a.offsets]
        out_xyz = np.concatenate(parts, axis=0)
        out_lab = np.tile(labels, len(a.offsets))
        del parts

        p = OUT / f"{sid}_{a.tag}.npz"
        np.savez(p, xyz=out_xyz, labels=out_lab)
        total += len(out_xyz)
        share = {CLASS_NAMES[c]: round(float((out_lab == c).mean()), 3)
                 for c in range(NUM_CLASSES)}
        print(f"[{sid}] {len(xyz)/1e6:.1f}M parents x {len(a.offsets)} -> "
              f"{len(out_xyz)/1e6:.1f}M pts  {share}  "
              f"({time.time()-t0:.0f}s)", flush=True)

    print(f"\n{a.tag}: {total/1e6:.1f}M scored Gold Coast points "
          f"(TUM adds 27.9M)", flush=True)


if __name__ == "__main__":
    main()
