"""Score a materialised sheeted cloud with the exact scorer.

The sweep evaluates offsets analytically from one k-nearest-neighbour query,
which is exact only while the winning parent lies inside the candidate set. When
the candidate cap binds, coverage stays right but the winning *label* can be
wrong, so the ranking is trustworthy and the absolute number is not.

This closes that gap for the selected configuration only: it builds the sheets
for real, hands the full cloud to score_semantics, and prints the number the
official scorer would produce on a scene we hold ground truth for. Expensive
enough to be worth running once, cheap enough that shipping without it would be
careless.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from twinworld.dataio import CLASS_NAMES, NUM_CLASSES, Scene
from twinworld.scoring import read_ply_xyz, score_semantics, voxel_downsample_indices

XL = Path("experiments/2026-08-29-xl-clouds")
OUT = Path("experiments/2026-08-30-sheets")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offsets", type=float, nargs="+", required=True)
    ap.add_argument("--parent-voxel", type=float, default=0.0)
    ap.add_argument("--scenes", nargs="+", default=["scene_009", "scene_010"])
    ap.add_argument("--tag", default="exact")
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    inter = np.zeros(NUM_CLASSES, np.int64)
    union = np.zeros(NUM_CLASSES, np.int64)
    rows = {}
    for sid in a.scenes:
        s = Scene("gold_coast", sid)
        t0 = time.time()
        d = np.load(XL / f"{sid}_xl_v18labeled.npz")
        xyz = d["xyz"].astype(np.float32)
        labels = d["labels"]
        nrm = np.load(XL / f"{sid}_xl_normals.npz")["normals"]
        if a.parent_voxel > 0:
            keep = voxel_downsample_indices(xyz.astype(np.float64), a.parent_voxel)
            xyz, labels, nrm = xyz[keep], labels[keep], nrm[keep]

        pts = np.concatenate(
            [(xyz + np.float32(o) * nrm) if o else xyz for o in a.offsets], axis=0)
        lab = np.tile(labels, len(a.offsets))
        print(f"[{s.key}] materialised {len(pts)/1e6:.1f}M pts "
              f"({time.time()-t0:.0f}s), scoring", flush=True)

        gt, gl = read_ply_xyz(s.gt_ply, with_classification=True)
        c = score_semantics(pts.astype(np.float64), lab, gt, gl)
        iou = c.per_class_iou
        inter += np.asarray(c.intersection, np.int64)
        union += np.asarray(c.union, np.int64)
        rows[s.key] = {"miou": round(float(c.miou), 4),
                       # label-free: the best any labelling of this cloud could
                       # reach, and the part that transfers to 011 and 012
                       "miou_ceiling": round(float(c.miou_ceiling), 4),
                       "matched_frac": round(1 - c.n_unmatched / max(c.n_total, 1), 4),
                       "points_millions": round(len(pts) / 1e6, 1),
                       "per_class": {CLASS_NAMES[i]: round(float(iou[i]), 4)
                                     for i in range(NUM_CLASSES)}}
        print(f"[{s.key}] ceiling {c.miou_ceiling:.4f}  matched "
              f"{1 - c.n_unmatched / max(c.n_total, 1):.4f}", flush=True)
        print(f"[{s.key}] exact mIoU {c.miou:.4f}  "
              + "  ".join(f"{CLASS_NAMES[i]} {iou[i]:.3f}"
                          for i in range(NUM_CLASSES))
              + f"  ({time.time()-t0:.0f}s)", flush=True)
        del pts, lab, xyz, nrm

    pooled = float(np.mean(np.where(union > 0, inter / np.maximum(union, 1), 0.0)))
    print(f"\npooled exact mIoU {pooled:.4f} for offsets {a.offsets} "
          f"parent_voxel {a.parent_voxel}", flush=True)
    (OUT / f"exact_{a.tag}.json").write_text(json.dumps(
        {"offsets": a.offsets, "parent_voxel": a.parent_voxel,
         "pooled_miou": round(pooled, 4), "scenes": rows}, indent=2))


if __name__ == "__main__":
    main()
