"""Where is geometry recall lost: unobserved surfaces, or filtered-away ones?

Recall at 10 cm sits around 0.45 while precision is 0.72, so recall is what
bounds the F-score. This splits the missed ground-truth points by whether the
training views could have seen them at all:

  in-frustum    inside at least one training camera's image bounds
  front-facing  its local surface normal points back towards a camera that sees it
  unoccluded    nothing in our own reconstruction sits between it and that camera

If recall over the observable subset is already high, MVS is doing its job and
the ceiling is set by the capture geometry, so effort belongs in rendering and
semantics. If it is low, denser or looser MVS is still worth pushing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from twinworld.dataio import Scene
from twinworld.scoring import BBOX_PADDING, crop_to_bbox, read_ply_xyz, voxel_downsample


def surface_normals(xyz: np.ndarray, k: int = 24, chunk: int = 200_000) -> np.ndarray:
    from scipy.spatial import cKDTree

    tree = cKDTree(xyz)
    n = np.empty_like(xyz)
    for a in range(0, len(xyz), chunk):
        b = min(a + chunk, len(xyz))
        _, idx = tree.query(xyz[a:b], k=k, workers=-1)
        P = xyz[idx]
        P = P - P.mean(axis=1, keepdims=True)
        C = np.einsum("mki,mkj->mij", P, P) / k
        _, V = np.linalg.eigh(C)
        n[a:b] = V[:, :, 0]
    return n


def analyse(scene: Scene, pred_ply: Path, voxel: float = 0.05) -> dict:
    gt_full, _ = read_ply_xyz(scene.gt_ply)
    pred_full, _ = read_ply_xyz(pred_ply)

    lo, hi = gt_full.min(0) - BBOX_PADDING, gt_full.max(0) + BBOX_PADDING
    gt = voxel_downsample(crop_to_bbox(gt_full, lo, hi), voxel)
    pred = voxel_downsample(crop_to_bbox(pred_full, lo, hi), voxel)

    from scipy.spatial import cKDTree

    d_gt, _ = cKDTree(pred).query(gt, workers=-1)
    matched = d_gt < 0.10

    normals = surface_normals(gt)
    frames = scene.train_frames

    in_frustum = np.zeros(len(gt), bool)
    front_facing = np.zeros(len(gt), bool)
    unoccluded = np.zeros(len(gt), bool)

    pred_tree = cKDTree(pred)
    for fr in frames:
        cam = gt @ fr.R.T + fr.t
        z = cam[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = fr.camera.fx * cam[:, 0] / z + fr.camera.cx
            v = fr.camera.fy * cam[:, 1] / z + fr.camera.cy
        vis = (z > 0) & (u >= 0) & (u < fr.camera.width) & (v >= 0) & (v < fr.camera.height)
        in_frustum |= vis

        ray = gt - fr.center
        ray = ray / np.maximum(np.linalg.norm(ray, axis=1, keepdims=True), 1e-9)
        facing = vis & (np.abs(np.einsum("ij,ij->i", ray, normals)) > 0.20)
        front_facing |= facing

        # occlusion: sample the segment from the point towards the camera and ask
        # whether our own reconstruction occupies any of it
        cand = np.where(facing & ~unoccluded)[0]
        if len(cand) == 0:
            continue
        blocked = np.zeros(len(cand), bool)
        for t in (0.5, 1.0, 2.0):  # metres in front of the surface
            probe = gt[cand] + ray[cand] * -t
            dd, _ = pred_tree.query(probe, workers=-1)
            blocked |= dd < voxel
        unoccluded[cand[~blocked]] = True

    def frac(mask):
        return float(mask.mean())

    def rec(mask):
        return float(matched[mask].mean()) if mask.any() else None

    return {
        "scene": scene.key,
        "n_gt_voxels": int(len(gt)),
        "n_pred_voxels": int(len(pred)),
        "recall_10cm_all": frac(matched),
        "gt_in_frustum_frac": frac(in_frustum),
        "gt_front_facing_frac": frac(front_facing),
        "gt_observable_frac": frac(unoccluded),
        "recall_in_frustum": rec(in_frustum),
        "recall_front_facing": rec(front_facing),
        "recall_observable": rec(unoccluded),
        "recall_unobservable": rec(~unoccluded),
        "voxel": voxel,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="auto")
    ap.add_argument("--exp", default="experiments/2026-08-08-geometry-bakeoff")
    ap.add_argument("--scenes", nargs="*", default=["tum/scene_000"])
    args = ap.parse_args()

    out = []
    for key in args.scenes:
        s = Scene(*key.split("/"))
        ply = Path(args.exp) / args.variant / s.dataset / s.scene_id / "fused.ply"
        if not ply.exists():
            print(f"skip {key}: {ply} missing")
            continue
        r = analyse(s, ply)
        out.append(r)
        print(json.dumps(r, indent=2), flush=True)

    p = Path(args.exp) / f"recall_diagnostic_{args.variant}.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
