"""Project the 3D ground truth into the training images to synthesise 2D masks.

Why this exists. Our semantic classifier reads 29 geometric features -
eigenvalue shape descriptors, verticality, plane offset, height above ground,
per-point colour. A window is coplanar with the wall it sits in: same plane,
same normal, same curvature, same height. There is no geometric signal that
separates them, so we are asking a geometry classifier to find a class defined by
appearance. Measured consequence: window IoU 0.061 and "other" 0.049, against
roof 0.443. Window and other are 19 percent of the ground truth but two of the
five classes, so they carry 40 percent of the mIoU - the metric averages classes,
not points.

In an image a window is unmistakable. The missing capability is a 2D semantic
segmenter, and the missing ingredient for training one is 2D annotations, which
the challenge does not provide.

But we can synthesise them. The 3D ground truth for scenes 009 and 010 carries
per-point class labels, and every training image has a known pose. Projecting the
labelled points into each image gives sparse but correct 2D supervision - the
labels exist already, just in the wrong dimensionality.

Occlusion matters: a point on the far wall must not paint a label onto the near
wall in front of it. Handled with a per-pixel depth buffer, keeping the nearest
point per pixel and rejecting anything behind it by more than `--z-tol`.

Output per image: a uint8 mask with class indices 0-4 and 255 for unlabelled,
which is the ignore index the training loss will skip.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from twinworld.dataio import CLASS_NAMES, NUM_CLASSES, Scene
from twinworld.scoring import read_ply_xyz

OUT_ROOT = Path("experiments/2026-08-24-2d-labels")
IGNORE = 255


def project(frame, xyz):
    """World points to pixel coordinates plus camera-space depth."""
    cam = xyz @ frame.R.T + frame.t
    z = cam[:, 2]
    front = z > 1e-6
    c = frame.camera
    u = np.full(len(xyz), -1.0)
    v = np.full(len(xyz), -1.0)
    u[front] = c.fx * cam[front, 0] / z[front] + c.cx
    v[front] = c.fy * cam[front, 1] / z[front] + c.cy
    return u, v, z, front


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+",
                    default=["gold_coast/scene_009", "gold_coast/scene_010"])
    ap.add_argument("--z-tol", type=float, default=0.30,
                    help="a point further than this behind the nearest point in "
                         "its pixel is treated as occluded and discarded")
    ap.add_argument("--dilate", type=int, default=2,
                    help="grow each labelled pixel by this radius. Projected "
                         "points are sparse in image space; a small dilation "
                         "turns speckle into trainable regions without crossing "
                         "class boundaries at these scales.")
    a = ap.parse_args()

    for key in a.scenes:
        s = Scene(*key.split("/"))
        if not s.has_gt_cloud:
            print(f"[{s.key}] no ground truth, skipping", flush=True)
            continue
        out = OUT_ROOT / s.dataset / s.scene_id
        out.mkdir(parents=True, exist_ok=True)

        gt, gl = read_ply_xyz(s.gt_ply, with_classification=True)
        keep = gl < NUM_CLASSES
        gt, gl = gt[keep], gl[keep].astype(np.uint8)
        print(f"\n[{s.key}] {len(gt)/1e6:.1f}M labelled GT points, "
              f"{len(s.train_frames)} training views", flush=True)

        tally = np.zeros(NUM_CLASSES, np.int64)
        for fi, fr in enumerate(s.train_frames):
            t0 = time.time()
            W, H = fr.camera.width, fr.camera.height
            u, v, z, front = project(fr, gt)
            px, py = np.round(u).astype(np.int64), np.round(v).astype(np.int64)
            ok = front & (px >= 0) & (px < W) & (py >= 0) & (py < H)
            if not ok.any():
                continue
            idx = np.flatnonzero(ok)
            flat = py[idx] * W + px[idx]
            zz = z[idx]

            # depth buffer: nearest point wins each pixel
            order = np.argsort(zz)
            flat_s, zz_s, idx_s = flat[order], zz[order], idx[order]
            first = np.ones(len(flat_s), bool)
            uniq, pos = np.unique(flat_s, return_index=True)
            nearest_z = np.full(W * H, np.inf, np.float32)
            nearest_z[uniq] = zz_s[pos]

            # keep points at or near the surface depth, drop occluded ones
            visible = zz_s <= nearest_z[flat_s] + a.z_tol
            flat_v, lab_v = flat_s[visible], gl[idx_s[visible]]

            mask = np.full(W * H, IGNORE, np.uint8)
            # later writes win; sort so nearest is written last
            o2 = np.argsort(-zz_s[visible])
            mask[flat_v[o2]] = lab_v[o2]
            mask = mask.reshape(H, W)

            if a.dilate > 0:
                from scipy.ndimage import grey_dilation
                lab_only = np.where(mask == IGNORE, 0, mask + 1)
                grown = grey_dilation(lab_only, size=(2 * a.dilate + 1,) * 2)
                mask = np.where(grown == 0, IGNORE, grown - 1).astype(np.uint8)

            np.save(out / f"{fr.stem}.npy", mask)
            lab_px = int((mask != IGNORE).sum())
            for c in range(NUM_CLASSES):
                tally[c] += int((mask == c).sum())
            print(f"  [{fr.stem}] {W}x{H}  labelled {100*lab_px/(W*H):5.1f}% "
                  f"({time.time()-t0:.0f}s)", flush=True)

        tot = max(tally.sum(), 1)
        print(f"[{s.key}] pixel class balance: "
              + "  ".join(f"{CLASS_NAMES[c]} {100*tally[c]/tot:.1f}%"
                          for c in range(NUM_CLASSES)), flush=True)


if __name__ == "__main__":
    main()
