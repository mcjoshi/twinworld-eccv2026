"""Raise facade coverage by rendering the splat model from tilted viewpoints.

The confusion diagnostic of 16 Aug reframed the semantic problem. Per-class
coverage on held-out scene_009:

    roof   0.751     wall   0.227     window 0.245     ground 0.424

Wall and window are 73 percent of the ground truth and are covered at barely a
quarter. Their IoU is bounded by those numbers regardless of the classifier -
and wall already converts 0.121 of its 0.227 ceiling, so the classifier is not
what is holding it back. Roofs are reconstructed well and facades badly, which
is exactly what a nadir-dominated drone capture produces.

`interpolated_frames` synthesises viewpoints between training cameras, which
preserves their nadir bias and therefore cannot fix this. This adds viewpoints
that are *rotated toward the horizon* from the training positions: the camera
stays where it was photographed from, so the model is not asked to invent a
viewpoint from an unobserved location, but it looks at facades squarely rather
than at grazing incidence. Facades are known to be visible in the training
imagery at about 41 degrees median incidence, so the information is present.

Readout is per-class coverage, which needs no classifier and is therefore cheap
to iterate on. If wall and window coverage do not move, nothing downstream is
worth running.

Adoption gate, fixed before running: wall AND window coverage must improve on
BOTH dev scenes, and the ceiling must rise, at a point cost that fits the
budget. Coverage gains that do not survive into achieved mIoU are reported as
such.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from twinworld.dataio import CLASS_NAMES, NUM_CLASSES, Frame, Scene, qvec2rotmat, rotmat2qvec
from twinworld.geometry.gs_depth import multiview_filter, render_depth, unproject
from twinworld.package import submission_cloud
from twinworld.render.trainer import TrainConfig
from twinworld.scoring import read_ply_xyz, voxel_downsample

V1 = Path("experiments/2026-08-08-pipeline-v1")
OUT = Path("experiments/2026-08-16-facade-coverage")
MATCH_R = 0.10
PACK_VOXEL = 0.10


def tilted_frames(frames, angles_deg, max_frames: int = 60):
    """Training viewpoints rotated toward the horizon about the camera's x axis.

    Position is unchanged, so this is a rotation-only extrapolation from a pose
    the scene was actually photographed from - much milder than inventing a new
    camera centre, and it points the principal ray at facades instead of roofs.
    """
    out = []
    step = max(1, len(frames) // max_frames)
    for fi, fr in enumerate(frames[::step]):
        for a in angles_deg:
            th = np.deg2rad(a)
            # rotate about camera x (right); +theta pitches the view up toward
            # the horizon in the OpenCV convention where +y is down, +z forward
            Rx = np.array([[1, 0, 0],
                           [0, np.cos(th), -np.sin(th)],
                           [0, np.sin(th), np.cos(th)]], dtype=np.float64)
            R_new = Rx @ fr.R
            tvec = -R_new @ fr.center
            out.append(Frame(-1, f"tilt_{fi}_{a}", rotmat2qvec(R_new), tvec,
                             fr.camera, None))
    return out


def per_class_coverage(sub_xyz, gt, gl):
    d, _ = cKDTree(sub_xyz).query(gt, k=1, distance_upper_bound=MATCH_R,
                                  workers=-1)
    cov_mask = np.isfinite(d)
    out = {}
    for c in range(NUM_CLASSES):
        m = gl == c
        out[CLASS_NAMES[c]] = round(float(cov_mask[m].mean()), 4) if m.any() else 0.0
    out["ceiling"] = round(float(np.mean([out[CLASS_NAMES[c]]
                                          for c in range(NUM_CLASSES)])), 4)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=2)
    ap.add_argument("--scenes", nargs="+", default=["gold_coast/scene_009",
                                                    "gold_coast/scene_010"])
    ap.add_argument("--angles", type=float, nargs="+", default=[20, 35, 50])
    ap.add_argument("--max-frames", type=int, default=40)
    ap.add_argument("--alpha-min", type=float, default=0.7)
    ap.add_argument("--min-views", type=int, default=2)
    ap.add_argument("--stride", type=int, default=2)
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    device = f"cuda:{a.gpu}"
    rows = []
    for key in a.scenes:
        s = Scene(*key.split("/"))
        ply = V1 / s.dataset / s.scene_id / "mvs" / "fused_gs_i.ply"
        splats = V1 / s.dataset / s.scene_id / "gs" / "splats.pt"
        if not splats.exists():
            print(f"[{s.key}] no splat model at {splats}, skipping", flush=True)
            continue

        gt, gl = read_ply_xyz(s.gt_ply, with_classification=True)
        base, _ = submission_cloud(s, ply, voxel=PACK_VOXEL)
        cov0 = per_class_coverage(base, gt, gl)
        print(f"\n[{s.key}] baseline {len(base)/1e6:.1f}M pts  "
              + "  ".join(f"{k} {v:.3f}" for k, v in cov0.items()), flush=True)

        t0 = time.time()
        cfg = TrainConfig()
        frames = tilted_frames(s.train_frames, a.angles, a.max_frames)
        print(f"[{s.key}] rendering {len(frames)} tilted views "
              f"(angles {a.angles})", flush=True)

        pts, counts = [], []
        for fr, depth, alpha, rgb in render_depth(splats, frames, cfg, device):
            # unproject returns (world_points, colours); counts must track the
            # per-view point totals for the multi-view agreement test
            w, _c = unproject(fr, depth.cpu().numpy(), alpha.cpu().numpy(),
                              rgb.cpu().numpy(), a.alpha_min, a.stride)
            pts.append(w)
            counts.append(len(w))
        if not pts or sum(counts) == 0:
            print(f"[{s.key}] no points recovered", flush=True)
            continue
        raw = np.concatenate(pts)
        print(f"[{s.key}] {len(raw)/1e6:.1f}M raw points from tilted views "
              f"({time.time()-t0:.0f}s)", flush=True)
        # multiview_filter returns a boolean mask, not the filtered points
        keep = multiview_filter(raw, counts, min_views=a.min_views)
        new = raw[keep]
        print(f"[{s.key}] {len(new)/1e6:.1f}M after {a.min_views}-view filter "
              f"({100*keep.mean():.1f}% kept)", flush=True)

        union = voxel_downsample(np.vstack([base, new]), PACK_VOXEL)
        cov1 = per_class_coverage(union, gt, gl)
        row = {"scene": s.key, "angles": a.angles,
               "baseline_points": int(len(base)), "union_points": int(len(union)),
               "coverage_before": cov0, "coverage_after": cov1,
               "delta": {k: round(cov1[k] - cov0[k], 4) for k in cov0},
               "seconds": round(time.time() - t0, 1)}
        rows.append(row)
        print(f"[{s.key}] union {len(union)/1e6:.1f}M pts  "
              + "  ".join(f"{k} {cov1[k]:.3f} ({cov1[k]-cov0[k]:+.3f})"
                          for k in cov0), flush=True)
        (OUT / "results.json").write_text(json.dumps(rows, indent=2))
        np.save(OUT / f"{s.scene_id}_tilted_points.npy",
                new.astype(np.float32))

    if rows:
        ok = all(r["delta"]["wall"] > 0 and r["delta"]["window"] > 0 for r in rows)
        print(f"\nwall AND window coverage improve on all scenes: {ok} -> "
              f"{'PROCEED to semantics' if ok else 'REJECT'}", flush=True)


if __name__ == "__main__":
    main()
