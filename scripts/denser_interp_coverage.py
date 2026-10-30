"""More interpolated viewpoints, aimed squarely at the coverage bottleneck.

The Gold Coast cloud is MVS unioned with splat-rendered depth at viewpoints
interpolated between adjacent training cameras. That union was adopted in
August and is what `fused_gs_i.ply` contains, built with per_pair=2 and
max_pairs=40 - roughly 80 synthetic views.

The 16 August diagnostic showed coverage is what caps the semantic term: wall
0.235 and window 0.253 against roof 0.763. Tilted viewpoints were rejected
(both tilt directions gave identical gains, so the model has no facade geometry
to reveal). Interpolation is different in kind: it stays inside the convex hull
of poses the scene was actually photographed from, which is where the splat
model is reliable, and it was already shown to help once.

The untested question is simply whether it was pushed far enough. This sweeps
the view budget upward and reports per-class coverage, which needs no classifier
and is therefore cheap.

Adoption gate, fixed before running: wall AND window coverage must improve on
BOTH dev scenes, and the point cost must leave the packaged cloud inside the
budget. Coverage that does not survive into achieved mIoU is reported as such.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from twinworld.dataio import CLASS_NAMES, NUM_CLASSES, Scene
from twinworld.geometry.gs_depth import (
    interpolated_frames, multiview_filter, render_depth, unproject,
)
from twinworld.package import submission_cloud
from twinworld.render.trainer import TrainConfig
from twinworld.scoring import read_ply_xyz, voxel_downsample

V1 = Path("experiments/2026-08-08-pipeline-v1")
OUT = Path("experiments/2026-08-19-denser-interp")
MATCH_R = 0.10
PACK_VOXEL = 0.10


def per_class_coverage(sub_xyz, gt, gl):
    d, _ = cKDTree(sub_xyz).query(gt, k=1, distance_upper_bound=MATCH_R, workers=-1)
    m = np.isfinite(d)
    out = {CLASS_NAMES[c]: (round(float(m[gl == c].mean()), 4) if (gl == c).any() else 0.0)
           for c in range(NUM_CLASSES)}
    out["ceiling"] = round(float(np.mean([out[CLASS_NAMES[c]]
                                          for c in range(NUM_CLASSES)])), 4)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=2)
    ap.add_argument("--scenes", nargs="+",
                    default=["gold_coast/scene_009", "gold_coast/scene_010"])
    ap.add_argument("--per-pair", type=int, default=5)
    ap.add_argument("--max-pairs", type=int, default=200)
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
            print(f"[{s.key}] no splat model, skipping", flush=True)
            continue

        # Final-phase scenes have no ground truth, so coverage cannot be
        # measured there. The points are still needed for the submission, so
        # render and save them and skip the measurement.
        has_gt = s.has_gt_cloud
        base, _ = submission_cloud(s, ply, voxel=PACK_VOXEL)
        if has_gt:
            gt, gl = read_ply_xyz(s.gt_ply, with_classification=True)
            cov0 = per_class_coverage(base, gt, gl)
            print(f"\n[{s.key}] baseline {len(base)/1e6:.1f}M  "
                  + "  ".join(f"{k} {v:.3f}" for k, v in cov0.items()), flush=True)
        else:
            gt = gl = None
            cov0 = None
            print(f"\n[{s.key}] baseline {len(base)/1e6:.1f}M (no GT: "
                  f"rendering points only)", flush=True)

        t0 = time.time()
        frames = interpolated_frames(s.train_frames, per_pair=a.per_pair,
                                     max_pairs=a.max_pairs)
        print(f"[{s.key}] rendering {len(frames)} interpolated views "
              f"(per_pair={a.per_pair}, max_pairs={a.max_pairs}; "
              f"production used 2/40)", flush=True)

        pts, cols, counts = [], [], []
        for fr, depth, alpha, rgb in render_depth(splats, frames, TrainConfig(),
                                                  device):
            w, c = unproject(fr, depth.cpu().numpy(), alpha.cpu().numpy(),
                             rgb.cpu().numpy(), a.alpha_min, a.stride)
            pts.append(w)
            cols.append(c)
            counts.append(len(w))
        if not pts or sum(counts) == 0:
            print(f"[{s.key}] no points recovered", flush=True)
            continue
        raw = np.concatenate(pts)
        raw_c = np.concatenate(cols)
        keep = multiview_filter(raw, counts, min_views=a.min_views)
        new, new_c = raw[keep], raw_c[keep]
        print(f"[{s.key}] {len(raw)/1e6:.1f}M raw -> {len(new)/1e6:.1f}M after "
              f"{a.min_views}-view filter ({100*keep.mean():.1f}% kept, "
              f"{time.time()-t0:.0f}s)", flush=True)

        union = voxel_downsample(np.vstack([base, new]), PACK_VOXEL)
        cov1 = per_class_coverage(union, gt, gl) if has_gt else None
        rows.append({"scene": s.key, "per_pair": a.per_pair,
                     "max_pairs": a.max_pairs, "views": len(frames),
                     "baseline_points": int(len(base)),
                     "union_points": int(len(union)),
                     "coverage_before": cov0, "coverage_after": cov1,
                     "delta": ({k: round(cov1[k] - cov0[k], 4) for k in cov0}
                               if has_gt else None),
                     "seconds": round(time.time() - t0, 1)})
        if has_gt:
            print(f"[{s.key}] union {len(union)/1e6:.1f}M  "
                  + "  ".join(f"{k} {cov1[k]:.3f} ({cov1[k]-cov0[k]:+.3f})"
                              for k in cov0), flush=True)
        else:
            print(f"[{s.key}] union {len(union)/1e6:.1f}M pts "
                  f"(+{(len(union)-len(base))/1e6:.1f}M)", flush=True)
        (OUT / "results.json").write_text(json.dumps(rows, indent=2))
        np.savez_compressed(OUT / f"{s.scene_id}_interp.npz",
                            xyz=new.astype(np.float32),
                            rgb=new_c.astype(np.uint8))

    if rows:
        scored = [r for r in rows if r["delta"] is not None]
        ok = bool(scored) and all(r["delta"]["wall"] > 0 and r["delta"]["window"] > 0
                                  for r in scored)
        print(f"\nwall AND window improve on all scenes: {ok} -> "
              f"{'PROCEED to semantics' if ok else 'REJECT'}", flush=True)


if __name__ == "__main__":
    main()
