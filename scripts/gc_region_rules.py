"""A graded Gold Coast cloud: fine where it can score, coarse everywhere else.

Established by the extent diagnostic: Gold Coast ground truth occupies about
2 percent of the area we reconstruct, and the semantic ceiling is limited by
voxel quantisation (0.42 at 10 cm, 0.26 at 20 cm) rather than by point count,
because a 20 cm voxel displaces points by up to 17 cm against a 10 cm match
radius.

The tempting move is to crop to the ground-truth bounding box. That is test-set
information and would collapse on the final scenes, so it is not evaluated here
even as a baseline - it is only reported as an unreachable oracle.

What is evaluated are region rules computable from inputs that exist for every
scene including the withheld ones: the provided train and test camera poses.
Inside the region the cloud keeps a fine voxel; outside it keeps today's 20 cm.
Points are never deleted, so a wrong region degrades to the current submission
rather than losing coverage - the downside is bounded at today's score.

Adoption gate, fixed before running: on BOTH dev scenes the rule must
  (a) contain at least 99 percent of ground-truth points, and
  (b) beat the 20 cm-everywhere ceiling,
at a point count no larger than the current profile.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from twinworld.dataio import Scene
from twinworld.package import submission_cloud
from twinworld.scoring import read_ply_xyz, score_semantics, voxel_downsample

V1 = Path("experiments/2026-08-08-pipeline-v1")
OUT = Path("experiments/2026-08-13-gc-region")
FINE = 0.04
COARSE = 0.20


def region_masks(s: Scene, xyz: np.ndarray) -> dict:
    """Candidate regions, all computable without ground truth."""
    out = {}
    out["test_frustum"] = s.test_frustum_mask(xyz)
    out["test_frustum_m64"] = s.test_frustum_mask(xyz, margin_px=64.0)
    out["train_frustum"] = s.train_frustum_mask(xyz)

    # ball around the test cameras' mean look-at point, at a radius set by the
    # spread of the test camera centres themselves (no ground truth involved)
    cams = np.array([fr.center for fr in s.test_frames])
    fwd = np.array([fr.R.T @ np.array([0.0, 0.0, 1.0]) for fr in s.test_frames])
    # approximate look-at: median scene depth along each optical axis
    centre = cams.mean(0)
    span = float(np.linalg.norm(xyz.max(0) - xyz.min(0)))
    for frac, name in ((0.15, "ball_p15"), (0.25, "ball_p25")):
        r = span * frac
        aim = centre + fwd.mean(0) * r * 0.5
        out[name] = np.linalg.norm(xyz - aim, axis=1) < r
    return out


def graded(xyz: np.ndarray, mask: np.ndarray) -> np.ndarray:
    fine = xyz[mask]
    coarse = voxel_downsample(xyz[~mask], COARSE) if (~mask).any() else np.empty((0, 3))
    return np.vstack([fine, coarse]) if len(coarse) else fine


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for sid in ("scene_009", "scene_010"):
        s = Scene("gold_coast", sid)
        ply = V1 / s.dataset / s.scene_id / "mvs" / "fused_gs_i.ply"
        gt, gl = read_ply_xyz(s.gt_ply, with_classification=True)

        t0 = time.time()
        xyz, _ = submission_cloud(s, ply, voxel=FINE)
        print(f"\n[{s.key}] fine cloud {len(xyz)/1e6:.1f}M pts at {FINE} m "
              f"({time.time()-t0:.0f}s)", flush=True)

        base = voxel_downsample(xyz, COARSE)
        c0 = score_semantics(base, np.zeros(len(base), np.uint8), gt, gl)
        print(f"  baseline 20cm-everywhere: {len(base)/1e6:6.2f}M pts  "
              f"ceiling {c0.miou_ceiling:.4f}", flush=True)
        rows.append({"scene": s.key, "rule": "baseline_20cm",
                     "points": int(len(base)),
                     "ceiling": round(float(c0.miou_ceiling), 4),
                     "gt_inside": 1.0})

        masks = region_masks(s, xyz)
        gt_masks = region_masks(s, gt)  # computed once, not per rule
        for name, mask in masks.items():
            t1 = time.time()
            g = graded(xyz, mask)
            c = score_semantics(g, np.zeros(len(g), np.uint8), gt, gl)
            # safety: what fraction of GT lies inside the region at all?
            gt_in = float(gt_masks[name].mean())
            rows.append({"scene": s.key, "rule": name,
                         "points": int(len(g)),
                         "frac_fine": round(float(mask.mean()), 4),
                         "ceiling": round(float(c.miou_ceiling), 4),
                         "gt_inside": round(gt_in, 4),
                         "seconds": round(time.time() - t1, 1)})
            print(f"  {name:18s} fine={mask.mean()*100:5.1f}%  "
                  f"N={len(g)/1e6:6.2f}M  ceiling {c.miou_ceiling:.4f}  "
                  f"GT inside region {gt_in*100:5.1f}%", flush=True)
            (OUT / "results.json").write_text(json.dumps(rows, indent=2))

    (OUT / "results.json").write_text(json.dumps(rows, indent=2))
    print("\ndone", flush=True)




if __name__ == "__main__":
    main()
