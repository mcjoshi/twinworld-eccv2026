"""Are the facades missing, or merely offset?

Wall coverage is 0.23: 77 percent of wall ground-truth points have no submitted
point within the scorer's 10 cm radius. Two very different worlds produce that
number.

  missing   we never reconstructed those surfaces. Distances from ground truth
            to our cloud are large and unstructured. Only better capture or
            better reconstruction helps.

  offset    we reconstructed them, but our surface sits systematically beside
            the ground-truth surface - a registration bias, a depth bias, or the
            LOD3 model's idealised wall plane sitting inside our textured
            facade. Distances then pile up in a narrow band just outside 10 cm,
            and a small dilation or offset correction recovers enormous coverage
            for free, because Gold Coast has no precision penalty.

This measures the distance from every ground-truth point to the nearest
submitted point, per class, and reports the distribution plus the coverage that
each candidate radius would yield. A sharp mode between 0.1 and 0.5 m means
offset; a long flat tail means missing.

Diagnostic only - nothing is adopted here.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from twinworld.dataio import CLASS_NAMES, NUM_CLASSES, Scene
from twinworld.package import submission_cloud
from twinworld.scoring import read_ply_xyz

V1 = Path("experiments/2026-08-08-pipeline-v1")
OUT = Path("experiments/2026-08-20-gt-distance")
PACK_VOXEL = 0.10
RADII = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.00, 2.00]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    out = {}
    for sid in ("scene_009", "scene_010"):
        s = Scene("gold_coast", sid)
        ply = V1 / s.dataset / s.scene_id / "mvs" / "fused_gs_i.ply"
        sub, _ = submission_cloud(s, ply, voxel=PACK_VOXEL)
        gt, gl = read_ply_xyz(s.gt_ply, with_classification=True)
        tree = cKDTree(sub)
        d, _ = tree.query(gt, k=1, workers=-1)

        rec = {}
        print(f"\n=== {s.key} ({len(gt)/1e6:.1f}M GT pts vs {len(sub)/1e6:.1f}M "
              f"submitted) ===", flush=True)
        header = f"{'class':8s} {'share':>7} " + "".join(f"{r:>7}" for r in RADII)
        print(header + f"  {'p25':>7}{'p50':>7}{'p75':>7}")
        for c in range(NUM_CLASSES):
            m = gl == c
            if not m.any():
                continue
            dc = d[m]
            cov = {str(r): round(float((dc <= r).mean()), 4) for r in RADII}
            q = np.percentile(dc, [25, 50, 75]).round(3)
            rec[CLASS_NAMES[c]] = {"share": round(float(m.mean()), 4),
                                   "coverage_at_radius": cov,
                                   "p25": float(q[0]), "p50": float(q[1]),
                                   "p75": float(q[2])}
            print(f"{CLASS_NAMES[c]:8s} {m.mean():7.3f} "
                  + "".join(f"{cov[str(r)]:7.3f}" for r in RADII)
                  + f"  {q[0]:7.3f}{q[1]:7.3f}{q[2]:7.3f}")

        # what the ceiling would be at each radius, if the metric allowed it
        print("\nmean-over-class coverage (= mIoU ceiling) at each radius:")
        for r in RADII:
            cs = [rec[CLASS_NAMES[c]]["coverage_at_radius"][str(r)]
                  for c in range(NUM_CLASSES) if CLASS_NAMES[c] in rec]
            print(f"  r={r:<5} ceiling {np.mean(cs):.4f}")
        out[s.key] = rec
        (OUT / "results.json").write_text(json.dumps(out, indent=2))

    print("\nreading: a jump in coverage between r=0.10 and r=0.20-0.30 means the "
          "surfaces exist but sit outside the match radius -> dilation or offset "
          "correction recovers them. A flat profile means they are absent.",
          flush=True)


if __name__ == "__main__":
    main()
