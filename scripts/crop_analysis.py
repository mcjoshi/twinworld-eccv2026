"""How much of the submitted cloud can be dropped without losing score?

At 2x resolution the Gold Coast clouds are 23-31 M points each and TUM 7-9 M, so
a full submission is of the order of 2 GB. Almost all of it is outside the
ground-truth bounding box and is discarded by the scorer before anything is
computed, but the bounding box is unknown for the final-phase scenes, so the
crop has to be defined by something we always have: the camera geometry.

This measures, on the development scenes where the ground truth is available,
how much size each candidate rule saves and how much recall it costs:

  train_frustum   inside at least one training view (the current rule)
  test_frustum    inside at least one held-out view, which is where the ground
                  truth was cropped around
  test_margin     the same with a pixel margin, to buy back the edges

A rule is only usable if it costs almost no recall on every development scene,
because on the final scenes there is no way to check.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from twinworld.dataio import Scene, dev_scenes
from twinworld.scoring import BBOX_PADDING, read_ply_xyz, score_geometry, voxel_downsample

BYTES_PER_POINT = {"tum": 12, "gold_coast": 13}  # 3x float32 (+ uint8 class)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="experiments/2026-08-08-pipeline-v1")
    ap.add_argument("--out", default="experiments/2026-08-08-crop-analysis")
    ap.add_argument("--voxel", type=float, default=0.02)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []

    for s in dev_scenes():
        ply = Path(args.run) / s.dataset / s.scene_id / "mvs" / "fused.ply"
        if not ply.exists():
            print(f"skip {s.key}: no cloud")
            continue
        xyz, _ = read_ply_xyz(ply)
        gt, _ = read_ply_xyz(s.gt_ply)
        lo, hi = gt.min(0) - BBOX_PADDING, gt.max(0) + BBOX_PADDING

        rules = {
            "none": np.ones(len(xyz), bool),
            "train_frustum": s.train_frustum_mask(xyz),
            "test_frustum": s.test_frustum_mask(xyz),
            "test_margin200": s.test_frustum_mask(xyz, margin_px=200),
            "gt_bbox_oracle": np.all((xyz >= lo) & (xyz <= hi), axis=1),
        }
        base_f = None
        for name, m in rules.items():
            kept = voxel_downsample(xyz[m], args.voxel)
            r = score_geometry(xyz[m], gt, s.key) if s.dataset == "tum" else None
            mb = len(kept) * BYTES_PER_POINT[s.dataset] / 1e6
            f = r.fscore if r else None
            if name == "none":
                base_f = f
            row = {"scene": s.key, "rule": name, "kept_frac": float(m.mean()),
                   "points_after_voxel": int(len(kept)), "approx_mb": round(mb, 1),
                   "fscore": f, "delta_fscore": (None if f is None or base_f is None
                                                 else round(f - base_f, 4))}
            rows.append(row)
            print(f"  {s.key:24s} {name:16s} keep {m.mean()*100:5.1f}%  "
                  f"{len(kept):9d} pts  ~{mb:7.1f} MB"
                  + (f"  F={f:.4f} ({row['delta_fscore']:+.4f})" if f is not None else ""),
                  flush=True)
        (out / "crop_analysis.json").write_text(json.dumps(rows, indent=2))

    (out / "crop_analysis.json").write_text(json.dumps(rows, indent=2))
    print("\ntotal approximate submission size by rule (development scenes only):")
    for name in ["none", "train_frustum", "test_frustum", "test_margin200", "gt_bbox_oracle"]:
        sel = [r for r in rows if r["rule"] == name]
        if sel:
            d = [r["delta_fscore"] for r in sel if r["delta_fscore"] is not None]
            print(f"  {name:16s} {sum(r['approx_mb'] for r in sel):8.1f} MB"
                  + (f"   worst F-score change {min(d):+.4f}" if d else ""))


if __name__ == "__main__":
    main()
