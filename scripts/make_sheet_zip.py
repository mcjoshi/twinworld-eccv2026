"""Package a sheeted submission: sheeted clouds on the scored Gold Coast scenes.

Everything except Gold Coast 011 and 012 is byte-identical to v18, which scored
1.4529 official. TUM geometry and every render are untouched, so the rendering
and F-score terms cannot move and any change in the final score is the semantic
term alone. That makes the upload a clean read on the sheet operation.

Scenes 009 and 010 keep their v18 clouds; the packaging probe decimates them to a
1 m voxel afterwards because the final phase does not score them.

downsample=None is mandatory and not incidental: the sheet offsets are smaller
than the parent voxel, so any downsample here would merge the sheets back into
their parents and quietly ship v18 again.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from twinworld.dataio import ALL_SCENES, GC_FINAL, Scene
from twinworld.package import SceneArtefacts, build_submission, submission_cloud

V1 = Path("experiments/2026-08-08-pipeline-v1")
V7 = Path("experiments/2026-08-11-pipeline-v7")
XL = Path("experiments/2026-08-29-xl-clouds")
SHEET = Path("experiments/2026-08-30-sheets")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="sheet tag, e.g. s07")
    ap.add_argument("--out", required=True, help="output zip path")
    a = ap.parse_args()

    arts, scored = [], 0
    for dataset, sid in ALL_SCENES:
        s = Scene(dataset, sid)
        rgb = V7 / dataset / sid / "gs" / "rgb"
        if s.needs_semantics:
            if sid in GC_FINAL:
                d = np.load(SHEET / f"{sid}_{a.tag}.npz")
            else:
                d = np.load(XL / f"{sid}_xl_v18labeled.npz")
            xyz, labels = d["xyz"].astype(np.float64), d["labels"]
            if sid in GC_FINAL:
                scored += len(xyz)
        else:
            xyz, _ = submission_cloud(
                s, V1 / dataset / sid / "mvs" / "fused_union.ply", voxel=0.04)
            labels = None
        arts.append(SceneArtefacts(scene=s, rgb_dir=rgb, xyz=xyz, labels=labels))
        print(f"  {s.key}: {len(xyz)/1e6:.1f}M", flush=True)

    rep = build_submission(arts, Path(a.out), downsample=None)
    print(f"\n{a.out}: {'VALID' if rep.ok else rep.errors[:3]}  "
          f"{rep.stats.get('zip_mb')} MB", flush=True)
    print(f"scored Gold Coast points: {scored/1e6:.1f}M", flush=True)


if __name__ == "__main__":
    main()
