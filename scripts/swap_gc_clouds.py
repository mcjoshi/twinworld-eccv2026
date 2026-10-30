"""Rewrite a submission zip, replacing only the scored Gold Coast point clouds.

Building a submission from scratch recomputes every TUM cloud and re-compresses
every render, which is slow and, more importantly, leaves open the question of
whether anything outside Gold Coast drifted. Copying the source zip entry by
entry and substituting two files removes the question: the renders and the TUM
geometry are the same bytes that scored 1.4529, so the rendering and F-score
terms cannot move and the whole delta is the semantic term.

Scenes 009 and 010 are left as the source has them - already decimated to a 1 m
voxel, because the final phase does not score them.

The submitted PLY is quantised to a 1 cm grid and never deduplicated, so sheet
offsets of 5 cm and up survive packaging intact.
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import numpy as np

from twinworld.package import _ply_bytes

SHEET = Path("experiments/2026-08-30-sheets")
V1 = Path("experiments/2026-08-08-pipeline-v1")
TARGETS = ("gold_coast/scene_011", "gold_coast/scene_012")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="submissions/v18-final.zip")
    ap.add_argument("--tag", required=True, help="sheet tag, e.g. s07")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tum-voxel", type=float, default=0.0,
                    help="also rebuild the scored TUM clouds at this voxel; "
                         "0 keeps the source zip's TUM bytes untouched")
    a = ap.parse_args()

    clouds = {}
    for key in TARGETS:
        sid = key.split("/")[1]
        d = np.load(SHEET / f"{sid}_{a.tag}.npz")
        clouds[key] = (d["xyz"], d["labels"])
        print(f"  {key}: {len(d['xyz'])/1e6:.1f}M sheeted points", flush=True)

    # TUM carries the F-score, which the scorer computes after its own 2 cm
    # downsample, so a coarser submission voxel throws away resolution the
    # scorer would have used. Only the final-phase scenes are worth rebuilding.
    if a.tum_voxel > 0:
        from twinworld.dataio import TUM_FINAL, Scene
        from twinworld.package import submission_cloud
        for sid in TUM_FINAL:
            s = Scene("tum", sid)
            xyz, _ = submission_cloud(
                s, V1 / "tum" / sid / "mvs" / "fused_union.ply", voxel=a.tum_voxel)
            clouds[f"tum/{sid}"] = (xyz.astype(np.float32), None)
            print(f"  tum/{sid}: {len(xyz)/1e6:.1f}M pts at {a.tum_voxel:.2f} m",
                  flush=True)

    expected_n = len(clouds)
    src = zipfile.ZipFile(a.src)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    swapped, before, after = 0, 0, 0

    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for info in src.infolist():
            parts = info.filename.split("/")
            key = "/".join(parts[:2]) if len(parts) >= 2 else ""
            if info.filename.endswith("point_cloud.ply") and key in clouds:
                xyz, lab = clouds.pop(key)
                before += info.file_size
                grid = 0.005 if key.startswith("tum") else 0.01
                blob = _ply_bytes(xyz, lab, grid=grid)
                after += len(blob)
                z.writestr(info.filename, blob)
                swapped += 1
                print(f"  swapped {key}: {len(xyz)/1e6:.1f}M pts, "
                      f"{len(blob)/1e6:.0f} MB raw", flush=True)
                del xyz, lab, blob
            else:
                z.writestr(info, src.read(info.filename))

    if clouds:
        raise SystemExit(f"never matched in {a.src}: {sorted(clouds)}")
    if swapped != expected_n:
        raise SystemExit(f"swapped {swapped}, expected {expected_n}")

    with zipfile.ZipFile(out) as chk:
        bad = chk.testzip()
        names = [n for n in chk.namelist() if n.endswith("point_cloud.ply")]
    print(f"\n{out}  {out.stat().st_size/1e6:.0f} MB  testzip={bad}  "
          f"{len(names)} scenes", flush=True)


if __name__ == "__main__":
    main()
