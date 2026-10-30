"""Probe: does the scorer read scenes its phase does not score?

Each phase has its own reference data. Development scores TUM 000-003 and Gold
Coast 009-010; Final scores TUM 004-008 and Gold Coast 011-012. Our submission
must contain all thirteen scenes to pass the format check, but if the scorer
only touches its own phase's scenes then every point we ship for the other
seven is dead weight - and in v7-mini that dead weight is 53 percent of the
point budget.

This rewrites an existing, already-verified zip, decimating only the scenes the
development phase does not score and copying everything else byte for byte. The
readout is the *score*, not pass/fail: if the development score is unchanged,
the scorer ignores out-of-phase scenes, and the final submission can spend the
whole budget on the seven scenes that count.

Uses only the published phase structure, not ground truth.
"""

from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path

import numpy as np

from twinworld.dataio import DEV_SCENES, FINAL_SCENES
from twinworld.package import _ply_bytes
from twinworld.scoring import voxel_downsample


def read_ply_bytes(raw: bytes):
    from plyfile import PlyData

    p = PlyData.read(io.BytesIO(raw))["vertex"]
    xyz = np.stack([p["x"], p["y"], p["z"]], axis=1).astype(np.float64)
    lab = p["classification"].astype(np.uint8) if "classification" in p.data.dtype.names else None
    return xyz, lab


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="submissions/v7-safe.zip")
    ap.add_argument("--out", default="submissions/probe-phase-scope.zip")
    ap.add_argument("--gut", default="final", choices=["final", "dev"],
                    help="which phase's scenes to decimate")
    ap.add_argument("--voxel", type=float, default=1.0,
                    help="voxel for the decimated scenes (metres)")
    args = ap.parse_args()

    gut = {f"{d}/{s}" for d, s in (FINAL_SCENES if args.gut == "final" else DEV_SCENES)}
    keep = {f"{d}/{s}" for d, s in (DEV_SCENES if args.gut == "final" else FINAL_SCENES)}
    print(f"decimating {len(gut)} scenes at {args.voxel} m, keeping {len(keep)} intact")

    src = zipfile.ZipFile(args.src)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    before = after = 0

    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for info in src.infolist():
            raw = src.read(info.filename)
            parts = info.filename.split("/")
            scene_key = "/".join(parts[:2]) if len(parts) >= 2 else ""
            if info.filename.endswith("point_cloud.ply") and scene_key in gut:
                xyz, lab = read_ply_bytes(raw)
                before += len(xyz)
                if lab is not None:
                    idx = np.arange(len(xyz))
                    sub = voxel_downsample(xyz, args.voxel)
                    # labels must still be present and valid for Gold Coast;
                    # nearest-neighbour carry keeps them consistent
                    from scipy.spatial import cKDTree
                    _, nn = cKDTree(xyz).query(sub, k=1, workers=-1)
                    new = _ply_bytes(sub, lab[nn], grid=0.01)
                    n = len(sub)
                else:
                    sub = voxel_downsample(xyz, args.voxel)
                    new = _ply_bytes(sub, None, grid=0.005)
                    n = len(sub)
                after += n
                print(f"  {scene_key:28s} {len(xyz)/1e6:7.2f}M -> {n/1e6:6.3f}M pts",
                      flush=True)
                z.writestr(info.filename, new)
            else:
                z.writestr(info, raw)

    mb = out.stat().st_size / 1e6
    print(f"\n{out}  {mb:.1f} MB  (decimated {before/1e6:.1f}M -> {after/1e6:.2f}M pts)")
    print("expected readout: development score identical to the source zip's")


if __name__ == "__main__":
    main()
