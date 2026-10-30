"""Build Gold Coast clouds that fuse stereo with the splatting model's own depth.

Measured on scene_009, the union nearly doubles the semantic ceiling, from 0.171
to 0.320, and the gains land precisely where stereo failed: wall coverage 0.023
to 0.125 and window 0.007 to 0.121.

Gold Coast only. The same union costs F-score on TUM (0.534 to 0.519) because it
trades precision for recall, and TUM is scored on an F-score while Gold Coast is
scored purely on whether ground truth has a submitted point within 10 cm.

No multi-view agreement filter is applied. Requiring two views to agree removes
most of the benefit (ceiling 0.320 to 0.223) because the surfaces at issue are
exactly the ones few views resolve well. The points are still a real estimate
from a model fitted to real photographs, not invented geometry, but they are
noisier than stereo output and that is why this is not used on TUM.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from twinworld.dataio import Scene
from twinworld.geometry.gs_depth import gs_point_cloud, interpolated_frames, render_depth, unproject
from twinworld.package import submission_cloud
from twinworld.render.trainer import TrainConfig, train_scene
from twinworld.scoring import voxel_downsample_indices, write_ply_xyz

ITERS = 7000
VOXEL = 0.06


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--src", default="experiments/2026-08-08-pipeline-v1")
    ap.add_argument("--work", default="/data2/manish_joshi/twinworld/work/gcfuse")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--alpha-min", type=float, default=0.7)
    ap.add_argument("--interpolated", action="store_true",
                    help="also render depth from viewpoints between training cameras")
    ap.add_argument("--suffix", default="")
    args = ap.parse_args()

    for key in args.scenes:
        s = Scene(*key.split("/"))
        out_dir = Path(args.src) / s.dataset / s.scene_id / "mvs"
        out_ply = out_dir / f"fused_gs{args.suffix}.ply"
        if out_ply.exists():
            print(f"[{s.key}] already built, skipping", flush=True)
            continue
        t0 = time.time()

        mvs, mvs_rgb = submission_cloud(s, out_dir / "fused.ply",
                                        cache=out_dir / "submission_cloud.npz")
        cfg = TrainConfig(iters=ITERS, refine_stop_iter=int(ITERS * 0.72))
        work = Path(args.work) / s.scene_id
        if not (work / "splats.pt").exists():
            train_scene(s, s.sparse_ply, work, cfg, log_every=ITERS)
        gs, gs_rgb = gs_point_cloud(s, work / "splats.pt", cfg, alpha_min=args.alpha_min,
                                    stride=args.stride, min_views=1)
        if args.interpolated:
            ip, ic = [], []
            for fr, d, a, rgb in render_depth(work / "splats.pt",
                                              interpolated_frames(s.train_frames, per_pair=2), cfg):
                w, c = unproject(fr, d.cpu().numpy(), a.cpu().numpy(), rgb.cpu().numpy(),
                                 args.alpha_min, args.stride)
                ip.append(w); ic.append(c)
            if ip:
                gs = np.vstack([gs] + ip)
                gs_rgb = np.vstack([gs_rgb] + ic)

        xyz = np.vstack([mvs, gs])
        rgb = np.vstack([mvs_rgb, gs_rgb]) if mvs_rgb is not None else None
        keep = voxel_downsample_indices(xyz, VOXEL)
        xyz = xyz[keep]
        rgb = rgb[keep] if rgb is not None else None

        write_ply_xyz(out_ply, xyz)
        np.savez_compressed(out_dir / f"submission_cloud_gs{args.suffix}_v{VOXEL}.npz",
                            xyz=xyz.astype(np.float32), **({"rgb": rgb} if rgb is not None else {}))
        (out_dir / f"fused_gs{args.suffix}.json").write_text(json.dumps(
            {"mvs_points": int(len(mvs)), "gs_points": int(len(gs)),
             "fused_points": int(len(xyz)), "voxel": VOXEL,
             "alpha_min": args.alpha_min, "stride": args.stride,
             "seconds": round(time.time() - t0, 1)}, indent=2))
        print(f"[{s.key}] mvs {len(mvs)} + gs {len(gs)} -> {len(xyz)} fused "
              f"({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
