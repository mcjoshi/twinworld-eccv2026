"""Does TSDF-fusing depth from ensemble members fix TUM's precision problem?

Raw GS depth union cost TUM F-score (0.534 -> 0.463): single-model depth is too
noisy, and TUM's metric charges for noise. But rendering showed that ensemble
members disagree independently - averaging 8 of them beat the best single run on
every metric. A TSDF volume is the geometric version of that average: each
member's depth maps carve signed distances, independent errors cancel, and the
extracted zero-crossing is the consensus surface.

Adoption rule, fixed in advance: mean F over the two test scenes must beat the
MVS baseline, with both scenes improving, for any TSDF variant to be considered
further (then validated on all four dev scenes before shipping).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from twinworld.dataio import Scene
from twinworld.geometry.gs_depth import render_depth
from twinworld.package import submission_cloud
from twinworld.render.trainer import TrainConfig, train_scene
from twinworld.scoring import read_ply_xyz, score_geometry, voxel_downsample

ITERS = 3000
N_MEMBERS = 8


def tsdf_fuse(scene, splat_paths, cfg, voxel=0.03, trunc=0.12, device="cuda"):
    import open3d as o3d

    vol = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel, sdf_trunc=trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    for sp in splat_paths:
        for fr, depth, alpha, rgb in render_depth(sp, scene.train_frames, cfg, device):
            d = depth.cpu().numpy()
            a = alpha.cpu().numpy()
            d = np.where(a >= 0.7, d, 0.0).astype(np.float32)  # 0 = no observation
            c = (rgb.cpu().numpy() * 255).astype(np.uint8)
            cam = fr.camera
            intr = o3d.camera.PinholeCameraIntrinsic(
                cam.width, cam.height, cam.fx, cam.fy, cam.cx, cam.cy)
            extr = fr.world_to_cam
            vol.integrate(
                o3d.geometry.RGBDImage.create_from_color_and_depth(
                    o3d.geometry.Image(np.ascontiguousarray(c)),
                    o3d.geometry.Image(d), depth_scale=1.0, depth_trunc=300.0,
                    convert_rgb_to_intensity=False),
                intr, extr)
    pc = vol.extract_point_cloud()
    return np.asarray(pc.points)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", default=["tum/scene_000", "tum/scene_002"])
    ap.add_argument("--src", default="experiments/2026-08-08-pipeline-v1")
    ap.add_argument("--work", default="/data2/manish_joshi/twinworld/work/tsdf")
    ap.add_argument("--out", default="experiments/2026-08-11-tsdf")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for key in args.scenes:
        s = Scene(*key.split("/"))
        ply = Path(args.src) / s.dataset / s.scene_id / "mvs" / "fused.ply"
        mvs, _ = submission_cloud(s, ply, cache=ply.parent / "submission_cloud.npz")
        gt, _ = read_ply_xyz(s.gt_ply)
        base = score_geometry(mvs, gt, s.key)
        print(f"\n[{s.key}] mvs baseline F={base.fscore:.4f} "
              f"(P@10 {base.precision[0.10]:.3f} R@10 {base.recall[0.10]:.3f})", flush=True)
        rows.append({"scene": s.key, "method": "mvs", "fscore": base.fscore})

        cfg = TrainConfig(iters=ITERS, refine_stop_iter=int(ITERS * 0.72))
        splats = []
        for i in range(N_MEMBERS):
            w = Path(args.work) / s.scene_id / f"m{i}"
            if not (w / "splats.pt").exists():
                c = TrainConfig(iters=ITERS, refine_stop_iter=int(ITERS * 0.72), seed=i)
                train_scene(s, ply, w, c, log_every=ITERS)
            splats.append(w / "splats.pt")
        print(f"  {len(splats)} member models ready", flush=True)

        for n_use, voxel in ((1, 0.03), (8, 0.03), (8, 0.02)):
            t0 = time.time()
            pts = tsdf_fuse(s, splats[:n_use], cfg, voxel=voxel)
            tag = f"tsdf_m{n_use}_v{voxel}"
            r = score_geometry(pts, gt, s.key)
            u = score_geometry(voxel_downsample(np.vstack([mvs, pts]), 0.02), gt, s.key)
            rows += [{"scene": s.key, "method": tag, "fscore": r.fscore,
                      "p10": r.precision[0.10], "r10": r.recall[0.10], "n": int(len(pts))},
                     {"scene": s.key, "method": tag + "+mvs", "fscore": u.fscore,
                      "p10": u.precision[0.10], "r10": u.recall[0.10]}]
            print(f"  {tag:18s} N={len(pts):9d} F={r.fscore:.4f} "
                  f"(P@10 {r.precision[0.10]:.3f} R@10 {r.recall[0.10]:.3f})  "
                  f"union F={u.fscore:.4f} ({u.fscore-base.fscore:+.4f})  "
                  f"({time.time()-t0:.0f}s)", flush=True)
            (out / "scores.json").write_text(json.dumps(rows, indent=2))

    (out / "scores.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
