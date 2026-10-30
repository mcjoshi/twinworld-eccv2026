"""TSDF surface for Gold Coast: the accuracy attack.

Root cause, measured: our surface sits a median 13.8 cm from ground truth, just
outside the scorer's 10 cm radius. TSDF-averaging the 240+ interp-max depth maps
cancels independent depth noise; the zero-crossing surface should land within
~5 cm. TSDF was rejected on 11 Aug FOR TUM, where its precision cost is charged;
Gold Coast charges no precision, and this combination was never run - the same
shape of gap as the dense-cloud retest that produced the campaign's biggest win.

Output per scene: the marching-cubes surface sampled at ~5 cm, saved for
ceiling measurement and the fusion gate.
"""
import argparse, time
from pathlib import Path
import numpy as np
import open3d as o3d
from twinworld.dataio import Scene
from twinworld.geometry.gs_depth import interpolated_frames, render_depth
from twinworld.render.trainer import TrainConfig

def conv_point(frames):
    A=np.zeros((3,3)); b=np.zeros(3)
    for f in frames:
        d=f.R.T@np.array([0.,0.,1.]); d/=np.linalg.norm(d)
        P=np.eye(3)-np.outer(d,d); A+=P; b+=P@f.center
    return np.linalg.solve(A,b)

ap = argparse.ArgumentParser()
ap.add_argument("--gpu", type=int, default=3)
ap.add_argument("--scenes", nargs="+", default=["gold_coast/scene_009","gold_coast/scene_010",
                                                "gold_coast/scene_011","gold_coast/scene_012"])
ap.add_argument("--voxel", type=float, default=0.05)
ap.add_argument("--per-pair", type=int, default=6)
ap.add_argument("--max-pairs", type=int, default=240)
a = ap.parse_args()
OUT = Path("experiments/2026-08-28-tsdf-gc"); OUT.mkdir(parents=True, exist_ok=True)
V1 = Path("experiments/2026-08-08-pipeline-v1")
dev = f"cuda:{a.gpu}"
for key in a.scenes:
    s = Scene(*key.split("/"))
    splats = V1/s.dataset/s.scene_id/"gs"/"splats.pt"
    if not splats.exists(): print(f"[{s.key}] no splats, skip", flush=True); continue
    t0 = time.time()
    c = conv_point(s.train_frames)
    vol = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=a.voxel, sdf_trunc=a.voxel*4,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor)
    frames = list(s.train_frames) + interpolated_frames(s.train_frames, a.per_pair, a.max_pairs)
    n=0
    for fr, depth, alpha, rgb in render_depth(splats, frames, TrainConfig(), dev):
        d = depth.cpu().numpy(); al = alpha.cpu().numpy()
        d = np.where(al >= 0.7, d, 0.0).astype(np.float32)
        cam = fr.camera
        intr = o3d.camera.PinholeCameraIntrinsic(cam.width, cam.height, cam.fx, cam.fy, cam.cx, cam.cy)
        vol.integrate(o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.zeros_like(d, np.uint8)[..., None].repeat(3, -1).copy()),
            o3d.geometry.Image(d), depth_scale=1.0, depth_trunc=250.0,
            convert_rgb_to_intensity=False), intr, fr.world_to_cam)
        n += 1
    pc = vol.extract_point_cloud()
    xyz = np.asarray(pc.points)
    keep = np.linalg.norm(xyz - c, axis=1) <= 120.0
    xyz = xyz[keep]
    np.savez_compressed(OUT/f"{s.scene_id}_tsdf.npz", xyz=xyz.astype(np.float32))
    print(f"[{s.key}] {n} views integrated -> {len(xyz)/1e6:.1f}M surface pts "
          f"({time.time()-t0:.0f}s)", flush=True)
