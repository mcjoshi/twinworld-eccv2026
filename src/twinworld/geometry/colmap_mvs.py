"""Dense geometry from COLMAP PatchMatch MVS using the challenge's given poses.

The dataset ships an undistorted PINHOLE model per scene plus a triangulated
`points3D.txt`, so there is no SfM to run: we go straight to undistortion (which
also builds the dense workspace layout), PatchMatch stereo, and fusion.

The sparse cloud spans the whole flight (roughly 1400 m) while a scene's actual
content is tens of metres, so depth ranges are derived from the points that the
train views actually see, not from the raw sparse extent.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from ..dataio import Scene, read_cameras_text, read_images_text

COLMAP_BIN = os.environ.get("COLMAP_BIN", "/data2/manish_joshi/envs/colmap/bin/colmap")
WORK_ROOT = Path(os.environ.get("TWINWORLD_WORK", "/data2/manish_joshi/twinworld/work"))


@dataclass
class MvsConfig:
    # Upsampling factor applied to the images and intrinsics before MVS.
    # Ground sampling distance is about 3.4 cm/px on TUM (36 m altitude, f=1053
    # px) while the ground truth is a 2 cm scan, so native resolution caps how
    # densely and how precisely depth can be estimated. Values above 1 cost
    # roughly the square of the factor in runtime.
    image_scale: float = 1.0
    max_image_size: int = 2000
    window_radius: int = 5
    window_step: int = 1
    num_samples: int = 15
    num_iterations: int = 5
    geom_consistency: bool = True
    filter_min_ncc: float = 0.1
    filter_min_triangulation_angle: float = 1.0
    filter_min_num_consistent: int = 2
    filter_geom_consistency_max_cost: float = 1.0
    cache_size: int = 32
    # fusion
    fusion_min_num_pixels: int = 3
    fusion_max_reproj_error: float = 2.0
    fusion_max_depth_error: float = 0.01
    fusion_max_normal_error: float = 10.0
    # depth range. `auto_depth` lets COLMAP derive a per-image range from the
    # sparse points each image actually observes, which is far tighter than any
    # single global range over a 1400 m flight and materially improves
    # PatchMatch sampling resolution.
    auto_depth: bool = True
    depth_margin: float = 0.4


def _run(cmd: list[str], log: Path, env: dict | None = None) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a") as f:
        f.write("\n$ " + " ".join(str(c) for c in cmd) + "\n")
        f.flush()
        r = subprocess.run([str(c) for c in cmd], stdout=f, stderr=subprocess.STDOUT, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"command failed ({r.returncode}), see {log}: {' '.join(map(str, cmd))}")


def sparse_depth_range(scene: Scene, margin: float) -> tuple[float, float]:
    """Depth range over the sparse points that the train views actually observe."""
    from plyfile import PlyData

    v = PlyData.read(str(scene.sparse_ply))["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
    xyz = xyz[scene.train_frustum_mask(xyz)]
    depths = []
    for fr in scene.train_frames:
        cam = xyz @ fr.R.T + fr.t
        d = cam[:, 2]
        d = d[d > 0]
        if len(d):
            depths.append(d)
    if not depths:
        return 1.0, 500.0
    d = np.concatenate(depths)
    lo, hi = np.percentile(d, 1), np.percentile(d, 99)
    span = hi - lo
    return float(max(0.1, lo - margin * span)), float(hi + margin * span)


def _write_undistorted_sparse(scene: Scene, dst: Path) -> None:
    """Copy the sparse model, dropping points3D tracks that reference images we
    do not have. COLMAP's undistorter validates track consistency, and the
    provided points3D is triangulated from train and test images alike."""
    dst.mkdir(parents=True, exist_ok=True)
    src = scene.train_sparse_dir
    shutil.copy(src / "cameras.txt", dst / "cameras.txt")
    shutil.copy(src / "images.txt", dst / "images.txt")

    cameras = read_cameras_text(src / "cameras.txt")
    valid_ids = {fr.image_id for fr in read_images_text(src / "images.txt", cameras)}
    out = []
    with open(src / "points3D.txt") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            p = line.split()
            head, track = p[:8], p[8:]
            kept = []
            for i in range(0, len(track) - 1, 2):
                if int(track[i]) in valid_ids:
                    kept += [track[i], track[i + 1]]
            if kept:
                out.append(" ".join(head + kept))
    (dst / "points3D.txt").write_text(
        "# 3D point list with one line of data per point:\n"
        "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"
        + "\n".join(out)
        + "\n"
    )


def _write_scaled_images(scene: Scene, dst: Path, scale: float) -> Path:
    """Resize the training images by `scale`, preserving their file names.

    COLMAP's `--max_image_size` only ever downscales, so upsampling has to
    happen before the undistorter sees the images, with the intrinsics scaled to
    match.
    """
    from PIL import Image

    dst.mkdir(parents=True, exist_ok=True)
    for fr in scene.train_frames:
        out = dst / fr.name
        if out.exists():
            continue
        with Image.open(fr.image_path) as im:
            w, h = im.size
            im.resize((round(w * scale), round(h * scale)), Image.LANCZOS).save(out, quality=98)
    return dst


def _scale_cameras_txt(path: Path, scale: float) -> None:
    """Scale WIDTH, HEIGHT and every intrinsic parameter in place."""
    lines_out = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            lines_out.append(line)
            continue
        p = line.split()
        p[2] = str(round(int(p[2]) * scale))
        p[3] = str(round(int(p[3]) * scale))
        p[4:] = [f"{float(v) * scale:.10f}" for v in p[4:]]
        lines_out.append(" ".join(p))
    path.write_text("\n".join(lines_out) + "\n")


def run_mvs(
    scene: Scene,
    out_dir: Path,
    cfg: MvsConfig | None = None,
    gpu_index: int = 2,
    keep_workspace: bool = False,
    tag: str = "default",
) -> dict:
    """Run undistort -> patch_match_stereo -> stereo_fusion for one scene.

    Returns a summary dict and writes `fused.ply` into `out_dir`.

    `tag` separates concurrent runs: COLMAP writes depth and normal maps into the
    workspace, so two configs sharing a workspace silently corrupt each other's
    results.
    """
    cfg = cfg or MvsConfig()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ws = WORK_ROOT / "mvs" / tag / scene.dataset / scene.scene_id
    log = out_dir / "colmap.log"
    t0 = time.time()

    if ws.exists():
        shutil.rmtree(ws)
    ws.mkdir(parents=True)

    sparse_in = ws / "sparse_in"
    _write_undistorted_sparse(scene, sparse_in)

    image_dir = scene.train_images_dir
    max_size = cfg.max_image_size
    if cfg.image_scale != 1.0:
        image_dir = _write_scaled_images(scene, ws / "images_scaled", cfg.image_scale)
        _scale_cameras_txt(sparse_in / "cameras.txt", cfg.image_scale)
        max_size = max(max_size, int(2200 * cfg.image_scale))

    _run(
        [
            COLMAP_BIN, "image_undistorter",
            "--image_path", image_dir,
            "--input_path", sparse_in,
            "--output_path", ws / "dense",
            "--output_type", "COLMAP",
            "--max_image_size", max_size,
        ],
        log,
    )

    if cfg.auto_depth:
        dmin = dmax = -1.0  # COLMAP: derive a per-image range from the sparse model
    else:
        dmin, dmax = sparse_depth_range(scene, cfg.depth_margin)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu_index))
    _run(
        [
            COLMAP_BIN, "patch_match_stereo",
            "--workspace_path", ws / "dense",
            "--workspace_format", "COLMAP",
            "--PatchMatchStereo.gpu_index", "0",
            "--PatchMatchStereo.depth_min", f"{dmin:.4f}",
            "--PatchMatchStereo.depth_max", f"{dmax:.4f}",
            "--PatchMatchStereo.window_radius", cfg.window_radius,
            "--PatchMatchStereo.window_step", cfg.window_step,
            "--PatchMatchStereo.num_samples", cfg.num_samples,
            "--PatchMatchStereo.num_iterations", cfg.num_iterations,
            "--PatchMatchStereo.geom_consistency", int(cfg.geom_consistency),
            "--PatchMatchStereo.filter_min_ncc", cfg.filter_min_ncc,
            "--PatchMatchStereo.filter_min_triangulation_angle", cfg.filter_min_triangulation_angle,
            "--PatchMatchStereo.filter_min_num_consistent", cfg.filter_min_num_consistent,
            "--PatchMatchStereo.filter_geom_consistency_max_cost", cfg.filter_geom_consistency_max_cost,
            "--PatchMatchStereo.cache_size", cfg.cache_size,
        ],
        log,
        env,
    )

    fused = out_dir / "fused.ply"
    _run(
        [
            COLMAP_BIN, "stereo_fusion",
            "--workspace_path", ws / "dense",
            "--workspace_format", "COLMAP",
            "--input_type", "geometric" if cfg.geom_consistency else "photometric",
            "--output_path", fused,
            "--StereoFusion.min_num_pixels", cfg.fusion_min_num_pixels,
            "--StereoFusion.max_reproj_error", cfg.fusion_max_reproj_error,
            "--StereoFusion.max_depth_error", cfg.fusion_max_depth_error,
            "--StereoFusion.max_normal_error", cfg.fusion_max_normal_error,
            "--StereoFusion.cache_size", cfg.cache_size,
        ],
        log,
        env,
    )

    from ..scoring import read_ply_xyz

    xyz, _ = read_ply_xyz(fused)
    summary = {
        "scene": scene.key,
        "n_points": int(len(xyz)),
        "depth_min": dmin,
        "depth_max": dmax,
        "seconds": round(time.time() - t0, 1),
        "config": asdict(cfg),
        "gpu_index": gpu_index,
    }
    (out_dir / "mvs_summary.json").write_text(json.dumps(summary, indent=2))
    if not keep_workspace:
        shutil.rmtree(ws, ignore_errors=True)
    return summary
