"""Sparse-view Gaussian Splatting trainer built directly on gsplat.

Written rather than reusing gsplat's `simple_trainer.py` for two reasons: the
challenge fixes its own train/test split in the data layout (the trainer must
not carve out its own held-out views), and the initialisation comes from our
dense MVS cloud rather than the sparse COLMAP points, which is the main lever
under 12 training views.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..dataio import Scene, Frame


@dataclass
class TrainConfig:
    iters: int = 15000
    sh_degree: int = 3
    sh_increase_every: int = 1000
    init_points: int = 1_000_000
    init_scale_knn: int = 3
    init_opacity: float = 0.5
    # losses
    ssim_lambda: float = 0.2
    scale_reg: float = 0.01
    opacity_reg: float = 0.0
    # learning rates
    lr_means: float = 1.6e-4
    lr_scales: float = 5e-3
    lr_quats: float = 1e-3
    lr_opacities: float = 5e-2
    lr_sh0: float = 2.5e-3
    lr_shN: float = 2.5e-3 / 20
    # densification
    strategy: str = "default"  # "default" | "mcmc"
    cap_max: int = 2_000_000
    refine_stop_iter: int = 12000
    prune_opa: float = 0.005
    grow_grad2d: float = 2e-4
    # rendering
    rasterize_mode: str = "antialiased"
    near_plane: float = 0.01
    far_plane: float = 1e10
    background: str = "black"
    # packed=True is incompatible with per-camera `backgrounds` in gsplat 1.5.3
    # (image_dims collapses to () in packed mode), and it is not what gsplat's
    # own trainer defaults to. 12 views on an 80 GB A100 do not need it.
    packed: bool = False
    # sparse-view regularisation
    random_bg: bool = False
    depth_lambda: float = 0.0
    seed: int = 0


def _knn_scale(xyz: np.ndarray, k: int = 3) -> np.ndarray:
    from scipy.spatial import cKDTree

    d, _ = cKDTree(xyz).query(xyz, k=k + 1, workers=-1)
    return np.clip(d[:, 1:].mean(axis=1), 1e-4, None)


def rgb_to_sh0(rgb: np.ndarray) -> np.ndarray:
    return (np.asarray(rgb, np.float32) / 255.0 - 0.5) / 0.28209479177387814


@dataclass
class Splats:
    params: torch.nn.ParameterDict
    optimizers: dict

    @property
    def n(self) -> int:
        return self.params["means"].shape[0]


def init_splats(
    xyz: np.ndarray, rgb: np.ndarray | None, cfg: TrainConfig, device: str, scene_scale: float
) -> Splats:
    rng = np.random.default_rng(cfg.seed)
    if len(xyz) > cfg.init_points:
        sel = rng.choice(len(xyz), cfg.init_points, replace=False)
        xyz = xyz[sel]
        rgb = rgb[sel] if rgb is not None else None
    if rgb is None:
        rgb = np.full((len(xyz), 3), 128, np.uint8)

    scales = np.log(_knn_scale(xyz, cfg.init_scale_knn))[:, None].repeat(3, 1)
    quats = np.zeros((len(xyz), 4), np.float32)
    quats[:, 0] = 1.0
    opac = np.full(len(xyz), math.log(cfg.init_opacity / (1 - cfg.init_opacity)), np.float32)

    n_sh = (cfg.sh_degree + 1) ** 2
    sh0 = rgb_to_sh0(rgb)[:, None, :]
    shN = np.zeros((len(xyz), n_sh - 1, 3), np.float32)

    def P(a, dtype=torch.float32):
        return torch.nn.Parameter(torch.as_tensor(np.ascontiguousarray(a), dtype=dtype, device=device))

    params = torch.nn.ParameterDict(
        {
            "means": P(xyz.astype(np.float32)),
            "scales": P(scales.astype(np.float32)),
            "quats": P(quats),
            "opacities": P(opac),
            "sh0": P(sh0.astype(np.float32)),
            "shN": P(shN),
        }
    )
    # gsplat's strategies scale the means learning rate by the scene extent
    lrs = {
        "means": cfg.lr_means * scene_scale,
        "scales": cfg.lr_scales,
        "quats": cfg.lr_quats,
        "opacities": cfg.lr_opacities,
        "sh0": cfg.lr_sh0,
        "shN": cfg.lr_shN,
    }
    optimizers = {
        k: torch.optim.Adam([{"params": params[k], "lr": lr, "name": k}], eps=1e-15)
        for k, lr in lrs.items()
    }
    return Splats(params, optimizers)


def frames_to_tensors(frames: list[Frame], device: str):
    viewmats = torch.stack(
        [torch.as_tensor(fr.world_to_cam, dtype=torch.float32) for fr in frames]
    ).to(device)
    Ks = torch.stack([torch.as_tensor(fr.camera.K, dtype=torch.float32) for fr in frames]).to(device)
    return viewmats, Ks


def load_gt_images(frames: list[Frame], device: str) -> torch.Tensor:
    from PIL import Image

    imgs = []
    for fr in frames:
        arr = np.asarray(Image.open(fr.image_path).convert("RGB"), dtype=np.float32) / 255.0
        imgs.append(torch.from_numpy(arr))
    return torch.stack(imgs).to(device)  # (N, H, W, 3)


def _ssim(a, b):
    """a, b: (1, 3, H, W) in [0, 1]."""
    from ..scoring import ssim as _s

    return _s(a, b)


def train_scene(
    scene: Scene,
    init_ply: Path,
    out_dir: Path,
    cfg: TrainConfig | None = None,
    device: str = "cuda",
    log_every: int = 1000,
) -> dict:
    from gsplat import rasterization
    from gsplat.strategy import DefaultStrategy, MCMCStrategy

    from ..geometry.densify import robust_core

    cfg = cfg or TrainConfig()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    t0 = time.time()

    # --- initialisation from the dense reconstruction
    from plyfile import PlyData

    v = PlyData.read(str(init_ply))["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    names = {p.name for p in v.properties}
    rgb = (
        np.stack([v["red"], v["green"], v["blue"]], 1).astype(np.uint8)
        if {"red", "green", "blue"} <= names
        else None
    )
    # Keep only what the training views can actually see: the sparse cloud spans
    # the whole 1400 m flight and distant points only produce floaters.
    m = scene.train_frustum_mask(xyz, margin_px=100)
    # Frustum membership alone is not enough. COLMAP fusion leaves a far-field
    # tail that is inside the frustum but kilometres away (on TUM scene_000 it
    # stretches the cloud to 9.4 km). Those points are given knn-derived initial
    # scales of the same order, so a handful of them become Gaussians the size
    # of the scene, and `scene_scale` used for the means learning rate is
    # unaffected only because it comes from the cameras.
    m &= robust_core(xyz, scene.train_frames)
    xyz, rgb = xyz[m], (rgb[m] if rgb is not None else None)

    frames = scene.train_frames
    centers = np.stack([fr.center for fr in frames])
    scene_scale = float(np.linalg.norm(centers - centers.mean(0), axis=1).max()) * 1.1
    scene_scale = max(scene_scale, 1.0)

    splats = init_splats(xyz, rgb, cfg, device, scene_scale)

    if cfg.strategy == "mcmc":
        strategy = MCMCStrategy(
            cap_max=cfg.cap_max, refine_stop_iter=cfg.refine_stop_iter, verbose=False
        )
    else:
        strategy = DefaultStrategy(
            prune_opa=cfg.prune_opa,
            grow_grad2d=cfg.grow_grad2d,
            refine_stop_iter=cfg.refine_stop_iter,
            verbose=False,
        )
    # MCMCStrategy.initialize_state takes no arguments; only the default
    # gradient-based strategy scales its thresholds by the scene extent.
    state = (
        strategy.initialize_state()
        if isinstance(strategy, MCMCStrategy)
        else strategy.initialize_state(scene_scale=scene_scale)
    )
    strategy.check_sanity(splats.params, splats.optimizers)

    viewmats, Ks = frames_to_tensors(frames, device)
    gt = load_gt_images(frames, device)
    H, W = frames[0].camera.height, frames[0].camera.width
    bg = torch.zeros(3, device=device) if cfg.background == "black" else torch.ones(3, device=device)

    schedulers = {
        "means": torch.optim.lr_scheduler.ExponentialLR(
            splats.optimizers["means"], gamma=0.01 ** (1.0 / cfg.iters)
        )
    }

    history = []
    n_train = len(frames)
    perm = np.random.permutation(n_train)
    for step in range(cfg.iters):
        i = perm[step % n_train]
        if step % n_train == 0:
            perm = np.random.permutation(n_train)

        sh_deg = min(cfg.sh_degree, step // cfg.sh_increase_every)
        colors = torch.cat([splats.params["sh0"], splats.params["shN"]], 1)
        background = (
            torch.rand(1, 3, device=device) if cfg.random_bg else bg[None]
        )

        render, alphas, info = rasterization(
            means=splats.params["means"],
            quats=splats.params["quats"],
            scales=torch.exp(splats.params["scales"]),
            opacities=torch.sigmoid(splats.params["opacities"]),
            colors=colors,
            viewmats=viewmats[i : i + 1],
            Ks=Ks[i : i + 1],
            width=W,
            height=H,
            sh_degree=sh_deg,
            packed=cfg.packed,
            near_plane=cfg.near_plane,
            far_plane=cfg.far_plane,
            rasterize_mode=cfg.rasterize_mode,
            backgrounds=background,
            absgrad=False,
            render_mode="RGB",
        )
        strategy.step_pre_backward(splats.params, splats.optimizers, state, step, info)

        pred = render[..., :3].clamp(0, 1)
        target = gt[i : i + 1]
        l1 = (pred - target).abs().mean()
        s = _ssim(pred.permute(0, 3, 1, 2), target.permute(0, 3, 1, 2))
        loss = (1 - cfg.ssim_lambda) * l1 + cfg.ssim_lambda * (1 - s)
        if cfg.scale_reg > 0:
            loss = loss + cfg.scale_reg * torch.exp(splats.params["scales"]).mean()
        if cfg.opacity_reg > 0:
            loss = loss + cfg.opacity_reg * torch.sigmoid(splats.params["opacities"]).abs().mean()

        loss.backward()

        # Optimiser first, densification second. `step_post_backward` grows and
        # prunes Gaussians, which rebuilds the parameter tensors and their Adam
        # state; stepping after that would apply gradients computed for the
        # previous set of Gaussians to a different one. This is the order
        # gsplat's own trainer uses.
        for opt in splats.optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        schedulers["means"].step()

        if isinstance(strategy, MCMCStrategy):
            strategy.step_post_backward(
                splats.params, splats.optimizers, state, step, info,
                lr=schedulers["means"].get_last_lr()[0],
            )
        else:
            strategy.step_post_backward(
                splats.params, splats.optimizers, state, step, info, packed=cfg.packed
            )

        if step % log_every == 0 or step == cfg.iters - 1:
            rec = {
                "step": step,
                "loss": float(loss),
                "l1": float(l1),
                "psnr": float(-10 * math.log10(max(float(((pred - target) ** 2).mean()), 1e-12))),
                "n_gauss": int(splats.params["means"].shape[0]),
                "sh_degree": sh_deg,
            }
            history.append(rec)
            print(
                f"  [{scene.key}] step {step:6d}  loss {rec['loss']:.4f}  "
                f"train psnr {rec['psnr']:.2f}  n={rec['n_gauss']}",
                flush=True,
            )

    ckpt = {k: v.detach().cpu() for k, v in splats.params.items()}
    torch.save({"splats": ckpt, "config": asdict(cfg)}, out_dir / "splats.pt")
    summary = {
        "scene": scene.key,
        "n_gaussians": int(splats.params["means"].shape[0]),
        "n_init_points": int(len(xyz)),
        "scene_scale": scene_scale,
        "seconds": round(time.time() - t0, 1),
        "config": asdict(cfg),
        "history": history,
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


@torch.no_grad()
def render_frames(
    splats_path: Path,
    frames: list[Frame],
    out_rgb_dir: Path,
    cfg: TrainConfig | None = None,
    device: str = "cuda",
) -> None:
    """Render `frames` and write `<out_rgb_dir>/<stem>.png` at reference resolution."""
    from gsplat import rasterization
    from PIL import Image

    blob = torch.load(splats_path, map_location=device, weights_only=False)
    p = {k: v.to(device) for k, v in blob["splats"].items()}
    cfg = cfg or TrainConfig(**blob.get("config", {}))

    out_rgb_dir = Path(out_rgb_dir)
    out_rgb_dir.mkdir(parents=True, exist_ok=True)
    colors = torch.cat([p["sh0"], p["shN"]], 1)
    bg = torch.zeros(1, 3, device=device) if cfg.background == "black" else torch.ones(1, 3, device=device)

    for fr in frames:
        viewmat = torch.as_tensor(fr.world_to_cam, dtype=torch.float32, device=device)[None]
        K = torch.as_tensor(fr.camera.K, dtype=torch.float32, device=device)[None]
        render, _, _ = rasterization(
            means=p["means"], quats=p["quats"], scales=torch.exp(p["scales"]),
            opacities=torch.sigmoid(p["opacities"]), colors=colors,
            viewmats=viewmat, Ks=K, width=fr.camera.width, height=fr.camera.height,
            sh_degree=cfg.sh_degree, packed=cfg.packed, near_plane=cfg.near_plane,
            far_plane=cfg.far_plane, rasterize_mode=cfg.rasterize_mode,
            backgrounds=bg, render_mode="RGB",
        )
        img = (render[0, ..., :3].clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
        Image.fromarray(img).save(out_rgb_dir / f"{fr.stem}.png")


@torch.no_grad()
def gaussian_centres(
    splats_path: Path,
    opacity_min: float = 0.30,
    scale_max: float = 0.50,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Surface-like Gaussian centres from a trained model, as a point cloud.

    Training already produces these, so they are a free second source of
    geometry. They are useful mainly for Gold Coast, where the semantic score is
    capped by how much ground truth lies within 10 cm of a submitted point and
    geometric precision is never scored, so extra plausible coverage is close to
    free.

    Two filters keep only the Gaussians that behave like surface: those too
    transparent to be committing to a surface, and those so large they are
    modelling low-frequency background rather than geometry.

    Returns (xyz, rgb).
    """
    blob = torch.load(splats_path, map_location=device, weights_only=False)
    p = blob["splats"]
    opacity = torch.sigmoid(p["opacities"])
    scale = torch.exp(p["scales"]).max(dim=1).values
    keep = (opacity >= opacity_min) & (scale <= scale_max)

    xyz = p["means"][keep].cpu().numpy().astype(np.float64)
    # the degree-0 spherical harmonic is the view-independent colour
    sh0 = p["sh0"][keep, 0].cpu().numpy()
    rgb = np.clip(sh0 * 0.28209479177387814 + 0.5, 0.0, 1.0)
    return xyz, (rgb * 255).astype(np.uint8)
