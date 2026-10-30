"""Dense geometry from a trained Gaussian Splatting model's own depth.

PatchMatch stereo needs photometric correspondence across views, and on these
captures it finds it on roofs and largely fails on facades: Gold Coast wall
coverage at 10 cm is 0.02 against 0.45 for roofs. But the facades are not
unobserved. Their median incidence angle to the best-placed training camera is
41 degrees, and the splatting model reconstructs them well enough to render the
training views at 25-35 dB.

So the radiance field holds geometry that stereo threw away. Rendering its
expected depth at the training poses and unprojecting recovers that geometry,
including on surfaces where correspondence search failed.

Two things make this trustworthy rather than wishful:

  alpha        a ray that accumulated little opacity did not hit a surface, so
               the depth along it is meaningless and is discarded
  agreement    a point is kept only if several training views independently put
               a surface near the same place, which is the same consistency
               requirement stereo imposes, applied after the fact

Without the second check this would happily invent surface in the empty space
that a single view cannot disambiguate.
"""

from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def render_depth(splats_path, frames, cfg, device: str = "cuda", chunk: int = 1):
    """Expected depth and alpha at each frame. Yields (frame, depth HxW, alpha HxW)."""
    from gsplat import rasterization

    blob = torch.load(splats_path, map_location=device, weights_only=False)
    p = {k: v.to(device) for k, v in blob["splats"].items()}
    colors = torch.cat([p["sh0"], p["shN"]], 1)
    bg = torch.zeros(1, 3, device=device)

    for fr in frames:
        viewmat = torch.as_tensor(fr.world_to_cam, dtype=torch.float32, device=device)[None]
        K = torch.as_tensor(fr.camera.K, dtype=torch.float32, device=device)[None]
        render, alphas, _ = rasterization(
            means=p["means"], quats=p["quats"], scales=torch.exp(p["scales"]),
            opacities=torch.sigmoid(p["opacities"]), colors=colors,
            viewmats=viewmat, Ks=K, width=fr.camera.width, height=fr.camera.height,
            sh_degree=cfg.sh_degree, packed=cfg.packed, near_plane=cfg.near_plane,
            far_plane=cfg.far_plane, rasterize_mode=cfg.rasterize_mode,
            backgrounds=bg, render_mode="RGB+ED",
        )
        # "ED" is expected depth along the view axis, already normalised by alpha
        depth = render[0, ..., 3].float()
        alpha = alphas[0, ..., 0].float()
        rgb = render[0, ..., :3].clamp(0, 1).float()
        yield fr, depth, alpha, rgb


def unproject(fr, depth: np.ndarray, alpha: np.ndarray, rgb: np.ndarray,
              alpha_min: float = 0.7, stride: int = 1):
    """Back-project a depth image into world coordinates.

    `depth` is along the camera's view axis, so the standard pinhole inverse
    applies directly.
    """
    H, W = depth.shape
    ys, xs = np.mgrid[0:H:stride, 0:W:stride]
    d = depth[::stride, ::stride]
    a = alpha[::stride, ::stride]
    c = rgb[::stride, ::stride]
    m = (a >= alpha_min) & np.isfinite(d) & (d > 0)
    if not m.any():
        return np.empty((0, 3)), np.empty((0, 3), np.uint8)

    xs, ys, d, c = xs[m], ys[m], d[m], c[m]
    cam = fr.camera
    x = (xs - cam.cx) * d / cam.fx
    y = (ys - cam.cy) * d / cam.fy
    pts_cam = np.stack([x, y, d], axis=1)
    world = (pts_cam - fr.tvec) @ fr.R  # R^T (p - t), written as row-vector form
    return world, (c * 255).astype(np.uint8)


def multiview_filter(points: np.ndarray, per_view_counts: list[int], min_views: int = 2,
                     radius: float = 0.10):
    """Keep points that at least `min_views` different views agree on.

    A point survives only if points contributed by other views lie within
    `radius` of it. That is the geometric-consistency check stereo applies
    during matching, applied here after rendering instead.
    """
    from scipy.spatial import cKDTree

    if min_views <= 1 or len(points) == 0:
        return np.ones(len(points), bool)
    view_id = np.concatenate([np.full(n, i, np.int32) for i, n in enumerate(per_view_counts)])
    tree = cKDTree(points)
    keep = np.zeros(len(points), bool)
    # count how many distinct views contribute a neighbour, in chunks to bound memory
    for a in range(0, len(points), 200_000):
        b = min(a + 200_000, len(points))
        nbrs = tree.query_ball_point(points[a:b], radius, workers=-1)
        for i, nb in enumerate(nbrs):
            if len(nb) < min_views:
                continue
            vs = view_id[nb]
            if (vs != view_id[a + i]).any():
                keep[a + i] = True
    return keep


def gs_point_cloud(scene, splats_path, cfg, alpha_min: float = 0.7, stride: int = 2,
                   min_views: int = 2, agree_radius: float = 0.10, device: str = "cuda"):
    """Dense cloud from a trained splatting model, rendered at the training poses."""
    pts, cols, counts = [], [], []
    for fr, depth, alpha, rgb in render_depth(splats_path, scene.train_frames, cfg, device):
        w, c = unproject(fr, depth.cpu().numpy(), alpha.cpu().numpy(),
                         rgb.cpu().numpy(), alpha_min, stride)
        pts.append(w)
        cols.append(c)
        counts.append(len(w))
    if not pts:
        return np.empty((0, 3)), np.empty((0, 3), np.uint8)
    P = np.concatenate(pts)
    C = np.concatenate(cols)
    keep = multiview_filter(P, counts, min_views, agree_radius)
    return P[keep], C[keep]


def interpolated_frames(frames, per_pair: int = 2, max_pairs: int = 40):
    """Extra viewpoints between nearby training cameras.

    The splatting model can be rendered from anywhere, and a surface seen at a
    grazing angle from every training pose may face a viewpoint between two of
    them squarely. Poses are interpolated between each camera and its nearest
    neighbours, with rotation via quaternion slerp so the intermediate poses are
    valid rather than merely averaged.

    These are the model's own extrapolations, so they are only useful where the
    metric asks about coverage rather than precision.
    """
    import numpy as np

    from ..dataio import Frame, rotmat2qvec, qvec2rotmat

    centres = np.stack([f.center for f in frames])
    out = []
    pairs = []
    for i, f in enumerate(frames):
        d = np.linalg.norm(centres - f.center, axis=1)
        d[i] = np.inf
        for j in np.argsort(d)[:2]:
            if i < j:
                pairs.append((i, int(j)))
    for i, j in pairs[:max_pairs]:
        a, b = frames[i], frames[j]
        qa, qb = a.qvec, b.qvec
        if qa @ qb < 0:
            qb = -qb
        for k in range(1, per_pair + 1):
            t = k / (per_pair + 1)
            q = (1 - t) * qa + t * qb
            q = q / np.linalg.norm(q)
            centre = (1 - t) * a.center + t * b.center
            R = qvec2rotmat(q)
            tvec = -R @ centre
            out.append(Frame(-1, f"interp_{i}_{j}_{k}", q, tvec, a.camera, None))
    return out
