"""Per-point features for 5-class semantic labelling of Gold Coast clouds.

The classifier is trained on the reconstruction we actually submit, not on the
ground-truth cloud: labels are transferred from the GT cloud onto our own MVS
points by nearest neighbour, so training and inference see the same noise,
density and colour statistics. Training on GT geometry directly would leave a
domain gap exactly where it hurts.

Class balance measured on scene_009/010 is wall 56, roof 21, window 17,
ground 3, other 2 percent, and mIoU is unweighted, so the two rare classes are
40 percent of the score. Features are chosen to separate them:

  ground vs roof   both horizontal, separated by height above the local terrain
  wall vs window   both vertical, separated by colour and local colour contrast
  other            mostly vegetation and clutter, separated by local geometric
                   disorder (sphericity) and colour

Neighbourhoods are k-nearest rather than radius-based so every stage is a
batched array operation. Point clouds here run to millions of points and a
per-point Python loop does not finish in useful time.
"""

from __future__ import annotations

import numpy as np

DEFAULT_KS: tuple[int, ...] = (16, 48, 128)
CHUNK = 200_000


def estimate_up(frames) -> np.ndarray:
    """Up direction in world coordinates, from the mean camera viewing ray.

    The captures are near-nadir drone flights, so the mean viewing direction
    points down and its negation is a reliable up vector. This avoids assuming
    which COLMAP axis happens to be vertical.
    """
    dirs = np.stack([fr.R.T @ np.array([0.0, 0.0, 1.0]) for fr in frames])
    mean_dir = dirs.mean(axis=0)
    return -mean_dir / np.linalg.norm(mean_dir)


def _neighbourhood_features(xyz: np.ndarray, tree, k: int, up: np.ndarray, rig: np.ndarray):
    """Batched local structure tensor features for one neighbourhood size.

    Returns linearity, planarity, sphericity, verticality, log local extent, and
    signed offset from the local plane.

    The offset is the point's own displacement along the local surface normal,
    relative to its neighbourhood's centroid. At the larger neighbourhood sizes
    the plane is the facade, so a recessed window sits at a systematically
    negative offset while the surrounding wall sits near zero. Planarity cannot
    express this: a window pane and the wall around it are both planar.
    """
    n = len(xyz)
    lin = np.empty(n, np.float32)
    pla = np.empty(n, np.float32)
    sph = np.empty(n, np.float32)
    vert = np.empty(n, np.float32)
    scale = np.empty(n, np.float32)
    offset = np.empty(n, np.float32)

    kk = min(k, n)
    for a in range(0, n, CHUNK):
        b = min(a + CHUNK, n)
        d, idx = tree.query(xyz[a:b], k=kk, workers=-1)
        nb = xyz[idx]  # (m, k, 3)
        centroid = nb.mean(axis=1)
        P = nb - centroid[:, None, :]
        C = np.einsum("mki,mkj->mij", P, P) / kk
        w, V = np.linalg.eigh(C)  # ascending
        w = np.clip(w[:, ::-1], 1e-12, None)  # descending
        s = w.sum(axis=1)
        lin[a:b] = (w[:, 0] - w[:, 1]) / s
        pla[a:b] = (w[:, 1] - w[:, 2]) / s
        sph[a:b] = w[:, 2] / s
        normal = V[:, :, 0]  # eigenvector of the smallest eigenvalue
        vert[a:b] = 1.0 - np.abs(normal @ up)  # 0 horizontal surface, 1 vertical
        scale[a:b] = np.log1p(d[:, -1])
        # Orient each normal towards the camera rig so the sign of the offset
        # means the same thing everywhere: negative is further from the camera,
        # which is what a recessed window is. Orienting by the up vector instead
        # would be arbitrary on walls, where the normal is perpendicular to up.
        to_rig = rig - centroid
        flip = np.sign(np.einsum("mi,mi->m", to_rig, normal))
        flip[flip == 0] = 1.0
        offset[a:b] = np.einsum("mi,mi->m", xyz[a:b] - centroid, normal * flip[:, None])
    return lin, pla, sph, vert, scale, offset


def _height_above_ground(xyz: np.ndarray, up: np.ndarray, cell: float = 3.0, q: float = 0.02):
    """Height above a coarse terrain model.

    The terrain model is a low quantile of height per horizontal cell, then a
    min-filter over the 8-neighbourhood of cells. That is robust to buildings
    covering most of a cell and to the ground being only partly observed.
    """
    e1 = np.array([1.0, 0.0, 0.0])
    if abs(e1 @ up) > 0.9:
        e1 = np.array([0.0, 1.0, 0.0])
    e1 = e1 - (e1 @ up) * up
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)

    h = xyz @ up
    iu = np.floor((xyz @ e1 - (xyz @ e1).min()) / cell).astype(np.int64)
    iv = np.floor((xyz @ e2 - (xyz @ e2).min()) / cell).astype(np.int64)
    nv = iv.max() + 1
    keys = iu * nv + iv
    uniq, inv = np.unique(keys, return_inverse=True)

    order = np.argsort(inv, kind="stable")
    sorted_inv = inv[order]
    starts = np.searchsorted(sorted_inv, np.arange(len(uniq)))
    ends = np.append(starts[1:], len(order))
    hs = h[order]
    ground = np.array(
        [np.quantile(hs[a:b], q) if b - a > 4 else hs[a:b].min() for a, b in zip(starts, ends)]
    )

    # min-filter over neighbouring cells, via a dense grid indexed by cell id
    cu, cv = uniq // nv, uniq % nv
    gu, gv = cu.max() + 1, cv.max() + 1
    grid = np.full((gu + 2, gv + 2), np.inf)
    grid[cu + 1, cv + 1] = ground
    stack = np.stack(
        [grid[1 + du : gu + 1 + du, 1 + dv : gv + 1 + dv] for du in (-1, 0, 1) for dv in (-1, 0, 1)]
    )
    smooth_grid = stack.min(axis=0)
    smooth = smooth_grid[cu, cv]
    smooth = np.where(np.isfinite(smooth), smooth, ground)
    return (h - smooth[inv]).astype(np.float32), h.astype(np.float32)


def _height_above_roof(xyz: np.ndarray, up: np.ndarray, hag: np.ndarray,
                       horiz: np.ndarray, cell: float = 3.0, q: float = 0.60,
                       min_building_height: float = 4.0) -> np.ndarray:
    """Signed height relative to the local roof plane.

    Roof level per horizontal cell is a quantile of that cell's *horizontal*
    building points: facades are excluded by the horizontality mask (a first
    version that used all building points put the roof level mid-facade on
    building edges), and rooftop plant is excluded by being a minority above
    the quantile (a first version using q=0.9 placed the roof level at the top
    of the plant itself, zeroing the very signal the feature exists for).
    Non-building cells inherit the nearest building cell's level by iterative
    3x3 max-fill, so ground reads strongly negative rather than zero.
    """
    from scipy.ndimage import maximum_filter

    e1 = np.array([1.0, 0.0, 0.0])
    if abs(e1 @ up) > 0.9:
        e1 = np.array([0.0, 1.0, 0.0])
    e1 = e1 - (e1 @ up) * up
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)

    h = xyz @ up
    iu = np.floor((xyz @ e1 - (xyz @ e1).min()) / cell).astype(np.int64)
    iv = np.floor((xyz @ e2 - (xyz @ e2).min()) / cell).astype(np.int64)
    nv = iv.max() + 1
    keys = iu * nv + iv
    uniq, inv = np.unique(keys, return_inverse=True)

    order = np.argsort(inv, kind="stable")
    sinv = inv[order]
    starts = np.searchsorted(sinv, np.arange(len(uniq)))
    ends = np.append(starts[1:], len(order))
    hs, hags, hors = h[order], hag[order], horiz[order]
    roof = np.full(len(uniq), np.nan)
    for k, (a, b) in enumerate(zip(starts, ends)):
        m = (hags[a:b] >= min_building_height) & hors[a:b]
        if m.sum() >= 8:
            roof[k] = np.quantile(hs[a:b][m], q)

    cu, cv = uniq // nv, uniq % nv
    grid = np.full((cu.max() + 3, cv.max() + 3), -np.inf)
    grid[cu + 1, cv + 1] = np.where(np.isnan(roof), -np.inf, roof)
    # iterative max-fill until every cell inherits some building's roof level
    for _ in range(64):
        if np.isfinite(grid).all():
            break
        g2 = maximum_filter(grid, size=3, mode="nearest")
        grid = np.where(np.isfinite(grid), grid, g2)
    roof_per_cell = grid[cu + 1, cv + 1]
    # scene with no buildings at all: neutral zero
    roof_per_cell = np.where(np.isfinite(roof_per_cell), roof_per_cell, h.max())
    return (h - roof_per_cell[inv]).astype(np.float32)


def compute_features(
    xyz: np.ndarray,
    rgb: np.ndarray | None,
    up: np.ndarray,
    ks: tuple[int, ...] = DEFAULT_KS,
    rig: np.ndarray | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Return (N, F) float32 features and their names.

    `rig` is the mean training camera centre, used to give surface normals a
    consistent orientation. It defaults to far along the up direction, which is
    right for near-nadir capture but should be passed explicitly.
    """
    from scipy.spatial import cKDTree

    xyz = np.asarray(xyz, dtype=np.float64)
    if rig is None:
        rig = xyz.mean(axis=0) + up * 1000.0
    tree = cKDTree(xyz)
    feats: list[np.ndarray] = []
    names: list[str] = []

    vert_small = None
    for k in ks:
        lin, pla, sph, vert, scale, offset = _neighbourhood_features(xyz, tree, k, up, rig)
        if vert_small is None:
            vert_small = vert
        feats += [lin, pla, sph, vert, scale, offset]
        names += [f"lin_k{k}", f"pla_k{k}", f"sph_k{k}", f"vert_k{k}",
                  f"logextent_k{k}", f"planeoffset_k{k}"]

    hag, h = _height_above_ground(xyz, up)
    feats += [hag, (h - h.mean()).astype(np.float32)]
    names += ["height_above_ground", "height_centred"]

    # height_above_roof was tried here and REJECTED by the LOSO gate
    # (2026-08-11): despite perfect synthetic separation (plant +1.5, roof 0,
    # wall -6, ground -12), on real reconstructions it made both folds worse
    # (0.1846 -> 0.1783) and specifically halved the "other" IoU it was built
    # for. The roof-level estimate is apparently too noisy on smeared fused
    # clouds. The implementation is kept below for the paper's negative-results
    # table; do not re-enable without a new gate.

    if rgb is not None:
        c = np.asarray(rgb, dtype=np.float32) / 255.0
        mx, mn = c.max(1), c.min(1)
        inten = c.mean(1)
        feats += [c[:, 0], c[:, 1], c[:, 2], mx, mx - mn, inten,
                  2 * c[:, 1] - c[:, 0] - c[:, 2]]
        names += ["r", "g", "b", "value", "saturation", "intensity", "excess_green"]

        # local colour statistics at the smallest scale: windows are darker and
        # higher-contrast than the wall they sit in
        k0 = min(ks[0], len(xyz))
        loc_mean = np.empty(len(xyz), np.float32)
        loc_std = np.empty(len(xyz), np.float32)
        for a in range(0, len(xyz), CHUNK):
            b = min(a + CHUNK, len(xyz))
            _, idx = tree.query(xyz[a:b], k=k0, workers=-1)
            vals = inten[idx]
            loc_mean[a:b] = vals.mean(axis=1)
            loc_std[a:b] = vals.std(axis=1)
        feats += [inten - loc_mean, loc_std]
        names += ["intensity_minus_local", "local_intensity_std"]

    X = np.stack([np.asarray(f, np.float32) for f in feats], axis=1)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0), names


def transfer_labels(
    src_xyz: np.ndarray,
    src_labels: np.ndarray,
    dst_xyz: np.ndarray,
    radius: float = 0.15,
) -> np.ndarray:
    """Nearest-neighbour label transfer, with 255 where nothing is within `radius`.

    Used to put GT labels onto our own reconstruction so the classifier trains on
    the distribution it will be applied to.
    """
    from scipy.spatial import cKDTree

    dist, idx = cKDTree(src_xyz).query(dst_xyz, workers=-1)
    out = np.full(len(dst_xyz), 255, dtype=np.uint8)
    ok = dist <= radius
    out[ok] = np.asarray(src_labels)[idx[ok]]
    return out


def _expected_feature_names(ks: tuple[int, ...] = DEFAULT_KS, with_colour: bool = True):
    names: list[str] = []
    for k in ks:
        names += [f"lin_k{k}", f"pla_k{k}", f"sph_k{k}", f"vert_k{k}",
                  f"logextent_k{k}", f"planeoffset_k{k}"]
    names += ["height_above_ground", "height_centred"]
    if with_colour:
        names += ["r", "g", "b", "value", "saturation", "intensity", "excess_green",
                  "intensity_minus_local", "local_intensity_std"]
    return names


# Used to invalidate cached features when the feature set changes.
FEATURE_NAMES_EXPECTED = _expected_feature_names()
