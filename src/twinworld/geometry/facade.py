"""Reconstruct facades by extruding them down from roof outlines.

Measured on Gold Coast scene_009, our reconstruction places roofs within 13 cm
of the ground truth but walls about a metre off, with only 1.7 percent of wall
points and 0.5 percent of window points covered at 10 cm. The walls are not
missing because they are unobservable (median incidence angle to the best-placed
training camera is 41 degrees); they are smeared along the viewing ray from the
roof edge towards the ground, which is the classic near-nadir MVS artefact.

The asymmetry is the opportunity. A building's walls are the vertical surfaces
directly beneath its roof outline, and the roof outline is the part we get right.
So instead of trying to make stereo work better on the facades, take the roof
outline we already trust and extrude it down to the local terrain.

This helps two of the five classes, wall and window, which together are 73
percent of Gold Coast points and 40 percent of the unweighted mIoU.

The output is a candidate cloud to union with the reconstruction, not a
replacement: it is a model of where facades should be, and it is only worth
keeping if it measurably raises coverage on the development scenes.
"""

from __future__ import annotations

import numpy as np


def _basis(up: np.ndarray):
    e1 = np.array([1.0, 0.0, 0.0])
    if abs(e1 @ up) > 0.9:
        e1 = np.array([0.0, 1.0, 0.0])
    e1 = e1 - (e1 @ up) * up
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(up, e1)


def height_grid(xyz: np.ndarray, up: np.ndarray, cell: float):
    """Rasterise the cloud into a horizontal grid of max and min height."""
    e1, e2 = _basis(up)
    h = xyz @ up
    u, v = xyz @ e1, xyz @ e2
    u0, v0 = u.min(), v.min()
    iu = np.floor((u - u0) / cell).astype(np.int64)
    iv = np.floor((v - v0) / cell).astype(np.int64)
    nu, nv = iu.max() + 1, iv.max() + 1

    top = np.full((nu, nv), -np.inf)
    bot = np.full((nu, nv), np.inf)
    np.maximum.at(top, (iu, iv), h)
    np.minimum.at(bot, (iu, iv), h)
    return top, bot, (u0, v0, nu, nv, e1, e2)


def terrain_height(bot: np.ndarray, occupied: np.ndarray, window: int = 12) -> np.ndarray:
    """Local ground level per cell: a min-filter over occupied cells.

    Buildings cover whole cells, so the ground under a building has to come from
    its surroundings rather than from the cell itself.
    """
    from scipy.ndimage import minimum_filter

    filled = np.where(occupied, bot, np.inf)
    ground = minimum_filter(filled, size=window, mode="nearest")
    # any cell still unresolved falls back to the scene's own low quantile
    finite = filled[np.isfinite(filled)]
    fallback = np.quantile(finite, 0.02) if finite.size else 0.0
    return np.where(np.isfinite(ground), ground, fallback)


def extrude_facades(
    xyz: np.ndarray,
    up: np.ndarray,
    cell: float = 0.30,
    min_building_height: float = 2.5,
    step_threshold: float = 1.5,
    point_spacing: float = 0.04,
    inward_shift: float = 0.5,
    max_points: int = 40_000_000,
):
    """Vertical surfaces beneath roof outlines and roof-height discontinuities.

    A cell is part of a building when its highest point sits `min_building_height`
    above the local terrain. A facade exists wherever a building cell borders a
    cell whose top is at least `step_threshold` lower: that covers both the
    outer walls (neighbour is ground) and internal walls between roof levels.

    Points are emitted along the vertical from the lower neighbour's top up to
    this cell's top, at `point_spacing`.

    They are placed on the shared edge between the two cells, then moved
    `inward_shift` cells back towards the building. Without that shift the wall
    sits on the outer boundary of the cell that straddles the true edge, so the
    error is uniform over a whole cell; centring it halves the median error, and
    the semantic match radius is only 10 cm.
    """
    top, bot, (u0, v0, nu, nv, e1, e2) = height_grid(xyz, up, cell)
    occupied = np.isfinite(top)
    ground = terrain_height(bot, occupied)

    height_above = np.where(occupied, top - ground, -np.inf)
    building = occupied & (height_above >= min_building_height)
    if not building.any():
        return np.empty((0, 3))

    # a neighbour's top, treating unoccupied cells as terrain
    neighbour_top = np.where(occupied, top, ground)

    segments = []  # (u_index, v_index, edge offset, bottom height, top height)
    for du, dv in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        src = building
        shifted = np.full_like(neighbour_top, -np.inf)
        sl_src = (
            slice(max(0, -du), nu - max(0, du)),
            slice(max(0, -dv), nv - max(0, dv)),
        )
        sl_dst = (
            slice(max(0, du), nu - max(0, -du)),
            slice(max(0, dv), nv - max(0, -dv)),
        )
        shifted[sl_dst] = neighbour_top[sl_src]

        drop = np.where(src, top - shifted, -np.inf)
        mask = src & (drop >= step_threshold) & np.isfinite(shifted)
        iu, iv = np.nonzero(mask)
        if len(iu) == 0:
            continue
        segments.append((iu, iv, du, dv, shifted[iu, iv], top[iu, iv]))

    if not segments:
        return np.empty((0, 3))

    out = []
    total = 0
    for iu, iv, du, dv, z_lo, z_hi in segments:
        # the wall sits on the boundary between the two cells
        cu = (iu + 0.5 + (0.5 - inward_shift) * du) * cell + u0
        cv = (iv + 0.5 + (0.5 - inward_shift) * dv) * cell + v0
        # sample across the shared edge as well as vertically, so the wall is a
        # surface rather than a line of points
        span = np.arange(-0.5, 0.5 + 1e-9, point_spacing / cell) * cell
        across_u = cu[:, None] + (span * abs(dv))[None, :]
        across_v = cv[:, None] + (span * abs(du))[None, :]

        n_v = np.maximum(1, np.ceil((z_hi - z_lo) / point_spacing).astype(np.int64))
        for k in range(int(n_v.max())):
            sel = k < n_v
            if not sel.any():
                break
            t = np.minimum(z_lo[sel] + k * point_spacing, z_hi[sel])
            pu = across_u[sel]
            pv = across_v[sel]
            m = pu.shape[1]
            pts = (
                pu.reshape(-1, 1) * e1
                + pv.reshape(-1, 1) * e2
                + np.repeat(t, m).reshape(-1, 1) * up
            )
            out.append(pts)
            total += len(pts)
            if total > max_points:
                return np.vstack(out)
    return np.vstack(out) if out else np.empty((0, 3))
