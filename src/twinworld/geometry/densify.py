"""Post-processing that turns an MVS cloud into a better-scoring submission cloud.

Two separate problems, measured on TUM scene_000 with the `auto` MVS variant:

  precision  F@5 is only 0.18 while F@20 is 0.72, so at fine thresholds the
             points are in roughly the right place but not accurately enough
  recall     0.45 at 10 cm; the ground truth is a dense scan containing facades
             that 12 near-nadir views never observe

Poisson reconstruction addresses both at once: it fits a surface through the
oriented points, which averages away per-point depth noise, and it interpolates
across holes small enough to be bridged. It also invents surface far from any
observation, so the result is trimmed back to the neighbourhood of the input
cloud, and separately confined to what the training views could see.
"""

from __future__ import annotations

import numpy as np


def read_cloud_with_normals(path):
    """COLMAP's `stereo_fusion` writes normals, which Poisson needs."""
    from plyfile import PlyData

    v = PlyData.read(str(path))["vertex"]
    names = {p.name for p in v.properties}
    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    nrm = (
        np.stack([v["nx"], v["ny"], v["nz"]], 1).astype(np.float64)
        if {"nx", "ny", "nz"} <= names
        else None
    )
    rgb = (
        np.stack([v["red"], v["green"], v["blue"]], 1).astype(np.uint8)
        if {"red", "green", "blue"} <= names
        else None
    )
    return xyz, nrm, rgb


def robust_core(xyz: np.ndarray, frames, k: float = 3.0, extra=None):
    """The compact part of an MVS cloud, as a mask.

    COLMAP fusion emits a small tail of far-field points: on TUM scene_000, 0.1
    percent of points stretch the bounding box from ~150 m to 9.4 km. Poisson
    reconstruction grids the bounding box, so that tail alone turns 4 cm cells
    into 4 m cells and destroys the fit.

    Cropping by per-axis percentiles was measured and rejected: on every
    development scene it cut inside the ground-truth bounding box. Distance from
    the camera rig is safe instead, because the ground truth sits where the
    cameras converge. `k = 3` times the median leaves 1.9x headroom over the
    furthest ground-truth point on scene_000 and 1.8x on scene_001.

    Nothing is lost by this crop: densification runs on the core and the result
    is unioned back with the full raw cloud.
    """
    rig = np.stack([fr.center for fr in frames]).mean(axis=0)
    d = np.linalg.norm(xyz - rig, axis=1)
    return d <= k * float(np.median(d))


def choose_poisson_depth(xyz: np.ndarray, target_cell: float = 0.04, cap: int = 13) -> int:
    """Octree depth that puts the finest Poisson cell near `target_cell` metres."""
    extent = float(np.max(xyz.max(axis=0) - xyz.min(axis=0)))
    return int(min(cap, max(8, np.ceil(np.log2(max(extent, 1e-6) / target_cell)))))


def poisson_densify(
    xyz: np.ndarray,
    normals: np.ndarray | None,
    rgb: np.ndarray | None = None,
    depth: int | None = None,
    density_quantile: float = 0.10,
    trim_distance: float = 0.15,
    sample_spacing: float = 0.02,
    max_samples: int = 12_000_000,
    target_cell: float = 0.04,
):
    """Fit a surface, resample it densely, and trim it back to the evidence.

    `depth=None` picks the octree depth from the cloud's own extent so the cell
    size lands near `target_cell`, which matters because these scenes differ in
    size by an order of magnitude.

    `density_quantile` drops the lowest-support vertices of the Poisson mesh;
    `trim_distance` then removes any sampled point further than that from an
    original MVS point, which is what stops the reconstruction from inventing
    geometry across large unobserved gaps.
    """
    import open3d as o3d
    from scipy.spatial import cKDTree

    if depth is None:
        depth = choose_poisson_depth(xyz, target_cell)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals)
    else:
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=24))
    if rgb is not None:
        pcd.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64) / 255.0)

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, linear_fit=False
    )
    densities = np.asarray(densities)
    if density_quantile > 0:
        keep = densities >= np.quantile(densities, density_quantile)
        mesh.remove_vertices_by_mask(~keep)

    area = float(mesh.get_surface_area())
    n = int(min(max_samples, max(1000, area / (sample_spacing**2))))
    sampled = mesh.sample_points_uniformly(number_of_points=n, use_triangle_normal=False)
    pts = np.asarray(sampled.points)
    cols = (np.asarray(sampled.colors) * 255).astype(np.uint8) if sampled.has_colors() else None

    d, _ = cKDTree(xyz).query(pts, workers=-1)
    m = d <= trim_distance
    return pts[m], (cols[m] if cols is not None else None), {
        "mesh_area_m2": round(area, 1),
        "sampled": n,
        "kept_after_trim": int(m.sum()),
        "depth": depth,
        "core_extent_m": [round(float(v), 1) for v in (xyz.max(axis=0) - xyz.min(axis=0))],
        "density_quantile": density_quantile,
        "trim_distance": trim_distance,
        "sample_spacing": sample_spacing,
    }


def confine_to_training_views(scene, xyz: np.ndarray, extra=None, margin_px: float = 0.0):
    """Drop anything no training camera could have seen.

    Measured on the six development scenes, training frustums contain 97.2-100
    percent of ground-truth points, so this costs almost no recall while
    removing Poisson surface extrapolated outside the observed region.
    """
    m = scene.train_frustum_mask(xyz, margin_px=margin_px)
    if extra is None:
        return xyz[m], None
    return xyz[m], extra[m]
