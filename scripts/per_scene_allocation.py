"""Test whether Gold Coast's pooled semantic score merits per-scene voxels.

This deliberately measures oracle-labelled coverage, not classifier accuracy:
the voxel decision can only change which GT points find a submitted point.  The
oracle makes the resulting intersection/union changes attributable solely to
packaging density.
"""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import numpy as np

from twinworld.dataio import CLASS_NAMES, NUM_CLASSES, Scene
from twinworld.scoring import (
    SemanticCounts,
    read_ply_xyz,
    score_semantics,
    voxel_downsample_indices,
)

RUN = Path("experiments/2026-08-08-pipeline-v1")
OUT = Path("experiments/2026-08-21-per-scene/results.json")
DEV = ("scene_009", "scene_010")
FINAL = ("scene_011", "scene_012")
VOXELS = (0.06, 0.10, 0.14, 0.20)
BUDGETS = (45_000_000, 60_000_000, 75_000_000)
REFERENCE_VOXEL = 0.06


def voxel_clouds(scene: Scene) -> dict[float, np.ndarray]:
    """Derive all candidates from the existing 6 cm package cloud in one read."""
    ply = RUN / scene.dataset / scene.scene_id / "mvs" / "fused_gs_i.ply"
    # Reuse only pre-existing pipeline caches.  Passing a new cache path reaches
    # an unrelated package-module cache writer that is unavailable in this run.
    cached = ply.parent / f"submission_cloud_gs_i_v{REFERENCE_VOXEL}.npz"
    if not cached.exists():
        raise FileNotFoundError(f"required existing package cache not found: {cached}")
    base = np.load(cached)["xyz"].astype(np.float64)
    return {
        voxel: base if voxel == REFERENCE_VOXEL else base[voxel_downsample_indices(base, voxel)]
        for voxel in VOXELS
    }


def oracle_counts(pred: np.ndarray, gt_xyz: np.ndarray, gt_labels: np.ndarray) -> SemanticCounts:
    """Label each submitted point from nearby GT so only coverage changes."""
    from scipy.spatial import cKDTree

    distance, index = cKDTree(gt_xyz).query(pred, distance_upper_bound=0.10, workers=-1)
    labels = np.full(len(pred), 255, dtype=np.uint8)
    matched = np.isfinite(distance)
    labels[matched] = gt_labels[index[matched]]
    return score_semantics(pred, labels, gt_xyz, gt_labels)


def counts_dict(counts: SemanticCounts) -> dict:
    return {
        "intersection": {CLASS_NAMES[i]: int(counts.intersection[i]) for i in range(NUM_CLASSES)},
        "union": {CLASS_NAMES[i]: int(counts.union[i]) for i in range(NUM_CLASSES)},
        "gt_per_class": {CLASS_NAMES[i]: int(counts.gt_per_class[i]) for i in range(NUM_CLASSES)},
        "matched_per_class": {CLASS_NAMES[i]: int(counts.matched_per_class[i]) for i in range(NUM_CLASSES)},
        "oracle_miou_ceiling": float(counts.miou),
    }


def frustum_solid_angle(scene: Scene) -> float:
    """Sum each training camera's rectangular-frustum solid angle (steradians)."""
    total = 0.0
    for frame in scene.train_frames:
        x = np.arctan(frame.camera.width / (2 * frame.camera.fx))
        y = np.arctan(frame.camera.height / (2 * frame.camera.fy))
        total += 4 * np.arcsin(np.sin(x) * np.sin(y))
    return float(total)


def proxy_values(scene: Scene, xyz: np.ndarray) -> dict[str, float]:
    extent = xyz.max(axis=0) - xyz.min(axis=0)
    return {
        "reconstructed_points_v0.06": float(len(xyz)),
        "reconstruction_bbox_volume_m3": float(np.prod(extent)),
        "train_image_count": float(len(scene.train_frames)),
        "train_frustum_solid_angle_sr": frustum_solid_angle(scene),
    }


def ratio_error(a: float, b: float, gt_a: float, gt_b: float) -> float:
    """Absolute log error of the scene-010/scene-009 size ratio."""
    return float(abs(np.log((b / a) / (gt_b / gt_a))))


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    dev_scenes = [Scene("gold_coast", sid) for sid in DEV]
    final_scenes = [Scene("gold_coast", sid) for sid in FINAL]
    dev_reference, final_reference = {}, {}
    gt = {}

    # Reference clouds provide all GT-free proxies and final point-cost curves.
    for scene in [*dev_scenes, *final_scenes]:
        candidates = voxel_clouds(scene)
        (dev_reference if scene.scene_id in DEV else final_reference)[scene.scene_id] = candidates
    for scene in dev_scenes:
        gt[scene.scene_id] = read_ply_xyz(scene.gt_ply, with_classification=True)

    measurements = {}
    mean_coverage = {}
    for voxel in VOXELS:
        pooled = SemanticCounts()
        per_scene = {}
        for scene in dev_scenes:
            pred = dev_reference[scene.scene_id][voxel]
            counts = oracle_counts(pred, *gt[scene.scene_id])
            pooled.add(counts)
            per_scene[scene.scene_id] = {"points": int(len(pred)), **counts_dict(counts)}
        measurements[str(voxel)] = {
            "pooled": counts_dict(pooled),
            "per_scene": per_scene,
            "points_total": sum(row["points"] for row in per_scene.values()),
        }
        mean_coverage[voxel] = float(pooled.miou)

    actual_gt = {sid: int(len(gt[sid][0])) for sid in DEV}
    dev_proxies = {
        scene.scene_id: proxy_values(scene, dev_reference[scene.scene_id][REFERENCE_VOXEL])
        for scene in dev_scenes
    }
    final_proxies = {
        scene.scene_id: proxy_values(scene, final_reference[scene.scene_id][REFERENCE_VOXEL])
        for scene in final_scenes
    }
    proxy_report = {}
    for name in next(iter(dev_proxies.values())):
        a, b = dev_proxies[DEV[0]][name], dev_proxies[DEV[1]][name]
        proxy_report[name] = {
            "scene_009": a,
            "scene_010": b,
            "scene_010_over_scene_009": b / a,
            "ground_truth_010_over_009": actual_gt[DEV[1]] / actual_gt[DEV[0]],
            "absolute_log_ratio_error": ratio_error(a, b, actual_gt[DEV[0]], actual_gt[DEV[1]]),
        }
    chosen_proxy = min(proxy_report, key=lambda name: proxy_report[name]["absolute_log_ratio_error"])

    # Count candidates derived from the existing 6 cm package cloud.  This is
    # exact at 6 cm and a conservative coarsening for the larger voxels.
    final_costs = {
        scene.scene_id: {voxel: int(len(final_reference[scene.scene_id][voxel])) for voxel in VOXELS}
        for scene in final_scenes
    }
    weights = np.array([final_proxies[sid][chosen_proxy] for sid in FINAL], dtype=np.float64)
    weights /= weights.sum()
    recommendations = {}
    for budget in BUDGETS:
        feasible = []
        for v011, v012 in product(VOXELS, repeat=2):
            points = final_costs["scene_011"][v011] + final_costs["scene_012"][v012]
            if points <= budget:
                utility = weights[0] * mean_coverage[v011] + weights[1] * mean_coverage[v012]
                feasible.append((utility, points, v011, v012))
        best = max(feasible)
        uniform = max(
            ((mean_coverage[v], final_costs["scene_011"][v] + final_costs["scene_012"][v], v)
             for v in VOXELS
             if final_costs["scene_011"][v] + final_costs["scene_012"][v] <= budget),
            default=None,
        )
        recommendations[str(budget)] = {
            "scene_011_voxel_m": best[2],
            "scene_012_voxel_m": best[3],
            "points": best[1],
            "proxy_weighted_oracle_ceiling": best[0],
            "best_uniform": None if uniform is None else {
                "voxel_m": uniform[2], "points": uniform[1], "oracle_ceiling": uniform[0],
            },
            "gain_over_best_uniform": None if uniform is None else best[0] - uniform[0],
        }

    output = {
        "method": {
            "packaging_cloud": str(RUN / "gold_coast/<scene>/mvs/fused_gs_i.ply"),
            "voxel_measurement": "Oracle labels isolate packaging coverage; score_semantics and "
                                 "SemanticCounts provide the reported pooled counts.",
            "warning": "Proxy evaluation has n=2 labelled scenes. It is weak evidence only; "
                       "the selected proxy must not be treated as a validated predictor.",
        },
        "dev_ground_truth_points": actual_gt,
        "pooled_count_scaling": measurements,
        "proxy_evaluation_at_voxel_m": REFERENCE_VOXEL,
        "proxy_evaluation": proxy_report,
        "chosen_proxy": chosen_proxy,
        "final_proxy_values": final_proxies,
        "final_package_point_costs": final_costs,
        "recommendations": recommendations,
    }
    OUT.write_text(json.dumps(output, indent=2))
    print(json.dumps(output["recommendations"], indent=2))


if __name__ == "__main__":
    main()
