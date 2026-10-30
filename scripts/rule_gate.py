"""Gate physically grounded semantic-label overrides on the v16 push clouds.

The pre-registered gate is deliberately stricter than selecting the highest
mean: a candidate is adopted only when it improves each dev scene's baseline
and their arithmetic mean.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from twinworld.dataio import Scene
from twinworld.scoring import SemanticCounts, read_ply_xyz, score_semantics
from twinworld.semantics.features import _height_above_ground, estimate_up


INPUT = Path("experiments/2026-08-27-slab-clouds")
OUT = Path("experiments/2026-08-29-rules")
SCENES = ("scene_009", "scene_010")
GROUND_HEIGHTS = (0.2, 0.4)
UP_THRESHOLDS = (0.85, 0.92)
ROOF_MIN_HEIGHTS = (3.0, 5.0)


@dataclass(frozen=True)
class RuleConfig:
    name: str
    ground_height: float | None = None
    up_threshold: float | None = None
    roof_min_height: float | None = None


def normals(xyz: np.ndarray, k: int = 24) -> np.ndarray:
    """Local PCA normals, copied from offset_sheets.py."""
    tree = cKDTree(xyz)
    out = np.empty_like(xyz)
    chunk = 400_000
    kk = min(k, len(xyz))
    for start in range(0, len(xyz), chunk):
        stop = min(start + chunk, len(xyz))
        _, idx = tree.query(xyz[start:stop], k=kk, workers=-1)
        neighbours = xyz[idx]
        neighbours = neighbours - neighbours.mean(axis=1, keepdims=True)
        covariance = np.einsum("mki,mkj->mij", neighbours, neighbours) / kk
        _, vectors = np.linalg.eigh(covariance)
        out[start:stop] = vectors[:, :, 0]
    return out


def configs() -> list[RuleConfig]:
    sweep: list[RuleConfig] = []
    for height in GROUND_HEIGHTS:
        sweep.append(RuleConfig(name=f"ground_h{height:g}", ground_height=height))
    for threshold in UP_THRESHOLDS:
        for minimum in ROOF_MIN_HEIGHTS:
            sweep.append(
                RuleConfig(
                    name=f"roof_t{threshold:g}_hmin{minimum:g}",
                    up_threshold=threshold,
                    roof_min_height=minimum,
                )
            )
    for height in GROUND_HEIGHTS:
        for threshold in UP_THRESHOLDS:
            for minimum in ROOF_MIN_HEIGHTS:
                sweep.append(
                    RuleConfig(
                        name=f"both_h{height:g}_t{threshold:g}_hmin{minimum:g}",
                        ground_height=height,
                        up_threshold=threshold,
                        roof_min_height=minimum,
                    )
                )
    return sweep


def score_from_correspondence(
    pred_labels: np.ndarray,
    gt_labels: np.ndarray,
    nearest_indices: np.ndarray,
    matched: np.ndarray,
) -> SemanticCounts:
    """Apply score_semantics's fixed nearest-point correspondence to labels."""
    counts = SemanticCounts(n_total=len(gt_labels))
    valid = gt_labels != 255
    labels = gt_labels[valid]
    is_matched = matched[valid]
    transferred = np.full(len(labels), -1, dtype=np.int64)
    transferred[is_matched] = pred_labels[nearest_indices[valid][is_matched]]
    transferred[transferred == 255] = -1
    counts.n_unmatched = int((transferred < 0).sum())
    for class_id in range(5):
        gt_class = labels == class_id
        pred_class = transferred == class_id
        counts.gt_per_class[class_id] = int(gt_class.sum())
        counts.intersection[class_id] = int((gt_class & pred_class).sum())
        counts.union[class_id] = int((gt_class | pred_class).sum())
        counts.matched_per_class[class_id] = int((gt_class & (transferred >= 0)).sum())
    return counts


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    candidates = configs()
    results: dict[str, object] = {
        "gate": {
            "rule": "both scenes must beat their unmodified v16 baseline and mean must improve",
            "ground_height_m": list(GROUND_HEIGHTS),
            "up_threshold": list(UP_THRESHOLDS),
            "roof_min_height_m": list(ROOF_MIN_HEIGHTS),
            "normal_neighbours": 24,
            "ground_cell_m": 3.0,
            "ground_quantile": 0.02,
        },
        "baseline": {},
        "candidates": {},
    }
    scene_scores: dict[str, dict[str, float]] = {candidate.name: {} for candidate in candidates}
    baselines: dict[str, float] = {}

    for scene_id in SCENES:
        scene = Scene("gold_coast", scene_id)
        input_path = INPUT / f"{scene_id}_push_v16labeled.npz"
        with np.load(input_path) as cloud:
            xyz = cloud["xyz"].astype(np.float64)
            labels = cloud["labels"].astype(np.uint8)
        gt_xyz, gt_labels = read_ply_xyz(scene.gt_ply, with_classification=True)
        if gt_labels is None:
            raise ValueError(f"{scene.gt_ply} has no classification field")

        # The required baseline score is calculated with the public scorer.
        baseline_counts = score_semantics(xyz, labels, gt_xyz, gt_labels)
        baselines[scene.key] = baseline_counts.miou
        results["baseline"][scene.key] = {
            "miou": baseline_counts.miou,
            "per_class_iou": baseline_counts.per_class_iou.tolist(),
        }
        print(f"[{scene.key}] baseline v16 mIoU {baseline_counts.miou:.6f}", flush=True)

        # The geometry is independent of label overrides. Cache the scorer's
        # correspondence once; score_from_correspondence is algebraically the
        # same label-counting operation as score_semantics.
        distances, nearest_indices = cKDTree(xyz).query(gt_xyz, workers=-1)
        matched = distances <= 0.10
        up = estimate_up(scene.train_frames)
        hag, _ = _height_above_ground(xyz, up)
        normal = normals(xyz)
        horizontal = np.abs(normal @ up)
        ground_masks = {height: hag <= height for height in GROUND_HEIGHTS}
        roof_masks = {
            (threshold, minimum): (horizontal > threshold) & (hag > minimum)
            for threshold in UP_THRESHOLDS
            for minimum in ROOF_MIN_HEIGHTS
        }

        for candidate in candidates:
            override = labels.copy()
            ground_count = 0
            roof_count = 0
            if candidate.ground_height is not None:
                ground_mask = ground_masks[candidate.ground_height]
                ground_count = int((override[ground_mask] != 0).sum())
                override[ground_mask] = 0
            if candidate.up_threshold is not None and candidate.roof_min_height is not None:
                roof_mask = roof_masks[(candidate.up_threshold, candidate.roof_min_height)]
                roof_count = int((override[roof_mask] != 2).sum())
                override[roof_mask] = 2
            counts = score_from_correspondence(override, gt_labels, nearest_indices, matched)
            scene_scores[candidate.name][scene.key] = counts.miou
            results["candidates"][candidate.name] = results["candidates"].get(candidate.name, {})
            results["candidates"][candidate.name][scene.key] = {
                "miou": counts.miou,
                "delta_vs_baseline": counts.miou - baseline_counts.miou,
                "ground_relabels": ground_count,
                "roof_relabels": roof_count,
                "per_class_iou": counts.per_class_iou.tolist(),
            }
            print(
                f"  {candidate.name:26s} mIoU {counts.miou:.6f} "
                f"({counts.miou - baseline_counts.miou:+.6f})",
                flush=True,
            )

    print("\nFinal gate table")
    print(f"{'configuration':28s} {'scene_009':>11s} {'scene_010':>11s} {'mean':>11s}  decision")
    print(
        f"{'baseline_v16':28s} "
        f"{baselines['gold_coast/scene_009']:11.6f} "
        f"{baselines['gold_coast/scene_010']:11.6f} "
        f"{np.mean(list(baselines.values())):11.6f}  REFERENCE"
    )
    adopted: list[str] = []
    for candidate in candidates:
        scores = scene_scores[candidate.name]
        mean = float(np.mean(list(scores.values())))
        passes = all(scores[key] > baselines[key] for key in baselines) and mean > np.mean(
            list(baselines.values())
        )
        decision = "ADOPTED" if passes else "REJECTED"
        if passes:
            adopted.append(candidate.name)
        results["candidates"][candidate.name]["mean_miou"] = mean
        results["candidates"][candidate.name]["decision"] = decision
        print(
            f"{candidate.name:28s} "
            f"{scores['gold_coast/scene_009']:11.6f} "
            f"{scores['gold_coast/scene_010']:11.6f} "
            f"{mean:11.6f}  {decision}"
        )
    results["adopted"] = adopted
    results["conclusion"] = (
        f"ADOPTED: {', '.join(adopted)}" if adopted else "No configuration passed the pre-registered gate."
    )
    (OUT / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(results["conclusion"], flush=True)


if __name__ == "__main__":
    main()
