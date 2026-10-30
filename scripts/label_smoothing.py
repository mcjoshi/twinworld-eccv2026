"""Evaluate spatial regularisation after semantic-label propagation.

The classifier assigns labels to a thinned feature cloud, then five-nearest-
neighbour voting transfers them to every point in the packaged submission
cloud.  This experiment changes only those transferred labels: the geometry,
classifier, propagation k, and scorer are fixed.

Pre-registered adoption gate, fixed before running: BOTH folds must beat their
own baseline (gold_coast/scene_009 > 0.1529 and gold_coast/scene_010 > 0.2067)
AND LOSO must exceed 0.1798.  Improving the mean alone is a REJECT.

The radius variants vote over every submitted-cloud point at the specified
radius.  ``confidence`` uses the 20 nearest neighbours: it keeps the
propagated label only when more than 60% of that neighbourhood agrees with it;
otherwise it uses the neighbourhood majority.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from twinworld.dataio import CLASS_NAMES, NUM_CLASSES, Scene
from twinworld.package import submission_cloud
from twinworld.scoring import read_ply_xyz, score_semantics
from twinworld.semantics.classifier import (
    ClassifierConfig, fit, predict, thinned_features, transfer_labels,
)

V1 = Path("experiments/2026-08-08-pipeline-v1")
OUT = Path("experiments/2026-08-21-label-smoothing")
PACK_VOXEL = 0.10
BASE = {"gold_coast/scene_009": 0.1529, "gold_coast/scene_010": 0.2067}
LOSO_GATE = 0.1798
K_PROP = 5
K_SMOOTH = 20
CHUNK = 200_000
VARIANTS = ("none", "majority_r15", "majority_r30", "majority_k20", "confidence")


def majority_from_indices(labels: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Return per-row majority labels, breaking ties consistently by class ID."""
    votes = np.zeros((len(indices), NUM_CLASSES), dtype=np.uint8)
    for klass in range(NUM_CLASSES):
        votes[:, klass] = (labels[indices] == klass).sum(axis=1)
    return np.argmax(votes, axis=1).astype(np.uint8)


def propagate_k(
    thin_xyz: np.ndarray,
    thin_labels: np.ndarray,
    full_xyz: np.ndarray,
    k: int = K_PROP,
) -> np.ndarray:
    """Transfer labels without materialising nearest-neighbour indices for the cloud."""
    tree = cKDTree(thin_xyz)
    out = np.empty(len(full_xyz), dtype=np.uint8)
    for start in range(0, len(full_xyz), CHUNK):
        end = min(start + CHUNK, len(full_xyz))
        _, indices = tree.query(full_xyz[start:end], k=k, workers=-1)
        out[start:end] = majority_from_indices(thin_labels, indices)
    return out


def radius_majority(
    tree: cKDTree,
    xyz: np.ndarray,
    labels: np.ndarray,
    radius: float,
) -> np.ndarray:
    """Vote over radius neighbours in bounded chunks to limit query memory."""
    out = np.empty(len(xyz), dtype=np.uint8)
    for start in range(0, len(xyz), CHUNK):
        end = min(start + CHUNK, len(xyz))
        neighbours = tree.query_ball_point(xyz[start:end], radius, workers=-1)
        for offset, indices in enumerate(neighbours):
            counts = np.bincount(labels[indices], minlength=NUM_CLASSES)
            out[start + offset] = np.argmax(counts)
    return out


def knn_majority(
    tree: cKDTree,
    xyz: np.ndarray,
    labels: np.ndarray,
    *,
    confidence: bool = False,
) -> np.ndarray:
    """Vote over 20 nearest neighbours, optionally retaining strong original votes."""
    out = np.empty(len(xyz), dtype=np.uint8)
    k = min(K_SMOOTH, len(xyz))
    for start in range(0, len(xyz), CHUNK):
        end = min(start + CHUNK, len(xyz))
        _, indices = tree.query(xyz[start:end], k=k, workers=-1)
        if k == 1:
            indices = indices[:, None]
        majority = majority_from_indices(labels, indices)
        if confidence:
            original = labels[start:end]
            agreement = (labels[indices] == original[:, None]).mean(axis=1)
            out[start:end] = np.where(agreement > 0.60, original, majority)
        else:
            out[start:end] = majority
    return out


def smooth_labels(variant: str, xyz: np.ndarray, labels: np.ndarray) -> np.ndarray:
    if variant == "none":
        return labels

    tree = cKDTree(xyz)
    if variant == "majority_r15":
        return radius_majority(tree, xyz, labels, 0.15)
    if variant == "majority_r30":
        return radius_majority(tree, xyz, labels, 0.30)
    if variant == "majority_k20":
        return knn_majority(tree, xyz, labels)
    if variant == "confidence":
        return knn_majority(tree, xyz, labels, confidence=True)
    raise ValueError(f"unknown smoothing variant: {variant}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = ClassifierConfig(max_feature_points=4_000_000, learning_rate=0.05)
    scenes = [Scene("gold_coast", scene_id) for scene_id in ("scene_009", "scene_010")]
    ply = {
        scene.key: V1 / scene.dataset / scene.scene_id / "mvs" / "fused_gs_i.ply"
        for scene in scenes
    }

    cache = {}
    for scene in scenes:
        full, _ = submission_cloud(scene, ply[scene.key], voxel=PACK_VOXEL)
        gt, gt_labels = read_ply_xyz(scene.gt_ply, with_classification=True)
        cache[scene.key] = (full, gt, gt_labels)
        print(f"[{scene.key}] submitted cloud: {len(full) / 1e6:.1f}M points", flush=True)

    features = {}
    for scene in scenes:
        thin, feature_matrix, _, _ = thinned_features(scene, ply[scene.key], cfg)
        gt, gt_labels = cache[scene.key][1:]
        train_labels = transfer_labels(gt, gt_labels, thin, cfg.label_transfer_radius)
        features[scene.key] = (thin, feature_matrix, train_labels)
        print(
            f"[{scene.key}] {len(thin)} feature points, "
            f"{int((train_labels < NUM_CLASSES).sum())} labelled",
            flush=True,
        )

    results = {
        "setup": {
            "scenes": [scene.key for scene in scenes],
            "cloud": "experiments/2026-08-08-pipeline-v1/"
            "gold_coast/<scene>/mvs/fused_gs_i.ply",
            "packaging_voxel": PACK_VOXEL,
            "classifier": {
                "max_feature_points": cfg.max_feature_points,
                "learning_rate": cfg.learning_rate,
            },
            "propagation_k": K_PROP,
            "smoothing_k": K_SMOOTH,
            "query_chunk_points": CHUNK,
            "confidence_agreement_threshold": 0.60,
            "gate": {
                "baselines": BASE,
                "loso": LOSO_GATE,
                "rule": "both folds beat their own baseline and LOSO exceeds the gate",
            },
        },
        "variants": {},
    }
    for variant in VARIANTS:
        fold = {}
        for held in scenes:
            started = time.time()
            train = next(scene for scene in scenes if scene.key != held.key)
            model = fit(features[train.key][1], features[train.key][2], cfg)
            thin, feature_matrix, _ = features[held.key]
            full, gt, gt_labels = cache[held.key]
            propagated = propagate_k(thin, predict(model, feature_matrix, n_jobs=cfg.n_jobs), full)
            smoothed = smooth_labels(variant, full, propagated)
            score = score_semantics(full, smoothed, gt, gt_labels)
            iou = score.per_class_iou
            fold[held.key] = {
                "miou": round(float(score.miou), 4),
                "ceiling": round(float(score.miou_ceiling), 4),
                "per_class": {
                    CLASS_NAMES[i]: round(float(iou[i]), 4) for i in range(NUM_CLASSES)
                },
                "seconds": round(time.time() - started, 1),
            }
            print(
                f"[{variant}][held {held.key}] mIoU {score.miou:.4f} "
                f"(base {BASE[held.key]:.4f}, {score.miou - BASE[held.key]:+.4f})",
                flush=True,
            )

        loso = float(np.mean([values["miou"] for values in fold.values()]))
        both_folds = all(fold[key]["miou"] > BASE[key] for key in BASE)
        verdict = "ADOPT" if both_folds and loso > LOSO_GATE else "REJECT"
        results["variants"][variant] = {
            "folds": fold,
            "loso_miou": round(loso, 4),
            "both_folds_beat_baseline": both_folds,
            "verdict": verdict,
        }
        print(
            f"[{variant}] LOSO {loso:.4f} vs gate {LOSO_GATE:.4f}; "
            f"both folds: {both_folds} -> {verdict}",
            flush=True,
        )

    (OUT / "results.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
