"""Gate per-class log-bias for the current fusion + ADE slab-push pipeline.

For each LOSO fold, the 2D segmenter and LightGBM model are trained only on the
other scene.  The log-bias is then coordinate-ascent tuned on that training
scene's transferred labels and frozen before it is applied to the held scene.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from twinworld.dataio import CLASS_NAMES, NUM_CLASSES, Scene
from twinworld.scoring import read_ply_xyz, score_semantics
from twinworld.semantics.classifier import (
    ClassifierConfig,
    fit,
    thinned_features,
    transfer_labels,
)
from ade_fusion import ade_maps, grouped_votes
from seg2d_fusion import vote_features
from seg2d_lift import predict_views, train_segmenter


V1 = Path("experiments/2026-08-08-pipeline-v1")
OUT = Path("experiments/2026-08-29-eiou")
BASE = {"gold_coast/scene_009": 0.1529, "gold_coast/scene_010": 0.2067}
INCUMBENT = 0.2498


def predict_proba(model, X: np.ndarray, batch: int = 500_000) -> np.ndarray:
    out = np.empty((len(X), NUM_CLASSES), np.float32)
    for start in range(0, len(X), batch):
        out[start:start + batch] = model.predict_proba(X[start:start + batch])
    return out


def pooled_miou(pred: np.ndarray, labels: np.ndarray) -> float:
    """Five-class IoU mean on labelled transferred points, including empty IoUs."""
    valid = labels < NUM_CLASSES
    pred, labels = pred[valid], labels[valid]
    ious = []
    for cls in range(NUM_CLASSES):
        union = np.count_nonzero((pred == cls) | (labels == cls))
        intersection = np.count_nonzero((pred == cls) & (labels == cls))
        ious.append(intersection / union if union else 0.0)
    return float(np.mean(ious))


def tune_log_bias(log_proba: np.ndarray, labels: np.ndarray, iterations: int = 3):
    """Coordinate-ascent a per-class additive log-probability bias."""
    bias = np.zeros(NUM_CLASSES, np.float32)
    grid = np.linspace(-2.5, 2.5, 21, dtype=np.float32)
    best = pooled_miou(np.argmax(log_proba, axis=1), labels)
    for _ in range(iterations):
        improved = False
        for cls in range(NUM_CLASSES):
            best_cls, best_value = bias[cls], best
            for value in grid:
                candidate = bias.copy()
                candidate[cls] = value
                score = pooled_miou(np.argmax(log_proba + candidate, axis=1), labels)
                if score > best_value:
                    best_cls, best_value = value, score
            if best_value > best:
                bias[cls], best = best_cls, best_value
                improved = True
        if not improved:
            break
    return bias, best


def propagate_k5(thin: np.ndarray, thin_labels: np.ndarray, full: np.ndarray) -> np.ndarray:
    """Propagate the class decision by an unweighted five-nearest-neighbour vote."""
    _, neighbours = cKDTree(thin).query(full, k=5, workers=-1)
    neighbour_labels = thin_labels[neighbours]
    votes = np.zeros((len(full), NUM_CLASSES), np.uint8)
    for cls in range(NUM_CLASSES):
        votes[:, cls] = np.count_nonzero(neighbour_labels == cls, axis=1)
    return np.argmax(votes, axis=1).astype(np.uint8)


def score_row(full, labels, gt, gt_labels):
    score = score_semantics(full, labels, gt, gt_labels)
    return {
        "miou": float(score.miou),
        "per_class": {
            CLASS_NAMES[i]: float(score.per_class_iou[i]) for i in range(NUM_CLASSES)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=3)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--ade-max-side", type=int, default=1024,
                        help="cap ADE inference resolution for a memory-occupied GPU")
    parser.add_argument("--tag", default="eiou")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}")
    scenes = [Scene("gold_coast", scene_id) for scene_id in ("scene_009", "scene_010")]
    cfg = ClassifierConfig(max_feature_points=4_000_000, learning_rate=0.05)

    geo = {}
    for scene in scenes:
        ply = V1 / scene.dataset / scene.scene_id / "mvs" / "fused_gs_i.ply"
        thin, features, _, _ = thinned_features(scene, ply, cfg)
        gt, gt_labels = read_ply_xyz(scene.gt_ply, with_classification=True)
        transferred = transfer_labels(gt, gt_labels, thin, cfg.label_transfer_radius)
        ade = grouped_votes(scene, thin, ade_maps(scene, device, args.ade_max_side))
        full = np.load(
            f"experiments/2026-08-27-slab-clouds/{scene.scene_id}_push.npz"
        )["xyz"].astype(np.float64)
        geo[scene.key] = {
            "thin": thin,
            "features": features,
            "labels": transferred,
            "ade": ade,
            "full": full,
            "gt": gt,
            "gt_labels": gt_labels,
        }
        print(f"[{scene.key}] ready: {len(thin):,} feature points", flush=True)

    folds = {}
    for held in scenes:
        started = time.time()
        train = next(scene for scene in scenes if scene.key != held.key)
        train_geo, held_geo = geo[train.key], geo[held.key]
        print(f"\n=== train {train.key}; held {held.key} ===", flush=True)
        segmenter, mean, std = train_segmenter(
            train, device, args.steps, 512, 6e-5, "nvidia/mit-b0"
        )
        vote = {
            scene.key: vote_features(
                scene, geo[scene.key]["thin"], predict_views(segmenter, scene, device, mean, std)
            )
            for scene in scenes
        }
        del segmenter
        torch.cuda.empty_cache()

        X_train = np.hstack([train_geo["features"], vote[train.key], train_geo["ade"]])
        X_held = np.hstack([held_geo["features"], vote[held.key], held_geo["ade"]])
        model = fit(X_train, train_geo["labels"], cfg)
        log_train = np.log(np.clip(predict_proba(model, X_train), 1e-9, 1.0))
        bias, train_miou = tune_log_bias(log_train, train_geo["labels"])
        log_held = np.log(np.clip(predict_proba(model, X_held), 1e-9, 1.0))

        argmax = score_row(
            held_geo["full"],
            propagate_k5(held_geo["thin"], np.argmax(log_held, axis=1).astype(np.uint8), held_geo["full"]),
            held_geo["gt"],
            held_geo["gt_labels"],
        )
        biased = score_row(
            held_geo["full"],
            propagate_k5(
                held_geo["thin"], np.argmax(log_held + bias, axis=1).astype(np.uint8), held_geo["full"]
            ),
            held_geo["gt"],
            held_geo["gt_labels"],
        )
        shifts = {
            name: biased["per_class"][name] - argmax["per_class"][name]
            for name in list(argmax["per_class"].keys())
        }
        folds[held.key] = {
            "train_scene": train.key,
            "tuned_log_bias": [float(value) for value in bias],
            "training_transferred_miou": train_miou,
            "argmax": argmax,
            "biased": biased,
            "per_class_iou_shift": shifts,
            "seconds": time.time() - started,
        }
        print(
            f"[held {held.key}] argmax {argmax['miou']:.4f}; "
            f"biased {biased['miou']:.4f}; bias {np.round(bias, 3).tolist()}",
            flush=True,
        )

    loso = float(np.mean([fold["biased"]["miou"] for fold in folds.values()]))
    both_folds = all(folds[key]["biased"]["miou"] > baseline for key, baseline in BASE.items())
    verdict = "ADOPT" if both_folds and loso > INCUMBENT else "REJECT"
    result = {
        "tag": args.tag,
        "pipeline": "fused_gs_i + mit-b0 vote_features + ADE grouped_votes + slab_push k=5",
        "bias_tuning": "coordinate ascent on training-scene transferred labels only",
        "baselines": BASE,
        "incumbent_loso": INCUMBENT,
        "loso_miou": loso,
        "both_folds_beat_baseline": both_folds,
        "verdict": verdict,
        "folds": folds,
    }
    (OUT / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"\n[{args.tag}] LOSO {loso:.4f} vs {INCUMBENT:.4f} -> {verdict}", flush=True)


if __name__ == "__main__":
    main()
