"""5-class semantic labelling of Gold Coast reconstructions.

Trained on our own reconstruction of scene_009/010 with labels transferred from
the ground-truth clouds, so training and inference share the same noise and
density. The two rare classes (ground 3 percent, other 2 percent) carry 40
percent of the unweighted mIoU, so training is class-balanced and the decision
rule is tuned on the metric rather than on accuracy.
"""

from __future__ import annotations

import os

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from ..dataio import NUM_CLASSES, Scene
from .features import (
    FEATURE_NAMES_EXPECTED, compute_features, estimate_up, transfer_labels,
)


@dataclass
class ClassifierConfig:
    # Sized to the data, not to habit. Only about 70 000 points per scene fall
    # within the label-transfer radius of any ground truth, so 600 estimators
    # across 5 classes (3000 trees) was fitting a model far larger than the
    # signal supports, and predicting it over millions of points is what made
    # this stage appear to hang.
    num_leaves: int = 31
    n_estimators: int = 200
    learning_rate: float = 0.10
    min_child_samples: int = 60
    subsample: float = 0.7
    subsample_freq: int = 1
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.0
    max_train_points: int = 1_500_000
    # One thread, not many. This machine runs at a load average well above its
    # 96 cores while the GPUs are busy, and LightGBM's OpenMP barriers spin-wait:
    # measured on a 20k fit, n_jobs=1 took 1.1 s and n_jobs=4 took 3.3 s. With
    # n_jobs=-1 a single fold did not finish in three hours. Parallelism here is
    # actively harmful, so the stage is single-threaded and cheap instead.
    n_jobs: int = 1
    label_transfer_radius: float = 0.15
    class_weighting: str = "balanced"  # "balanced" | "sqrt" | "none"
    # Features are computed on a thinned copy of the cloud and the labels are
    # then propagated back by nearest neighbour. A k=128 neighbourhood query
    # over 18.5 M points did not finish in two hours, and the extra points
    # cannot change any label: semantics is decided at a 10 cm match radius,
    # far coarser than the spacing being thrown away.
    feature_voxel: float = 0.15
    max_feature_points: int = 1_500_000
    seed: int = 0


def _class_weights(y: np.ndarray, mode: str) -> np.ndarray:
    counts = np.bincount(y, minlength=NUM_CLASSES).astype(np.float64)
    counts[counts == 0] = 1.0
    if mode == "balanced":
        w = counts.sum() / (NUM_CLASSES * counts)
    elif mode == "sqrt":
        w = np.sqrt(counts.sum() / (NUM_CLASSES * counts))
    else:
        w = np.ones(NUM_CLASSES)
    return w


def _cache_path(cloud_ply: Path) -> Path:
    return Path(cloud_ply).parent / "features.npz"


def cached_features(scene: Scene, cloud_ply: Path, use_cache: bool = True):
    """Features for a cloud, cached beside it.

    Feature extraction is a few minutes per Gold Coast scene, and the same
    features are needed for cross-validation, for fitting the final model, and
    for inference. The cache is keyed on the cloud's path and point count so a
    regenerated cloud invalidates it.
    """
    from ..package import submission_cloud

    cache = _cache_path(cloud_ply)
    # Work on the cloud that is actually submitted, not the raw fusion output.
    # The 2x Gold Coast clouds are 23-31 M points and collapse to a fraction of
    # that once the far-field tail is dropped and the 2 cm voxel is applied, and
    # points that never reach the scorer should never reach the classifier
    # either. It is also the same distribution the labels are transferred onto,
    # which is the point of training on our own reconstruction.
    cp = Path(cloud_ply)
    xyz, rgb = submission_cloud(scene, cp, cache=cp.parent / f"{cp.stem}_subcloud.npz")
    if use_cache and cache.exists():
        d = np.load(cache, allow_pickle=True)
        # validate on the feature names too: adding a feature must invalidate a
        # cache written before it existed, or the model is fed the wrong columns
        if int(d["n"]) == len(xyz) and list(d["names"]) == list(FEATURE_NAMES_EXPECTED):
            return xyz, d["X"], list(d["names"])

    up = estimate_up(scene.train_frames)
    rig = np.stack([fr.center for fr in scene.train_frames]).mean(axis=0)
    X, feat_names = compute_features(xyz, rgb, up, rig=rig)
    if use_cache:
        np.savez_compressed(cache, X=X, names=np.array(feat_names), n=len(xyz))
    return xyz, X, feat_names


def thinned_features(scene: Scene, cloud_ply: Path, cfg: "ClassifierConfig"):
    """Features on a thinned copy of the submission cloud.

    Returns (thin_xyz, X, names, full_xyz) so callers can propagate labels back
    to every submitted point.
    """
    from ..package import submission_cloud
    from ..scoring import voxel_downsample_indices

    cp = Path(cloud_ply)
    full_xyz, rgb = submission_cloud(scene, cp, cache=cp.parent / f"{cp.stem}_subcloud.npz")
    idx = voxel_downsample_indices(full_xyz, cfg.feature_voxel)
    if len(idx) > cfg.max_feature_points:
        idx = np.sort(np.random.default_rng(cfg.seed).choice(
            idx, cfg.max_feature_points, replace=False))
    xyz = full_xyz[idx]
    rgb_t = rgb[idx] if rgb is not None else None

    # Cache key must include the feature budget as well as the voxel: two
    # configs sharing a voxel but differing in budget produce different
    # thinnings, and concurrent runs writing one path corrupt the file.
    cache = cp.parent / f"{cp.stem}_features_v{cfg.feature_voxel}_n{cfg.max_feature_points}.npz"
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        if int(d["n"]) == len(xyz) and list(d["names"]) == list(FEATURE_NAMES_EXPECTED):
            return xyz, d["X"], list(d["names"]), full_xyz

    up = estimate_up(scene.train_frames)
    rig = np.stack([fr.center for fr in scene.train_frames]).mean(axis=0)
    X, names = compute_features(xyz, rgb_t, up, rig=rig)
    # Atomic write: concurrent runs must never observe a partial file. Write to
    # a process-unique temporary and rename, which is atomic within a filesystem.
    tmp = cache.with_suffix(f".tmp{os.getpid()}.npz")
    np.savez_compressed(tmp, X=X, names=np.array(names), n=len(xyz))
    os.replace(tmp, cache)
    return xyz, X, names, full_xyz


def propagate(thin_xyz: np.ndarray, thin_labels: np.ndarray, full_xyz: np.ndarray) -> np.ndarray:
    """Give every submitted point the label of its nearest classified point."""
    from scipy.spatial import cKDTree

    _, idx = cKDTree(thin_xyz).query(full_xyz, workers=-1)
    return thin_labels[idx]


def build_training_set(
    scene: Scene, cloud_ply: Path, cfg: ClassifierConfig
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Features and transferred labels for one labelled dev scene."""
    from plyfile import PlyData

    gv = PlyData.read(str(scene.gt_ply))["vertex"]
    gt_xyz = np.stack([gv["x"], gv["y"], gv["z"]], 1).astype(np.float64)
    gt_lab = np.asarray(gv["classification"]).astype(np.int32)

    xyz, X, feat_names = cached_features(scene, cloud_ply)
    y = transfer_labels(gt_xyz, gt_lab, xyz, cfg.label_transfer_radius)
    return X, y, feat_names


def features_for_cloud(scene: Scene, cloud_ply: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Features for a cloud we are going to label. Returns (xyz, X, names)."""
    return cached_features(scene, cloud_ply)


def fit(X: np.ndarray, y: np.ndarray, cfg: ClassifierConfig):
    """Fit on points whose transferred label is a real class (255 dropped)."""
    import lightgbm as lgb

    m = y < NUM_CLASSES
    X, y = X[m], y[m].astype(np.int32)
    rng = np.random.default_rng(cfg.seed)
    if len(X) > cfg.max_train_points:
        sel = rng.choice(len(X), cfg.max_train_points, replace=False)
        X, y = X[sel], y[sel]

    w = _class_weights(y, cfg.class_weighting)[y]
    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=NUM_CLASSES,
        num_leaves=cfg.num_leaves,
        n_estimators=cfg.n_estimators,
        learning_rate=cfg.learning_rate,
        min_child_samples=cfg.min_child_samples,
        subsample=cfg.subsample,
        subsample_freq=cfg.subsample_freq,
        colsample_bytree=cfg.colsample_bytree,
        reg_lambda=cfg.reg_lambda,
        random_state=cfg.seed,
        n_jobs=cfg.n_jobs,
        force_row_wise=True,
        verbose=-1,
    )
    model.fit(X, y, sample_weight=w)
    return model


def predict(model, X: np.ndarray, batch: int = 500_000, n_jobs: int = 1) -> np.ndarray:
    """Batched prediction with a bounded thread count.

    `model.predict` re-reads the booster's own thread setting, so it is pinned
    here too: the default of one thread per core is what made this step appear
    to hang rather than merely be slow.
    """
    try:
        model.booster_.params["num_threads"] = n_jobs
        model.set_params(n_jobs=n_jobs)
    except Exception:  # noqa: BLE001 - never fail prediction over a thread hint
        pass
    out = np.empty(len(X), dtype=np.uint8)
    for a in range(0, len(X), batch):
        out[a : a + batch] = model.predict(X[a : a + batch]).astype(np.uint8)
    return out


def training_features(scene: Scene, cloud_ply: Path, cfg: "ClassifierConfig"):
    """Features and labels for training, sampled where the labels actually are.

    The uniform thinning used for inference wastes almost all of its budget:
    of 1.5 M uniformly sampled points, only ~26 000 lie within the label-transfer
    radius of any ground truth, and those are what the model trains on. Sampling
    the training set from the labelled region instead multiplies the supervision
    at identical cost. Inference is unchanged and stays uniform - this only
    changes which points contribute gradient, not which points get labels.
    """
    from scipy.spatial import cKDTree

    from ..package import submission_cloud
    from ..scoring import read_ply_xyz, voxel_downsample_indices

    cp = Path(cloud_ply)
    full_xyz, rgb = submission_cloud(scene, cp, cache=cp.parent / f"{cp.stem}_subcloud.npz")
    gt_xyz, gt_lab = read_ply_xyz(scene.gt_ply, with_classification=True)

    d, _ = cKDTree(gt_xyz).query(full_xyz, workers=-1)
    near = d <= cfg.label_transfer_radius
    idx_near = np.nonzero(near)[0]
    # thin the labelled region only enough to bound the feature cost
    keep = voxel_downsample_indices(full_xyz[idx_near], 0.10)
    idx_near = idx_near[keep]
    rng = np.random.default_rng(cfg.seed)
    if len(idx_near) > cfg.max_feature_points:
        idx_near = np.sort(rng.choice(idx_near, cfg.max_feature_points, replace=False))
    # background context so neighbourhood features see real surroundings
    idx_bg = np.sort(rng.choice(len(full_xyz), min(len(full_xyz), 500_000), replace=False))
    idx = np.unique(np.concatenate([idx_near, idx_bg]))

    xyz = full_xyz[idx]
    cache = cp.parent / f"{cp.stem}_trainfeat_v2.npz"
    if cache.exists():
        dd = np.load(cache, allow_pickle=True)
        if int(dd["n"]) == len(xyz) and list(dd["names"]) == list(FEATURE_NAMES_EXPECTED):
            y = transfer_labels(gt_xyz, gt_lab, xyz, cfg.label_transfer_radius)
            return dd["X"], y

    up = estimate_up(scene.train_frames)
    rig = np.stack([fr.center for fr in scene.train_frames]).mean(axis=0)
    X, names = compute_features(xyz, rgb[idx] if rgb is not None else None, up, rig=rig)
    np.savez_compressed(cache, X=X, names=np.array(names), n=len(xyz))
    y = transfer_labels(gt_xyz, gt_lab, xyz, cfg.label_transfer_radius)
    return X, y


def leave_one_scene_out(
    scenes: list[Scene], clouds: dict[str, Path], cfg: ClassifierConfig, out_dir: Path
) -> dict:
    """Validate the way the challenge scores: real GT, 10 cm nearest-neighbour
    transfer, per-class IoU pooled, unweighted mIoU.

    With only two labelled scenes this is a two-fold estimate, so the spread
    between folds is reported as the uncertainty rather than a single number.
    """
    from ..scoring import read_ply_xyz, score_semantics, SemanticCounts

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    cache = {}
    for s in scenes:
        t0 = time.time()
        thin, X, _, full = thinned_features(s, clouds[s.key], cfg)
        gv = read_ply_xyz(s.gt_ply, with_classification=True)
        y = transfer_labels(gv[0], gv[1], thin, cfg.label_transfer_radius)
        cache[s.key] = (thin, X, y, full)
        print(f"  [{s.key}] {len(thin)} classified pts (of {len(full)} submitted), "
              f"label hist {np.bincount(y, minlength=256)[:5].tolist()} "
              f"(+{int((y==255).sum())} beyond {cfg.label_transfer_radius} m of any GT) "
              f"({time.time()-t0:.0f}s)", flush=True)

    folds = []
    for held in scenes:
        tr = [s for s in scenes if s.key != held.key]
        Xtr = np.concatenate([cache[s.key][1] for s in tr])
        ytr = np.concatenate([cache[s.key][2] for s in tr])
        model = fit(Xtr, ytr, cfg)

        thin, X, _, xyz = cache[held.key]
        pred = propagate(thin, predict(model, X, n_jobs=cfg.n_jobs), xyz)
        gt_xyz, gt_lab = read_ply_xyz(held.gt_ply, with_classification=True)
        counts = score_semantics(xyz, pred, gt_xyz, gt_lab)
        iou = counts.per_class_iou
        folds.append(
            {
                "held_out": held.key,
                "miou": counts.miou,
                "per_class_iou": [None if np.isnan(v) else round(float(v), 4) for v in iou],
                "miou_ceiling": counts.miou_ceiling,
                "per_class_coverage": [None if np.isnan(v) else round(float(v), 4)
                                       for v in counts.per_class_coverage],
                "unmatched_frac": round(counts.n_unmatched / max(counts.n_total, 1), 4),
                "n_gt_points": counts.n_total,
            }
        )
        print(
            f"  [held out {held.key}] mIoU {counts.miou:.4f} of a ceiling of "
            f"{counts.miou_ceiling:.4f}  IoU {np.round(iou, 3).tolist()}  "
            f"coverage {np.round(counts.per_class_coverage, 3).tolist()}",
            flush=True,
        )

    mious = [f["miou"] for f in folds]
    result = {
        "config": asdict(cfg),
        "folds": folds,
        "miou_mean": float(np.mean(mious)),
        "miou_spread": float(np.max(mious) - np.min(mious)),
        "seconds": round(time.time() - t0, 1),
    }
    (out_dir / "semantics_cv.json").write_text(json.dumps(result, indent=2))
    return result


def fit_final(scenes: list[Scene], clouds: dict[str, Path], cfg: ClassifierConfig):
    """Fit on all labelled scenes, for application to the withheld final scenes."""
    from ..scoring import read_ply_xyz

    Xs, ys = [], []
    for s in scenes:
        thin, X, _, _ = thinned_features(s, clouds[s.key], cfg)
        gt_xyz, gt_lab = read_ply_xyz(s.gt_ply, with_classification=True)
        Xs.append(X)
        ys.append(transfer_labels(gt_xyz, gt_lab, thin, cfg.label_transfer_radius))
    return fit(np.concatenate(Xs), np.concatenate(ys), cfg)
