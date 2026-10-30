"""Combine mesh hole filling with facade-normal offset sheets.

The methods address different geometric errors: Poisson samples fill in-plane
facade holes, while +/- 0.15 m sheets cover out-of-plane wall displacement.
The mesh priority union retains original points in occupied 10 cm cells before
normals and sheets are computed.

Pre-registered adoption gate, fixed before running: BOTH folds must beat their
own baselines (gold_coast/scene_009 > 0.1529 and gold_coast/scene_010 >
0.2067), and LOSO must exceed the mesh incumbent of 0.1931. Mean-only
improvement is a REJECT. Point count and multiplier versus the base cloud are
reported because submissions must remain under the server point limit.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from twinworld.dataio import CLASS_NAMES, NUM_CLASSES, Scene
from twinworld.package import submission_cloud
from twinworld.scoring import read_ply_xyz, score_semantics, voxel_downsample_indices
from twinworld.semantics.classifier import (
    ClassifierConfig, fit, predict, thinned_features, transfer_labels,
)
from twinworld.semantics.features import estimate_up

V1 = Path("experiments/2026-08-08-pipeline-v1")
OUT = Path("experiments/2026-08-20-mesh-offsets")
PACK_VOXEL = 0.10
MATCH_R = 0.10
BASE = {"gold_coast/scene_009": 0.1529, "gold_coast/scene_010": 0.2067}
INCUMBENT = 0.1931
K_PROP = 5


def propagate_k(thin_xyz, thin_labels, full_xyz, k=K_PROP):
    _, idx = cKDTree(thin_xyz).query(full_xyz, k=k, workers=-1)
    lab = thin_labels[idx]
    votes = np.zeros((len(full_xyz), NUM_CLASSES), np.float32)
    for c in range(NUM_CLASSES):
        votes[:, c] = (lab == c).sum(axis=1)
    return np.argmax(votes, axis=1).astype(np.uint8)


def per_class_coverage(sub, gt, gl):
    d, _ = cKDTree(sub).query(gt, k=1, distance_upper_bound=MATCH_R, workers=-1)
    m = np.isfinite(d)
    out = {CLASS_NAMES[c]: round(float(m[gl == c].mean()), 4)
           for c in range(NUM_CLASSES) if (gl == c).any()}
    out["ceiling"] = round(float(np.mean(list(out.values()))), 4)
    return out


def mesh_points(xyz, rgb, depth, trim, sample_n, normal_k=30):
    import open3d as o3d

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(xyz)
    if rgb is not None:
        pc.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64) / 255.0)
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=normal_k))
    pc.orient_normals_consistent_tangent_plane(k=15)

    mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pc, depth=depth, linear_fit=False)
    dens = np.asarray(dens)
    keep = dens > np.quantile(dens, 0.10)
    mesh.remove_vertices_by_mask(~keep)
    if len(mesh.triangles) == 0:
        return np.empty((0, 3)), np.empty((0, 3), np.uint8)

    s = mesh.sample_points_uniformly(number_of_points=int(sample_n))
    sp = np.asarray(s.points)
    sc = (np.asarray(s.colors) * 255).astype(np.uint8) if s.has_colors() else None
    d, _ = cKDTree(xyz).query(sp, k=1, distance_upper_bound=trim, workers=-1)
    ok = np.isfinite(d)
    return sp[ok], (sc[ok] if sc is not None else None)


def normals(xyz, k=24):
    """Local PCA normals. The smallest-eigenvalue direction is the surface normal."""
    tree = cKDTree(xyz)
    n = np.empty_like(xyz)
    chunk = 400_000
    for start in range(0, len(xyz), chunk):
        stop = min(start + chunk, len(xyz))
        _, idx = tree.query(xyz[start:stop], k=k, workers=-1)
        nb = xyz[idx]
        nb = nb - nb.mean(axis=1, keepdims=True)
        cov = np.einsum("mki,mkj->mij", nb, nb) / k
        _, vectors = np.linalg.eigh(cov)
        n[start:stop] = vectors[:, :, 0]
    return n


def priority_union(base, base_rgb, mesh, mesh_rgb):
    """Keep base points and admit voxelized mesh samples only in empty base cells."""
    def keys(points):
        return np.floor(points / PACK_VOXEL).astype(np.int64)

    base_voxels = np.ascontiguousarray(keys(base)).view(
        np.dtype((np.void, 8 * 3))).ravel()
    idx = voxel_downsample_indices(mesh, PACK_VOXEL)
    mesh, mesh_rgb = mesh[idx], (mesh_rgb[idx] if mesh_rgb is not None else None)
    mesh_voxels = np.ascontiguousarray(keys(mesh)).view(
        np.dtype((np.void, 8 * 3))).ravel()
    free = ~np.isin(mesh_voxels, base_voxels)
    full = np.vstack([base, mesh[free]])
    full_rgb = (np.vstack([base_rgb, mesh_rgb[free]])
                if base_rgb is not None and mesh_rgb is not None else None)
    return full, full_rgb, int(free.sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=11)
    ap.add_argument("--trim", type=float, default=0.30)
    ap.add_argument("--sample-mult", type=float, default=1.5)
    ap.add_argument("--offset", type=float, default=0.15)
    ap.add_argument("--vertical-thresh", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--tag", default="mesh_d11_t030_offset015")
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    cfg = ClassifierConfig(max_feature_points=4_000_000, learning_rate=a.lr)
    scenes = [Scene("gold_coast", scene) for scene in ("scene_009", "scene_010")]
    ply = {s.key: V1 / s.dataset / s.scene_id / "mvs" / "fused_gs_i.ply"
           for s in scenes}

    cache = {}
    for s in scenes:
        start = time.time()
        base, base_rgb = submission_cloud(s, ply[s.key], voxel=PACK_VOXEL)
        gt, gl = read_ply_xyz(s.gt_ply, with_classification=True)
        cov0 = per_class_coverage(base, gt, gl)
        print(f"[{a.tag}][{s.key}] base {len(base)/1e6:.1f}M", flush=True)

        mesh, mesh_rgb = mesh_points(
            base, base_rgb, a.depth, a.trim, len(base) * a.sample_mult)
        merged, _, mesh_added = priority_union(base, base_rgb, mesh, mesh_rgb)
        print(f"[{a.tag}][{s.key}] mesh union {len(merged)/1e6:.1f}M "
              f"(+{mesh_added/1e6:.2f}M) ({time.time() - start:.0f}s)", flush=True)

        nrm = normals(merged)
        vertical = np.abs(nrm @ estimate_up(s.train_frames)) < a.vertical_thresh
        facades, facade_normals = merged[vertical], nrm[vertical]
        combined = np.vstack([
            merged,
            facades + facade_normals * a.offset,
            facades - facade_normals * a.offset,
        ])
        full = combined[voxel_downsample_indices(combined, PACK_VOXEL)]
        cov1 = per_class_coverage(full, gt, gl)
        print(f"[{a.tag}][{s.key}] final {len(full)/1e6:.1f}M "
              f"({len(full) / len(base):.2f}x), vertical {vertical.mean() * 100:.1f}%  "
              + "  ".join(f"{name} {cov1[name]:.3f} ({cov1[name] - cov0[name]:+.3f})"
                          for name in cov0), flush=True)
        cache[s.key] = (full, gt, gl, len(base))

    feat = {}
    for s in scenes:
        thin, features, _, _ = thinned_features(
            s, ply[s.key], ClassifierConfig(max_feature_points=4_000_000))
        gt, gl = cache[s.key][1:3]
        labels = transfer_labels(gt, gl, thin, cfg.label_transfer_radius)
        feat[s.key] = (thin, features, labels)

    folds = {}
    for held in scenes:
        start = time.time()
        train = next(s for s in scenes if s.key != held.key)
        model = fit(feat[train.key][1], feat[train.key][2], cfg)
        thin, features, _ = feat[held.key]
        full, gt, gl, base_count = cache[held.key]
        predicted = propagate_k(thin, predict(model, features, n_jobs=cfg.n_jobs), full)
        score = score_semantics(full, predicted, gt, gl)
        folds[held.key] = {
            "miou": round(float(score.miou), 4),
            "ceiling": round(float(score.miou_ceiling), 4),
            "points": int(len(full)),
            "base_points": int(base_count),
            "point_multiplier": round(len(full) / base_count, 3),
            "per_class": {CLASS_NAMES[i]: round(float(score.per_class_iou[i]), 4)
                          for i in range(NUM_CLASSES)},
            "seconds": round(time.time() - start, 1),
        }
        print(f"[{a.tag}][held {held.key}] mIoU {score.miou:.4f} "
              f"(base {BASE[held.key]:.4f}, {score.miou - BASE[held.key]:+.4f})",
              flush=True)

    loso = float(np.mean([fold["miou"] for fold in folds.values()]))
    both = all(folds[key]["miou"] > BASE[key] for key in BASE)
    result = {
        "tag": a.tag,
        "depth": a.depth,
        "trim": a.trim,
        "sample_mult": a.sample_mult,
        "offset": a.offset,
        "vertical_thresh": a.vertical_thresh,
        "lr": a.lr,
        "propagation_k": K_PROP,
        "loso_miou": round(loso, 4),
        "both_folds": both,
        "vs_incumbent": round(loso - INCUMBENT, 4),
        "verdict": "ADOPT" if both and loso > INCUMBENT else "REJECT",
        "folds": folds,
    }
    print(f"[{a.tag}] LOSO {loso:.4f} vs incumbent {INCUMBENT:.4f} "
          f"({loso - INCUMBENT:+.4f}) both folds: {both} -> {result['verdict']}",
          flush=True)
    (OUT / "results.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
