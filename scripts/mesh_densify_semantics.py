"""Submit a surface, not a point sample.

Gold Coast has no precision term: a submitted point far from ground truth costs
nothing in score. The only thing that matters is whether each ground-truth point
finds *some* submitted point within 10 cm, and then what label that point holds.
Under that metric the optimal submission is not a sampled point cloud - it is a
dense surface.

Our facades are covered at 0.23 not because we failed to see them but because
the stereo cloud is a sparse, hole-ridden sample of them. A mesh interpolates
across those holes. Sampling the mesh densely converts holes into coverage
without inventing any new observation, which is the crucial difference from the
splat-rendered points rejected on 19 August: those added *new and inaccurate*
geometry, while this interpolates *between points we already trust*.

Guard against Poisson's known failure mode - it closes surfaces across genuinely
empty space, hallucinating geometry. Mesh vertices further than `--trim` from any
original point are discarded, so small holes are filled and large voids are not.

Labels are carried from the original cloud by k=5 majority vote, the propagation
adopted on 19 August.

Adoption gate, fixed before running: BOTH folds must beat their own baseline
(009 > 0.1529, 010 > 0.2067) and LOSO must beat the incumbent 0.1834
(lr=0.05). Coverage that does not convert is reported as such - three separate
densification attempts have already failed exactly that way.
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

V1 = Path("experiments/2026-08-08-pipeline-v1")
OUT = Path("experiments/2026-08-20-mesh-densify")
PACK_VOXEL = 0.10
MATCH_R = 0.10
BASE = {"gold_coast/scene_009": 0.1529, "gold_coast/scene_010": 0.2067}
INCUMBENT = 0.1834
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
    # drop the lowest-support vertices: Poisson's fabricated regions
    keep = dens > np.quantile(dens, 0.10)
    mesh.remove_vertices_by_mask(~keep)
    if len(mesh.triangles) == 0:
        return np.empty((0, 3)), np.empty((0, 3), np.uint8)

    s = mesh.sample_points_uniformly(number_of_points=int(sample_n))
    sp = np.asarray(s.points)
    sc = (np.asarray(s.colors) * 255).astype(np.uint8) if s.has_colors() else None

    # trim: keep only samples near real observations, so small holes are filled
    # and large hallucinated voids are not
    d, _ = cKDTree(xyz).query(sp, k=1, distance_upper_bound=trim, workers=-1)
    ok = np.isfinite(d)
    return sp[ok], (sc[ok] if sc is not None else None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=11)
    ap.add_argument("--trim", type=float, default=0.30)
    ap.add_argument("--sample-mult", type=float, default=1.5,
                    help="mesh samples as a multiple of the input point count")
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--tag", default="mesh_d11_t030")
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    cfg = ClassifierConfig(max_feature_points=4_000_000, learning_rate=a.lr)
    scenes = [Scene("gold_coast", s) for s in ("scene_009", "scene_010")]
    ply = {s.key: V1 / s.dataset / s.scene_id / "mvs" / "fused_gs_i.ply"
           for s in scenes}

    cache = {}
    for s in scenes:
        t0 = time.time()
        base, base_rgb = submission_cloud(s, ply[s.key], voxel=PACK_VOXEL)
        gt, gl = read_ply_xyz(s.gt_ply, with_classification=True)
        cov0 = per_class_coverage(base, gt, gl)
        print(f"[{a.tag}][{s.key}] base {len(base)/1e6:.1f}M  "
              + "  ".join(f"{k} {v:.3f}" for k, v in cov0.items()), flush=True)

        mp, mc = mesh_points(base, base_rgb, a.depth, a.trim,
                             len(base) * a.sample_mult)
        print(f"[{a.tag}][{s.key}] mesh samples kept {len(mp)/1e6:.2f}M "
              f"(depth={a.depth}, trim={a.trim} m) ({time.time()-t0:.0f}s)",
              flush=True)

        # priority union: originals never displaced, mesh only in empty cells
        def keys(p):
            return np.floor(p / PACK_VOXEL).astype(np.int64)
        bv = np.ascontiguousarray(keys(base)).view(
            np.dtype((np.void, 8 * 3))).ravel()
        mv = np.ascontiguousarray(keys(mp)).view(
            np.dtype((np.void, 8 * 3))).ravel()
        idx = voxel_downsample_indices(mp, PACK_VOXEL)
        mp, mc = mp[idx], (mc[idx] if mc is not None else None)
        mv = np.ascontiguousarray(keys(mp)).view(
            np.dtype((np.void, 8 * 3))).ravel()
        free = ~np.isin(mv, bv)
        full = np.vstack([base, mp[free]])
        full_rgb = (np.vstack([base_rgb, mc[free]])
                    if base_rgb is not None and mc is not None else None)
        cov1 = per_class_coverage(full, gt, gl)
        print(f"[{a.tag}][{s.key}] union {len(full)/1e6:.1f}M "
              f"(+{free.sum()/1e6:.2f}M)  "
              + "  ".join(f"{k} {cov1[k]:.3f} ({cov1[k]-cov0[k]:+.3f})"
                          for k in cov0), flush=True)
        cache[s.key] = (full, full_rgb, gt, gl, cov0, cov1)

    # features/labels come from the ORIGINAL cloud - the mesh adds submission
    # points, not training signal
    feat = {}
    for s in scenes:
        thin, X, _, _ = thinned_features(s, ply[s.key],
                                         ClassifierConfig(max_feature_points=4_000_000))
        gt, gl = cache[s.key][2], cache[s.key][3]
        y = transfer_labels(gt, gl, thin, cfg.label_transfer_radius)
        feat[s.key] = (thin, X, y)
        print(f"[{a.tag}][{s.key}] {len(thin)} feature pts, "
              f"{int((y < NUM_CLASSES).sum())} labelled", flush=True)

    fold = {}
    for held in scenes:
        t0 = time.time()
        tr = [s for s in scenes if s.key != held.key][0]
        model = fit(feat[tr.key][1], feat[tr.key][2], cfg)
        thin, X, _ = feat[held.key]
        full, _, gt, gl, cov0, cov1 = cache[held.key]
        pred = propagate_k(thin, predict(model, X, n_jobs=cfg.n_jobs), full)
        c = score_semantics(full, pred, gt, gl)
        iou = c.per_class_iou
        fold[held.key] = {"miou": round(float(c.miou), 4),
                          "ceiling": round(float(c.miou_ceiling), 4),
                          "points": int(len(full)),
                          "per_class": {CLASS_NAMES[i]: round(float(iou[i]), 4)
                                        for i in range(NUM_CLASSES)},
                          "seconds": round(time.time() - t0, 1)}
        print(f"[{a.tag}][held {held.key}] mIoU {c.miou:.4f} "
              f"(base {BASE[held.key]:.4f}, {c.miou-BASE[held.key]:+.4f}) "
              f"ceiling {c.miou_ceiling:.4f}  "
              + "  ".join(f"{CLASS_NAMES[i]} {iou[i]:.3f}"
                          for i in range(NUM_CLASSES)), flush=True)

    loso = float(np.mean([f["miou"] for f in fold.values()]))
    both = all(fold[k]["miou"] > BASE[k] for k in BASE)
    row = {"tag": a.tag, "depth": a.depth, "trim": a.trim, "lr": a.lr,
           "loso_miou": round(loso, 4), "both_folds": both,
           "vs_incumbent": round(loso - INCUMBENT, 4),
           "verdict": "ADOPT" if both and loso > INCUMBENT else "REJECT",
           "folds": fold}
    print(f"\n[{a.tag}] LOSO {loso:.4f} vs incumbent {INCUMBENT:.4f} "
          f"({loso-INCUMBENT:+.4f}) both folds: {both} -> {row['verdict']}",
          flush=True)
    (OUT / f"{a.tag}.json").write_text(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
