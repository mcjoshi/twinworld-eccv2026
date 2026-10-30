"""2D segmentation votes as FEATURES, not as a replacement classifier.

The pure lift (24 Aug) scored LOSO 0.0804 against a 0.1798 baseline. The
segmenter was fine - it converged cleanly - but only 47 percent of submitted
points are visible in any training view, and the other 53 percent received a
constant fallback label. Replacing geometry with image evidence forces a global
choice when the correct choice is per-point.

So instead: append the per-point 2D class votes to the 29 geometric features and
let the gradient-boosted tree decide when to trust them. Points never
photographed get an all-zero vote vector and a `seen` flag of 0, which the model
can condition on, rather than a confidently wrong label.

This targets CONVERSION rather than coverage. We reach about 40 percent of our
coverage ceiling, and the classes we lose are the ones geometry cannot represent:
window IoU 0.061 and other 0.049, against roof 0.443. A window is coplanar with
its wall, so no eigenvalue, verticality or height feature can separate them -
but windows are 18 percent of labelled pixels in image space.

Leave-one-scene-out is preserved end to end: the segmenter for a fold is trained
only on the OTHER scene, so no image model ever sees the scene it is scored on.

Adoption gate, fixed before running: BOTH folds must beat their own baseline
(009 > 0.1529, 010 > 0.2067) and LOSO must beat the shipped v12 at 0.1915.
Per-class IoU reported so a window/other gain is visible even if the mean moves
little.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from twinworld.dataio import CLASS_NAMES, NUM_CLASSES, Scene
from twinworld.package import submission_cloud
from twinworld.scoring import read_ply_xyz, score_semantics
from twinworld.semantics.classifier import (ClassifierConfig, fit, predict,
                                            thinned_features, transfer_labels)
from seg2d_lift import train_segmenter, predict_views, lift  # reuse verbatim

V1 = Path("experiments/2026-08-08-pipeline-v1")
OUT = Path("experiments/2026-08-25-seg2d-fusion")
PACK = 0.10
BASE = {"gold_coast/scene_009": 0.1529, "gold_coast/scene_010": 0.2067}
V12 = 0.1915


def vote_features(scene, pts, pred_maps, z_tol=0.30):
    """Per-point vote fractions over the 5 classes, plus a seen indicator."""
    from scipy.spatial import cKDTree  # noqa: F401  (kept for parity with lift)
    votes = np.zeros((len(pts), NUM_CLASSES), np.float32)
    for fr in scene.train_frames:
        pm = pred_maps.get(fr.stem)
        if pm is None: continue
        H, W = pm.shape
        cam = pts @ fr.R.T + fr.t
        z = cam[:, 2]; front = z > 1e-6
        c = fr.camera
        u = np.full(len(pts), -1.0); v = np.full(len(pts), -1.0)
        u[front] = c.fx*cam[front,0]/z[front] + c.cx
        v[front] = c.fy*cam[front,1]/z[front] + c.cy
        px, py = np.round(u).astype(np.int64), np.round(v).astype(np.int64)
        ok = front & (px>=0)&(px<W)&(py>=0)&(py<H)
        idx = np.flatnonzero(ok)
        if not len(idx): continue
        flat = py[idx]*W + px[idx]
        nearest = np.full(W*H, np.inf, np.float32)
        np.minimum.at(nearest, flat, z[idx].astype(np.float32))
        vis = z[idx] <= nearest[flat] + z_tol
        keep, lab = idx[vis], pm.reshape(-1)[flat[vis]]
        votes[keep, lab] += 1.0
    tot = votes.sum(1, keepdims=True)
    seen = (tot[:,0] > 0).astype(np.float32)
    frac = np.divide(votes, np.maximum(tot, 1.0))
    return np.hstack([frac, seen[:,None]]).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=3)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--final-voxel", type=float, default=None,
                    help="voxel applied to the dense union. The raw dense cloud "
                         "projects to 106.6M final-phase points, 2 percent under "
                         "the 108.6M that FAILED; a coarser final voxel must buy "
                         "the margin, and its LOSO cost has to be measured.")
    ap.add_argument("--cloud", default="plain",
                    choices=["plain", "v12", "dense", "dense_sheets", "densemax",
                             "slab_safe", "slab_push"],
                    help="which packaged cloud the labels are scored on. The "
                         "25 Aug fusion gate ran on 'plain' (the 10 cm cloud), "
                         "while the shipped v13 uses 'v12' (offset sheets). "
                         "Comparing fusion's 0.2103 to v12's 0.1915 mixed the "
                         "two, so the shipped combination is unmeasured.")
    ap.add_argument("--model", default="nvidia/mit-b0",
                    help="SegFormer backbone. mit-b0 is 3.7M parameters and is "
                         "all we have used. More TRAINING STEPS on b0 was "
                         "rejected (0.2123 -> 0.2064), but a larger PRETRAINED "
                         "backbone is a different lever: it supplies ImageNet "
                         "visual structure rather than more free parameters to "
                         "overfit on 23 images.")
    ap.add_argument("--seeds", type=int, default=1,
                    help="train this many segmenters and average their vote "
                         "features. Ensembling independent runs is what produced "
                         "our entire rendering gain; it has never been applied to "
                         "the segmenter.")
    ap.add_argument("--tag", default="fusion_mitb0")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device(f"cuda:{a.gpu}")
    scenes = [Scene("gold_coast", s) for s in ("scene_009","scene_010")]
    ply = {s.key: V1/s.dataset/s.scene_id/"mvs"/"fused_gs_i.ply" for s in scenes}
    cfg = ClassifierConfig(max_feature_points=4_000_000, learning_rate=0.05)

    geo = {}
    for s in scenes:
        thin, X, _, _ = thinned_features(s, ply[s.key], ClassifierConfig(max_feature_points=4_000_000))
        gt, gl = read_ply_xyz(s.gt_ply, with_classification=True)
        y = transfer_labels(gt, gl, thin, cfg.label_transfer_radius)
        if a.cloud in ("slab_safe", "slab_push"):
            # prebuilt slab-crop clouds (build_slab_clouds.py): base cropped at
            # R=120 around the optical-axis convergence point (100 percent of
            # dev GT at 2x margin), interp band within 0.15 m of the base.
            import numpy as _np
            tier = a.cloud.split("_")[1]
            full = _np.load(f"experiments/2026-08-27-slab-clouds/{s.scene_id}_{tier}.npz")["xyz"].astype(_np.float64)
        elif a.cloud == "densemax":
            # maximum-density interpolated views (per_pair=10, max_pairs=400,
            # stride 1) - the accuracy mechanism pushed to its limit
            import numpy as _np
            from twinworld.scoring import voxel_downsample_indices as _vdi
            from twinworld.geometry.densify import robust_core as _rc
            base, _ = submission_cloud(s, ply[s.key], voxel=PACK)
            d_ = _np.load(f"experiments/2026-08-26-interp-max/{s.scene_id}_interp.npz")
            nx = d_["xyz"].astype(_np.float64)
            nx = nx[_rc(nx, s.train_frames)]
            full = _np.vstack([base, nx])
            full = full[_vdi(full, a.final_voxel or PACK)]
        elif a.cloud == "dense_sheets":
            # dense cloud PLUS outward offset sheets - the two adopted coverage
            # mechanisms, never combined under fusion labels. Sheets on the plain
            # cloud raised the ceiling 0.42 -> 0.54; on the dense cloud's 0.49
            # they should reach toward the ~0.60 the leaders' scores back-solve to.
            import numpy as _np
            from twinworld.scoring import voxel_downsample_indices as _vdi
            from twinworld.geometry.densify import robust_core as _rc
            from twinworld.semantics.features import estimate_up as _eu
            from scipy.spatial import cKDTree as _KD
            base, _ = submission_cloud(s, ply[s.key], voxel=PACK)
            d_ = _np.load(f"experiments/2026-08-19-denser-interp/{s.scene_id}_interp.npz")
            nx = d_["xyz"].astype(_np.float64)
            nx = nx[_rc(nx, s.train_frames)]
            dense = _np.vstack([base, nx])
            dense = dense[_vdi(dense, PACK)]
            # PCA normals, outward-oriented, sheets on near-vertical thinned base
            tr_ = _KD(dense)
            nrm = _np.empty_like(dense)
            for a0 in range(0, len(dense), 400_000):
                b0 = min(a0 + 400_000, len(dense))
                _, idx = tr_.query(dense[a0:b0], k=24, workers=-1)
                nb = dense[idx]; nb = nb - nb.mean(axis=1, keepdims=True)
                cov = _np.einsum("mki,mkj->mij", nb, nb) / 24
                _, v_ = _np.linalg.eigh(cov)
                nrm[a0:b0] = v_[:, :, 0]
            rig = _np.stack([f.center for f in s.train_frames]).mean(axis=0)
            flip = _np.einsum("ij,ij->i", nrm, dense - rig) < 0
            nrm[flip] *= -1.0
            up = _eu(s.train_frames)
            vert = _np.abs(nrm @ up) < 0.5
            rng = _np.random.default_rng(0)
            drop = vert & (rng.random(len(dense)) > 0.25)
            kept = dense[~drop]
            sheet = dense[vert] + nrm[vert] * 0.15
            full = _np.vstack([kept, sheet])
            full = full[_vdi(full, a.final_voxel or PACK)]
        elif a.cloud == "dense":
            # denser interpolated views: ceiling 0.421 -> 0.489, measured 19 Aug.
            # Rejected then because the extra points carried bad geometric labels
            # and evicted good ones - a labelling failure, tested before 2D fusion
            # existed. Retested here against the labelling that fixes exactly that.
            import numpy as _np
            from twinworld.scoring import voxel_downsample_indices as _vdi
            from twinworld.geometry.densify import robust_core as _rc
            base, _ = submission_cloud(s, ply[s.key], voxel=PACK)
            d_ = _np.load(f"experiments/2026-08-19-denser-interp/{s.scene_id}_interp.npz")
            nx = d_["xyz"].astype(_np.float64)
            nx = nx[_rc(nx, s.train_frames)]
            full = _np.vstack([base, nx])
            full = full[_vdi(full, a.final_voxel or PACK)]
        elif a.cloud == "v12":
            full = np.load(V1/s.dataset/s.scene_id/"mvs"/"fused_gs_i_v12.npz"
                           )["xyz"].astype(np.float64)
        else:
            full, _ = submission_cloud(s, ply[s.key], voxel=PACK)
        geo[s.key] = (thin, X, y, gt, gl, full)
        print(f"[{s.key}] {len(thin)} feature pts, {len(full)/1e6:.1f}M packaged", flush=True)

    fold = {}
    for held in scenes:
        t0 = time.time()
        tr = [s for s in scenes if s.key != held.key][0]
        print(f"\n=== segmenter trained on {tr.key}; fold holds out {held.key} ===", flush=True)
        models = []
        for sd_i in range(a.seeds):
            torch.manual_seed(sd_i)
            models.append(train_segmenter(tr, dev, a.steps, 512, 6e-5, a.model))
        model, mean, std = models[0]
        vf = {}
        for s in scenes:                      # votes averaged over the segmenters
            acc = None
            for (m_i, mu_i, sd_i) in models:
                v = vote_features(s, geo[s.key][0], predict_views(m_i, s, dev, mu_i, sd_i))
                acc = v if acc is None else acc + v
            vf[s.key] = (acc / len(models)).astype(np.float32)
            print(f"  votes on {s.key}: {100*vf[s.key][:,-1].mean():.1f}% of feature pts seen", flush=True)
        del models, model; torch.cuda.empty_cache()

        Xtr = np.hstack([geo[tr.key][1], vf[tr.key]])
        Xh  = np.hstack([geo[held.key][1], vf[held.key]])
        m = fit(Xtr, geo[tr.key][2], cfg)
        thin, _, _, gt, gl, full = geo[held.key]
        from scipy.spatial import cKDTree
        pred_thin = predict(m, Xh, n_jobs=cfg.n_jobs)
        _, nn = cKDTree(thin).query(full, k=5, workers=-1)
        lab = pred_thin[nn]
        vt = np.zeros((len(full), NUM_CLASSES), np.float32)
        for c in range(NUM_CLASSES): vt[:,c] = (lab==c).sum(1)
        pred = np.argmax(vt,1).astype(np.uint8)
        cnt = score_semantics(full, pred, gt, gl); iou = cnt.per_class_iou
        fold[held.key] = {"miou": round(float(cnt.miou),4),
                          "per_class": {CLASS_NAMES[i]: round(float(iou[i]),4) for i in range(NUM_CLASSES)},
                          "seconds": round(time.time()-t0,1)}
        print(f"[held {held.key}] mIoU {cnt.miou:.4f} (base {BASE[held.key]:.4f}, "
              f"{cnt.miou-BASE[held.key]:+.4f})  "
              + "  ".join(f"{CLASS_NAMES[i]} {iou[i]:.3f}" for i in range(NUM_CLASSES)), flush=True)

    loso = float(np.mean([f["miou"] for f in fold.values()]))
    both = all(fold[k]["miou"] > BASE[k] for k in BASE)
    row = {"tag": a.tag, "loso_miou": round(loso,4), "both_folds": both,
           "vs_v12": round(loso-V12,4),
           "verdict": "ADOPT" if both and loso > V12 else "REJECT", "folds": fold}
    print(f"\n[{a.tag}] LOSO {loso:.4f} vs v12 {V12} ({loso-V12:+.4f}) both folds: {both} -> {row['verdict']}", flush=True)
    (OUT/f"{a.tag}.json").write_text(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
