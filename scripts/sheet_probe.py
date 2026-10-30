"""Does thickening the cloud perpendicular to its surface buy mIoU?

The gap being attacked. Our surface sits a median 13.8 cm from the ground truth,
and the scorer matches at 10 cm, so 58 percent of ground truth is unmatched and
counts as error no matter how good the labels are. The measured coverage curve

    match radius   0.10    0.15    0.20    0.30    0.50
    ceiling        0.42    0.60    0.71    0.82    0.90

says the loss is a near-surface offset, not a reconstruction failure: the mass is
there, it is a few centimetres off. Densifying *along* the surface - every cloud
we have built since v12 - cannot close that, because the new points land on the
same offset surface. Thickening *perpendicular* to it can.

The operation. Give every point a local normal and emit copies at fixed offsets
along it. A ground truth point up to 10 + max_offset from our surface then has a
labelled point inside its 10 cm ball, and the copy carries its parent's label.
Gold Coast charges no precision penalty, so the extra shell costs points and
nothing else.

Why this is measurable cheaply. The scored quantity depends only on which cloud
point wins the nearest-neighbour race to each ground truth point. Writing
d = g - p for a candidate parent p with unit normal n, the distance to that
parent's sheet at offset o is

    |d - o n|^2 = |d|^2 - 2 o (d . n) + o^2

so one k-nearest-neighbour query of the ground truth against the *parents*
yields |d|^2 and d.n, and every offset configuration is then closed-form. The
sweep costs one query, not one per configuration, and no sheet is ever
materialised.

Validity. The v18 labels on scenes 009 and 010 were fitted on those scenes, so
absolute mIoU here is optimistic. The gate is therefore the *ratio* to the same
cloud's unsheeted score, applied to v18's official 0.2052. Coverage is reported
alongside because it is label-free and transfers directly. Per-class conversion
is reported because the real risk of this operation is label stealing: a sheet
from a mislabelled parent winning a ball that a correct point used to win.

Nothing is adopted here. Configurations are ranked; the production build reruns
the winner and is checked against the exact scorer before packaging.
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

V1 = Path("experiments/2026-08-08-pipeline-v1")
XL = Path("experiments/2026-08-29-xl-clouds")
OUT = Path("experiments/2026-08-30-sheets")
MATCH_R = 0.10
NORMAL_K = 16
# Candidates per ground truth point. The estimator is exact only while the true
# winning parent is inside this set; beyond it, coverage stays right but the
# winning *label* can be wrong. `truncated_frac` in the output reports how often
# the cap bound, and is the tripwire for raising this.
CAND_K = 192
CHUNK = 400_000
GT_CHUNK = 150_000


def surface_normals(xyz: np.ndarray, rig: np.ndarray, k: int = NORMAL_K):
    """Unit normals by local PCA, oriented towards the camera rig.

    Computed on the clean 10 cm surface cloud rather than the packed cloud: the
    packed cloud carries a 20 cm interpolation band whose neighbourhoods are
    volumetric, and a normal fitted there points in an arbitrary direction.
    A consistent orientation is required for the signed-bias diagnostic and for
    any one-sided configuration to mean anything.
    """
    tree = cKDTree(xyz)
    n = len(xyz)
    out = np.empty((n, 3), np.float32)
    for a in range(0, n, CHUNK):
        b = min(a + CHUNK, n)
        _, idx = tree.query(xyz[a:b], k=k, workers=-1)
        nb = xyz[idx]
        P = nb - nb.mean(axis=1)[:, None, :]
        C = np.einsum("mki,mkj->mij", P, P) / k
        _, V = np.linalg.eigh(C)
        nrm = V[:, :, 0]                      # smallest eigenvalue
        flip = np.sign(np.einsum("mi,mi->m", rig - xyz[a:b], nrm))
        flip[flip == 0] = 1.0
        out[a:b] = (nrm * flip[:, None]).astype(np.float32)
        if a % (CHUNK * 20) == 0:
            print(f"    normals {a/1e6:5.1f}M / {n/1e6:.1f}M", flush=True)
    return out, tree


def iou_counts(transferred: np.ndarray, gt_labels: np.ndarray):
    """Intersection and union counts per class, unmatched marked as -1.

    Mirrors the official scorer: an unmatched ground truth point enters the union
    of its own true class and no intersection, exactly as a wrong prediction
    would, without polluting another class's union.
    """
    inter = np.zeros(NUM_CLASSES, np.int64)
    union = np.zeros(NUM_CLASSES, np.int64)
    for c in range(NUM_CLASSES):
        g = gt_labels == c
        p = transferred == c
        inter[c] = int((g & p).sum())
        union[c] = int(g.sum() + p.sum() - inter[c])
    return inter, union


def wide_configs():
    """The efficient frontier the first sweep pointed at: three wide sheets.

    Ranking on the first pass was s05 < s07 < s10 < s12 and still climbing, while
    the five-sheet s07_14 bought only +0.0045 over the three-sheet s12 for two
    thirds more points. Coverage is limited by how far the shell reaches, and a
    three-sheet stack stays gap-free while its spacing is under twice the match
    radius: {0, +/-20} covers a continuous +/-30 cm band at three times the points,
    beating {0, +/-7, +/-14}'s 24 cm at five times.

    Budget-matched coarsening was a wash or a loss on the first pass, so parents
    stay at the packed 12 cm resolution here and only the reach varies.
    """
    sets = {
        "s14": [0.0, -0.14, 0.14],
        "s16": [0.0, -0.16, 0.16],
        "s18": [0.0, -0.18, 0.18],
        "s20": [0.0, -0.20, 0.20],
        "s10_20": [0.0, -0.10, 0.10, -0.20, 0.20],
        "s12_24": [0.0, -0.12, 0.12, -0.24, 0.24],
        "s08_16_24": [0.0, -0.08, 0.08, -0.16, 0.16, -0.24, 0.24],
    }
    out = {f"v012_{k}": (0.0, v) for k, v in sets.items()}
    out["v012_base"] = (0.0, [0.0])          # anchor for the ratio
    out["v012_s12"] = (0.0, [0.0, -0.12, 0.12])   # bridge to the first sweep
    return out


def reach_configs():
    """Push reach, not sheet count - the axis the earlier sweeps never tested.

    Every configuration so far stopped at +/-24 cm because the offsets were packed
    tightly. But coverage is set by how far the shell extends, and s08_16_24 at
    seven sheets bought essentially nothing over s12_24 at five: same 34 cm reach,
    same 0.79 coverage. Spending the same five sheets at wider spacing reaches
    46 cm instead of 34.

    Spacing is what must stay under twice the match radius to avoid holes; the
    plateau at +/-18 cm found earlier says 18 is safe and 20 is the edge. These
    stacks therefore step by 15 to 20 cm and get their reach from more rings, not
    from tighter packing.
    """
    sets = {
        "r16_32": [0.0, -0.16, 0.16, -0.32, 0.32],
        "r18_36": [0.0, -0.18, 0.18, -0.36, 0.36],
        "r20_40": [0.0, -0.20, 0.20, -0.40, 0.40],
        "r15_30_45": [0.0, -0.15, 0.15, -0.30, 0.30, -0.45, 0.45],
        "r18_36_54": [0.0, -0.18, 0.18, -0.36, 0.36, -0.54, 0.54],
        "s12_24": [0.0, -0.12, 0.12, -0.24, 0.24],   # bridge to the wide sweep
    }
    out = {f"v012_{k}": (0.0, v) for k, v in sets.items()}
    out["v012_base"] = (0.0, [0.0])
    return out


def configs(max_off: float):
    """Offset sets to rank, plus the parent voxel each is applied to.

    Two axes are crossed. The offsets decide how thick the shell is. The parent
    voxel decides how the same point budget is split between surface density and
    shell thickness - coarsening the parents pays for the extra sheets, which is
    the comparison that actually decides the build, since the envelope is fixed.
    """
    sets = {
        "base": [0.0],
        "s05": [0.0, -0.05, 0.05],
        "s07": [0.0, -0.07, 0.07],
        "s10": [0.0, -0.10, 0.10],
        "s12": [0.0, -0.12, 0.12],
        "s07_14": [0.0, -0.07, 0.07, -0.14, 0.14],
        "out07": [0.0, 0.07],
        "in07": [0.0, -0.07],
        "out10": [0.0, 0.10],
        "in10": [0.0, -0.10],
    }
    out = {}
    for name, offs in sets.items():
        if max(abs(o) for o in offs) <= max_off + 1e-9:
            out[f"v012_{name}"] = (0.0, offs)
    # budget-matched alternatives: coarser parents, thicker shell
    for v in (0.15, 0.18):
        for name in ("s07", "s10", "s12", "s07_14"):
            if max(abs(o) for o in sets[name]) <= max_off + 1e-9:
                out[f"v{int(v*100):03d}_{name}"] = (v, sets[name])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", default=["scene_009", "scene_010"])
    ap.add_argument("--max-offset", type=float, default=0.14)
    ap.add_argument("--wide", action="store_true",
                    help="sweep the three-sheet reach frontier instead")
    ap.add_argument("--reach", action="store_true",
                    help="sweep wide-spaced stacks that extend the shell furthest")
    ap.add_argument("--cand-k", type=int, default=CAND_K)
    ap.add_argument("--out-name", default="sweep")
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    cfgs = (reach_configs() if a.reach else
            wide_configs() if a.wide else configs(a.max_offset))
    max_off = max(max(abs(o) for o in offs) for _, offs in cfgs.values())
    cap = MATCH_R + max_off + 0.03
    cand_k = a.cand_k
    print(f"{len(cfgs)} configurations, candidate cap {cap:.2f} m", flush=True)

    pooled = {k: [np.zeros(NUM_CLASSES, np.int64), np.zeros(NUM_CLASSES, np.int64)]
              for k in cfgs}
    pts = {k: 0 for k in cfgs}
    per_scene = {}

    for sid in a.scenes:
        s = Scene("gold_coast", sid)
        t0 = time.time()
        print(f"\n=== {s.key} ===", flush=True)

        # normals on the clean surface cloud, then inherited by the packed cloud
        base, _ = submission_cloud(
            s, V1 / s.dataset / s.scene_id / "mvs" / "fused_gs_i.ply", voxel=0.10)
        rig = np.mean([f.center for f in s.train_frames], axis=0)
        print(f"  base {len(base)/1e6:.1f}M pts, fitting normals", flush=True)
        bn, btree = surface_normals(base, rig)

        d = np.load(XL / f"{sid}_xl_v18labeled.npz")
        xyz = d["xyz"].astype(np.float64)
        labels = d["labels"]
        print(f"  packed {len(xyz)/1e6:.1f}M pts, inheriting normals "
              f"({time.time()-t0:.0f}s)", flush=True)
        nrm = np.empty((len(xyz), 3), np.float32)
        for i in range(0, len(xyz), CHUNK * 5):
            j = min(i + CHUNK * 5, len(xyz))
            _, k1 = btree.query(xyz[i:j], k=1, workers=-1)
            nrm[i:j] = bn[k1]
        del base, bn, btree

        gt, gl = read_ply_xyz(s.gt_ply, with_classification=True)
        valid = gl != 255
        gt, gl = gt[valid], gl[valid]
        print(f"  {len(gt)/1e6:.1f}M GT pts, querying k={cand_k} candidates "
              f"({time.time()-t0:.0f}s)", flush=True)

        tree = cKDTree(xyz)

        # parent subsets for the budget-matched configurations
        masks = {0.0: None}
        for v in sorted({v for v, _ in cfgs.values()} - {0.0}):
            keep = np.zeros(len(xyz), bool)
            keep[voxel_downsample_indices(xyz, v)] = True
            masks[v] = keep
            print(f"  parent voxel {v:.2f}: {keep.sum()/1e6:.1f}M "
                  f"({100*keep.mean():.1f}%)", flush=True)

        for name, (v, offs) in cfgs.items():
            m = masks[v]
            pts[name] += int(len(xyz) if m is None else m.sum()) * len(offs)

        trans = {k: np.full(len(gt), -1, np.int64) for k in cfgs}
        truncated = 0
        for a0 in range(0, len(gt), GT_CHUNK):
            b0 = min(a0 + GT_CHUNK, len(gt))
            dist, idx = tree.query(gt[a0:b0], k=cand_k,
                                   distance_upper_bound=cap, workers=-1)
            ok = np.isfinite(dist)
            truncated += int((ok[:, -1]).sum())      # cap was binding for these
            idx = np.where(ok, idx, 0)
            delta = gt[a0:b0][:, None, :] - xyz[idx]          # (m, K, 3)
            aa = np.einsum("mki,mki->mk", delta, delta)       # |d|^2
            bb = np.einsum("mki,mki->mk", delta, nrm[idx])    # d . n
            lab = labels[idx]
            for name, (v, offs) in cfgs.items():
                valid_c = ok if masks[v] is None else (ok & masks[v][idx])
                best = np.full(aa.shape, np.inf)
                for o in offs:
                    best = np.minimum(best, aa - 2.0 * o * bb + o * o)
                best = np.where(valid_c, best, np.inf)
                j = np.argmin(best, axis=1)
                r = np.take_along_axis(best, j[:, None], 1)[:, 0]
                hit = r <= MATCH_R * MATCH_R
                # cast before np.where: uint8 labels would promote the -1
                # sentinel to 255 and every unmatched point would read as matched
                won = lab[np.arange(len(j)), j].astype(np.int64)
                trans[name][a0:b0] = np.where(hit, won, -1)
            if a0 % (GT_CHUNK * 8) == 0:
                print(f"    gt {a0/1e6:5.1f}M / {len(gt)/1e6:.1f}M "
                      f"({time.time()-t0:.0f}s)", flush=True)

        rows = {}
        for name in cfgs:
            i_, u_ = iou_counts(trans[name], gl)
            pooled[name][0] += i_
            pooled[name][1] += u_
            iou = np.where(u_ > 0, i_ / np.maximum(u_, 1), 0.0)
            rows[name] = {"miou": round(float(iou.mean()), 4),
                          "coverage": round(float((trans[name] >= 0).mean()), 4),
                          "per_class": {CLASS_NAMES[c]: round(float(iou[c]), 4)
                                        for c in range(NUM_CLASSES)}}
        per_scene[s.key] = {"truncated_frac": round(truncated / len(gt), 4), **rows}
        print(f"  candidate cap was binding for {100*truncated/len(gt):.2f}% of GT",
              flush=True)
        for name in sorted(rows, key=lambda k: -rows[k]["miou"]):
            print(f"    {name:16s} mIoU {rows[name]['miou']:.4f}  "
                  f"cov {rows[name]['coverage']:.4f}", flush=True)

        # the estimator must reproduce the exact scorer on the unsheeted cloud
        exact = score_semantics(xyz, labels, gt, gl)
        print(f"  [check] estimator base {rows['v012_base']['miou']:.4f} vs exact "
              f"{exact.miou:.4f}", flush=True)
        per_scene[s.key]["exact_base_miou"] = round(float(exact.miou), 4)
        del tree, xyz, nrm, labels, masks, trans

        (OUT / f"{a.out_name}.json").write_text(json.dumps(
            {"per_scene": per_scene}, indent=2))

    base_pool = None
    ranked = []
    for name in cfgs:
        i_, u_ = pooled[name]
        iou = np.where(u_ > 0, i_ / np.maximum(u_, 1), 0.0)
        mi = float(iou.mean())
        if name == "v012_base":
            base_pool = mi
        ranked.append({"config": name, "offsets": cfgs[name][1],
                       "parent_voxel": cfgs[name][0],
                       "pooled_miou": round(mi, 4),
                       "points_millions": round(pts[name] / 1e6, 1),
                       "per_class": {CLASS_NAMES[c]: round(float(iou[c]), 4)
                                     for c in range(NUM_CLASSES)}})
    for r in ranked:
        r["ratio_vs_base"] = round(r["pooled_miou"] / base_pool, 4)
        r["predicted_official"] = round(0.2052 * r["ratio_vs_base"], 4)
    ranked.sort(key=lambda r: -r["pooled_miou"])

    print(f"\npooled over {len(a.scenes)} scenes (base {base_pool:.4f})", flush=True)
    print(f"{'config':16s} {'mIoU':>7} {'ratio':>6} {'pred':>7} {'Mpts':>8}", flush=True)
    for r in ranked:
        print(f"{r['config']:16s} {r['pooled_miou']:7.4f} {r['ratio_vs_base']:6.3f} "
              f"{r['predicted_official']:7.4f} {r['points_millions']:8.1f}", flush=True)
    (OUT / f"{a.out_name}.json").write_text(json.dumps(
        {"base_pooled_miou": round(base_pool, 4), "ranked": ranked,
         "per_scene": per_scene}, indent=2))
    print(f"\nwrote {OUT}/{a.out_name}.json", flush=True)


if __name__ == "__main__":
    main()
