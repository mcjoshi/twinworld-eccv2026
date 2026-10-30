"""Leave-one-scene-out test of the 2D-segmentation lift.

Adoption rule, fixed before running: the 2D lift replaces the point classifier
only if held-out mIoU beats 0.1717, the LightGBM result on the same v6 clouds.
Both folds must improve for the switch to be trusted, matching how the v6 cloud
decision was made.

Unvoted points (no view sees them, or occlusion-gated everywhere) fall back to
the LightGBM labels already computed for these clouds, so the comparison
isolates what 2D appearance adds rather than what abstention costs.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from twinworld.dataio import CLASS_NAMES, IGNORE_LABEL, NUM_CLASSES, Scene
from twinworld.geometry.gs_depth import interpolated_frames, render_depth
from twinworld.render.trainer import TrainConfig
from twinworld.scoring import read_ply_xyz, score_semantics
from twinworld.semantics.seg2d import (
    finetune, lift_votes, make_model, predict_map, project_labels,
)


def load_image(path):
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"))


def scene_views(s: Scene, work: Path, cfg):
    """(frame, image, rendered depth) for training views and interpolated views.

    Training views use the real photos; interpolated views use the splatting
    model's own renders, since no photo exists there.
    """
    views = []
    depths = []
    sp = work / "splats.pt"
    for fr, depth, alpha, rgb in render_depth(sp, s.train_frames, cfg):
        views.append((fr, load_image(fr.image_path)))
        depths.append(depth.cpu().numpy())
    for fr, depth, alpha, rgb in render_depth(sp, interpolated_frames(s.train_frames, per_pair=2), cfg):
        views.append((fr, (rgb.cpu().numpy() * 255).astype(np.uint8)))
        depths.append(depth.cpu().numpy())
    return views, depths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="experiments/2026-08-10-pipeline-v6")
    ap.add_argument("--work", default="/data2/manish_joshi/twinworld/work/gcfuse")
    ap.add_argument("--out", default="experiments/2026-08-11-seg2d")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = TrainConfig(iters=7000, refine_stop_iter=5040)
    scenes = [Scene("gold_coast", sid) for sid in ("scene_009", "scene_010")]

    # per-scene assets
    assets = {}
    for s in scenes:
        t0 = time.time()
        gt_xyz, gt_lab = read_ply_xyz(s.gt_ply, with_classification=True)
        views, depths = scene_views(s, Path(args.work) / s.scene_id, cfg)
        maps = [project_labels(fr, gt_xyz, gt_lab) for fr, _ in views[:len(s.train_frames)]]
        labelled_frac = np.mean([(m != IGNORE_LABEL).mean() for m in maps])
        d = np.load(Path(args.run) / s.dataset / s.scene_id / "semantics" / "labels.npz")
        assets[s.key] = {
            "gt": (gt_xyz, gt_lab), "views": views, "depths": depths,
            "label_maps": maps, "cloud": d["xyz"].astype(np.float64),
            "lgbm": d["labels"],
        }
        print(f"[{s.key}] {len(views)} views ({len(s.train_frames)} real), "
              f"label maps cover {labelled_frac*100:.1f}% of pixels ({time.time()-t0:.0f}s)",
              flush=True)

    results = {}
    for held in scenes:
        t0 = time.time()
        train = [s for s in scenes if s.key != held.key][0]
        A = assets[train.key]
        model = make_model(args.device)
        imgs = [img for _, img in A["views"][:len(train.train_frames)]]
        model = finetune(model, imgs, A["label_maps"], args.device, epochs=args.epochs)

        H = assets[held.key]
        fam = []
        for (fr, img) in H["views"]:
            fam.append((fr, predict_map(model, img, args.device)))
        labels, votes = lift_votes(H["cloud"], fam, H["depths"])
        n_unvoted = int((labels == IGNORE_LABEL).sum())
        labels = np.where(labels == IGNORE_LABEL, H["lgbm"], labels)

        gt_xyz, gt_lab = H["gt"]
        c = score_semantics(H["cloud"], labels, gt_xyz, gt_lab)
        iou = c.per_class_iou
        results[held.key] = {
            "miou": c.miou, "ceiling": c.miou_ceiling,
            "unvoted_frac": round(n_unvoted / len(labels), 4),
            "per_class_iou": {CLASS_NAMES[i]: round(float(iou[i]), 4)
                              for i in range(NUM_CLASSES)},
        }
        print(f"[held out {held.key}] seg2d mIoU {c.miou:.4f} (ceiling {c.miou_ceiling:.4f}, "
              f"{n_unvoted/len(labels)*100:.1f}% fell back to lgbm)   "
              + "  ".join(f"{CLASS_NAMES[i]} {iou[i]:.3f}" for i in range(NUM_CLASSES))
              + f"   ({time.time()-t0:.0f}s)", flush=True)

    (out / "results.json").write_text(json.dumps(results, indent=2))
    mean = float(np.mean([r["miou"] for r in results.values()]))
    print(f"\nseg2d LOSO mean {mean:.4f}   (LightGBM on the same clouds: 0.1717; "
          f"folds 0.1449 / 0.1985)")
    both_better = all(results[k]["miou"] > b for k, b in
                      [("gold_coast/scene_009", 0.1449), ("gold_coast/scene_010", 0.1985)])
    print("both folds improve:", both_better, " -> adopt" if both_better and mean > 0.1717
          else "  -> keep LightGBM")


if __name__ == "__main__":
    main()
