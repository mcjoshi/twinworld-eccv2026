"""Train a 2D segmenter on projected labels, lift its predictions back to 3D.

The gap this attacks. Our 29 geometric features cannot represent a window: a
window is coplanar with its wall, so plane offset, verticality, curvature and
height are all identical. Measured window IoU is 0.061 and "other" is 0.049,
against roof 0.443. Those two classes are 19 percent of the ground truth and
two of five classes, so they carry 40 percent of the mIoU.

In image space the picture inverts. Projecting the 3D ground truth into the
training views (scripts/make_2d_labels.py) shows windows are 18 percent of
labelled pixels - the signal is abundant and we have never used it.

Pipeline: fine-tune SegFormer on the synthesised 2D masks of one scene, predict
on the other scene's training views, then lift by projecting every submitted
point into every view and taking a majority vote over the predicted labels,
with a depth buffer so occluded points do not collect labels from surfaces in
front of them.

An earlier crude version of this was tried on 9 August and rejected, with the
recorded reason "label noise at 11 px window scale exceeded the gain". That ran
at native resolution with no learned model - it lifted raw 2D predictions from a
generic segmenter. This differs in the two ways that mattered: the segmenter is
trained on this benchmark's own labels, and the lift votes across views instead
of trusting one.

Adoption gate, fixed before running: BOTH folds must beat their own baseline
(gold_coast/scene_009 > 0.1529, gold_coast/scene_010 > 0.2067) and LOSO must beat
the shipped v12 configuration at 0.1915. Per-class IoU is reported so that a gain
concentrated in window and other is visible even if the mean moves little.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from twinworld.dataio import CLASS_NAMES, NUM_CLASSES, Scene
from twinworld.package import submission_cloud
from twinworld.scoring import read_ply_xyz, score_semantics

V1 = Path("experiments/2026-08-08-pipeline-v1")
LAB = Path("experiments/2026-08-24-2d-labels")
OUT = Path("experiments/2026-08-24-seg2d-lift")
PACK_VOXEL = 0.10
IGNORE = 255
import os
AUG = os.environ.get('SEG2D_AUG', '0') == '1'
BASE = {"gold_coast/scene_009": 0.1529, "gold_coast/scene_010": 0.2067}
V12 = 0.1915


def load_pairs(scene: Scene):
    pairs = []
    for fr in scene.train_frames:
        m = LAB / scene.dataset / scene.scene_id / f"{fr.stem}.npy"
        if m.exists() and fr.image_path is not None:
            pairs.append((fr, m))
    return pairs


def train_segmenter(scene, device, steps, crop, lr, model_name):
    from transformers import SegformerForSemanticSegmentation

    model = SegformerForSemanticSegmentation.from_pretrained(
        model_name, num_labels=NUM_CLASSES, ignore_mismatched_sizes=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    pairs = load_pairs(scene)
    print(f"  training on {len(pairs)} labelled views", flush=True)

    imgs, masks = [], []
    for fr, mp in pairs:
        imgs.append(np.asarray(Image.open(fr.image_path).convert("RGB")))
        masks.append(np.load(mp))
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)

    rng = np.random.default_rng(0)
    model.train()
    for step in range(steps):
        bi, bm = [], []
        for _ in range(4):
            k = rng.integers(len(imgs))
            im, mk = imgs[k], masks[k]
            H, W = mk.shape
            y = rng.integers(0, max(H - crop, 1)); x = rng.integers(0, max(W - crop, 1))
            ic = im[y:y + crop, x:x + crop]
            mc = mk[y:y + crop, x:x + crop]
            if rng.random() < 0.5:
                ic, mc = ic[:, ::-1], mc[:, ::-1]
            ic = ic.astype(np.float32) / 255.0
            if AUG:
                # Photometric jitter targets the domain gap, not capacity. The
                # segmenter trains on scenes 009/010 and is applied to 011/012 -
                # different buildings, different lighting - and our LOSO estimate
                # of 0.2123 returned 0.15 on those scenes, a 29 percent shortfall.
                # Capacity has been rejected six times in this project; transfer
                # is the untested axis.
                ic = ic * rng.uniform(0.75, 1.3)                  # brightness
                ic = (ic - ic.mean()) * rng.uniform(0.75, 1.3) + ic.mean()  # contrast
                ic = ic * rng.uniform(0.9, 1.1, size=(1, 1, 3))   # per-channel gain
                if rng.random() < 0.3:                            # mild blur
                    sh = int(rng.integers(1, 3))   # not `k`: that is the image index
                    ic[sh:] = 0.5 * ic[sh:] + 0.5 * ic[:-sh]
                # rng.uniform returns float64 and promotes the array;
                # SegFormer's weights are float32
                ic = np.clip(ic, 0.0, 1.0).astype(np.float32)
            bi.append(((ic - mean) / std).transpose(2, 0, 1))
            bm.append(mc.astype(np.int64))
        x_t = torch.from_numpy(np.ascontiguousarray(np.stack(bi))).to(device)
        y_t = torch.from_numpy(np.ascontiguousarray(np.stack(bm))).to(device)
        logits = model(pixel_values=x_t).logits
        logits = F.interpolate(logits, size=y_t.shape[-2:], mode="bilinear",
                               align_corners=False)
        loss = F.cross_entropy(logits, y_t, ignore_index=IGNORE)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        if step % 200 == 0 or step == steps - 1:
            print(f"    step {step:5d}  loss {loss.item():.4f}", flush=True)
    return model, mean, std


@torch.no_grad()
def predict_views(model, scene, device, mean, std):
    """Per-view predicted label maps for every training view of a scene."""
    model.eval()
    out = {}
    for fr in scene.train_frames:
        if fr.image_path is None:
            continue
        im = np.asarray(Image.open(fr.image_path).convert("RGB"))
        H, W = im.shape[:2]
        x = ((im.astype(np.float32) / 255.0 - mean) / std).transpose(2, 0, 1)
        t = torch.from_numpy(x[None]).to(device)
        logits = model(pixel_values=t).logits
        logits = F.interpolate(logits, size=(H, W), mode="bilinear",
                               align_corners=False)
        out[fr.stem] = logits.argmax(1)[0].to(torch.uint8).cpu().numpy()
    return out


def lift(scene, xyz, pred_maps, z_tol=0.30):
    """Majority vote of 2D predictions over every view that sees each point."""
    votes = np.zeros((len(xyz), NUM_CLASSES), np.uint16)
    for fr in scene.train_frames:
        pm = pred_maps.get(fr.stem)
        if pm is None:
            continue
        H, W = pm.shape
        cam = xyz @ fr.R.T + fr.t
        z = cam[:, 2]
        front = z > 1e-6
        u = np.full(len(xyz), -1.0); v = np.full(len(xyz), -1.0)
        c = fr.camera
        u[front] = c.fx * cam[front, 0] / z[front] + c.cx
        v[front] = c.fy * cam[front, 1] / z[front] + c.cy
        px, py = np.round(u).astype(np.int64), np.round(v).astype(np.int64)
        ok = front & (px >= 0) & (px < W) & (py >= 0) & (py < H)
        idx = np.flatnonzero(ok)
        if not len(idx):
            continue
        flat = py[idx] * W + px[idx]
        # depth buffer over OUR cloud: a point far behind the nearest one in its
        # pixel is occluded and must not collect that pixel's label
        nearest = np.full(W * H, np.inf, np.float32)
        np.minimum.at(nearest, flat, z[idx].astype(np.float32))
        vis = z[idx] <= nearest[flat] + z_tol
        keep = idx[vis]
        lab = pm.reshape(-1)[flat[vis]]
        votes[keep, lab] += 1
    seen = votes.sum(1) > 0
    out = np.full(len(xyz), 1, np.uint8)          # wall is the majority fallback
    out[seen] = np.argmax(votes[seen], axis=1).astype(np.uint8)
    return out, seen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=3)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--lr", type=float, default=6e-5)
    ap.add_argument("--model", default="nvidia/mit-b0")
    ap.add_argument("--tag", default="seg2d_mitb0")
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{a.gpu}")
    scenes = [Scene("gold_coast", s) for s in ("scene_009", "scene_010")]

    fold = {}
    for held in scenes:
        t0 = time.time()
        tr = [s for s in scenes if s.key != held.key][0]
        print(f"\n=== train on {tr.key}, lift onto {held.key} ===", flush=True)
        model, mean, std = train_segmenter(tr, device, a.steps, a.crop, a.lr, a.model)
        preds = predict_views(model, held, device, mean, std)
        print(f"  predicted {len(preds)} views ({time.time()-t0:.0f}s)", flush=True)

        ply = V1 / held.dataset / held.scene_id / "mvs" / "fused_gs_i.ply"
        full, _ = submission_cloud(held, ply, voxel=PACK_VOXEL)
        lab, seen = lift(held, full, preds)
        print(f"  lifted onto {len(full)/1e6:.1f}M points, "
              f"{100*seen.mean():.1f}% seen by >=1 view", flush=True)

        gt, gl = read_ply_xyz(held.gt_ply, with_classification=True)
        c = score_semantics(full, lab, gt, gl)
        iou = c.per_class_iou
        fold[held.key] = {"miou": round(float(c.miou), 4),
                          "seen_frac": round(float(seen.mean()), 4),
                          "per_class": {CLASS_NAMES[i]: round(float(iou[i]), 4)
                                        for i in range(NUM_CLASSES)},
                          "seconds": round(time.time() - t0, 1)}
        print(f"[held {held.key}] mIoU {c.miou:.4f} "
              f"(base {BASE[held.key]:.4f}, {c.miou-BASE[held.key]:+.4f})  "
              + "  ".join(f"{CLASS_NAMES[i]} {iou[i]:.3f}"
                          for i in range(NUM_CLASSES)), flush=True)
        del model
        torch.cuda.empty_cache()

    loso = float(np.mean([f["miou"] for f in fold.values()]))
    both = all(fold[k]["miou"] > BASE[k] for k in BASE)
    row = {"tag": a.tag, "model": a.model, "steps": a.steps,
           "loso_miou": round(loso, 4), "both_folds": both,
           "vs_v12": round(loso - V12, 4),
           "verdict": "ADOPT" if both and loso > V12 else "REJECT",
           "folds": fold}
    print(f"\n[{a.tag}] LOSO {loso:.4f} vs v12 {V12:.4f} ({loso-V12:+.4f}) "
          f"both folds: {both} -> {row['verdict']}", flush=True)
    (OUT / f"{a.tag}.json").write_text(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
