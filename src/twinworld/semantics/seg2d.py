"""2D semantic segmentation lifted to the 3D cloud.

The point classifier has hit its structural limit: with v6 clouds the coverage
ceiling is 0.43-0.48 but achieved held-out mIoU is 0.17, and even on perfect
ground-truth geometry the point features only reach 0.31 on window and 0.09 on
"other". Windows are an appearance class: a recessed dark rectangle in a facade
is obvious in an image and nearly invisible to local point statistics.

So classify in image space instead, where a pretrained backbone brings texture
priors learned from millions of images, and lift the predictions to 3D:

  1. project the labelled ground-truth clouds of the two dev scenes into their
     own training images (z-buffered, so occlusion is respected)
  2. fine-tune a pretrained 2D segmenter on those sparse label maps
  3. predict full label maps for every training and interpolated view
  4. give each 3D point the majority label of its projections

The pretrained backbone matters for generalisation, which is the point: the
final scenes are unseen buildings, and hand features fitted to two scenes carry
scene-specific colour statistics, while ImageNet-scale texture priors do not.
"""

from __future__ import annotations

import numpy as np
import torch

from ..dataio import IGNORE_LABEL, NUM_CLASSES


# --------------------------------------------------------------------------
# label projection: GT cloud -> per-view sparse label maps
# --------------------------------------------------------------------------
def project_labels(frame, gt_xyz: np.ndarray, gt_labels: np.ndarray,
                   z_eps: float = 0.25, subsample: int = 1):
    """Z-buffered projection of a labelled cloud into one view.

    Returns an HxW uint8 label map with 255 where no point projects. A depth
    buffer keeps the nearest point per pixel, and points more than `z_eps`
    behind the winner are discarded, so back-facade points seen through a
    building do not paint their label onto the front.
    """
    cam = frame.camera
    H, W = cam.height, cam.width
    pc = gt_xyz[::subsample] @ frame.R.T + frame.tvec
    lab = gt_labels[::subsample]
    z = pc[:, 2]
    ok = z > 0.5
    pc, lab, z = pc[ok], lab[ok], z[ok]
    u = np.round(cam.fx * pc[:, 0] / z + cam.cx).astype(np.int64)
    v = np.round(cam.fy * pc[:, 1] / z + cam.cy).astype(np.int64)
    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, lab, z = u[ok], v[ok], lab[ok], z[ok]

    flat = v * W + u
    order = np.argsort(z, kind="stable")
    flat, lab, z = flat[order], lab[order], z[order]
    first = np.unique(flat, return_index=True)[1]

    depth = np.full(H * W, np.inf, np.float32)
    depth[flat[first]] = z[first]
    keep = z <= depth[flat] + z_eps
    flat, lab = flat[keep], lab[keep]
    # write back-to-front so the nearest surviving point wins the pixel
    out = np.full(H * W, IGNORE_LABEL, np.uint8)
    out[flat[::-1]] = lab[::-1]
    return out.reshape(H, W)


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------
def make_model(device: str = "cuda"):
    """DeepLabV3-ResNet50 pretrained on COCO, head re-initialised for 5 classes."""
    import torchvision

    m = torchvision.models.segmentation.deeplabv3_resnet50(
        weights=torchvision.models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT
    )
    m.classifier[4] = torch.nn.Conv2d(256, NUM_CLASSES, 1)
    m.aux_classifier[4] = torch.nn.Conv2d(256, NUM_CLASSES, 1)
    return m.to(device)


NORM_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
NORM_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _to_tensor(img: np.ndarray, device):
    t = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)
    return ((t - NORM_MEAN) / NORM_STD).to(device)


def finetune(model, images: list[np.ndarray], label_maps: list[np.ndarray],
             device: str = "cuda", epochs: int = 40, crop: int = 512,
             lr: float = 1e-4, seed: int = 0, log_every: int = 10):
    """Fine-tune on random crops of the labelled views.

    Class-balanced cross-entropy with 255 ignored. Random crops both augment the
    tiny image set and keep memory bounded; crops are re-drawn every step so the
    handful of views is seen from many croppings.
    """
    rng = np.random.default_rng(seed)
    # inverse-frequency class weights from the label maps themselves
    counts = np.zeros(NUM_CLASSES, np.float64)
    for lm in label_maps:
        for c in range(NUM_CLASSES):
            counts[c] += (lm == c).sum()
    counts[counts == 0] = counts[counts > 0].min()
    w = counts.sum() / (NUM_CLASSES * counts)
    weight = torch.tensor(w / w.sum() * NUM_CLASSES, dtype=torch.float32, device=device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=IGNORE_LABEL)
    model.train()
    n = len(images)
    for ep in range(epochs):
        tot = 0.0
        order = rng.permutation(n)
        for i in order:
            # two crops per step: DeepLabV3's ASPP global-pool branch reduces to
            # 1x1 spatially, and BatchNorm cannot compute statistics from a
            # single value per channel, so batch size 1 crashes in train mode
            xs, ys_ = [], []
            for j in (i, int(rng.integers(0, n))):
                img, lm = images[j], label_maps[j]
                H, W = lm.shape
                y0 = int(rng.integers(0, max(1, H - crop)))
                x0 = int(rng.integers(0, max(1, W - crop)))
                lc = lm[y0:y0 + crop, x0:x0 + crop]
                if (lc != IGNORE_LABEL).sum() < 500:
                    continue
                xs.append(_to_tensor(img[y0:y0 + crop, x0:x0 + crop], device))
                ys_.append(torch.from_numpy(lc.astype(np.int64)))
            if len(xs) < 2:
                continue
            x = torch.stack(xs)
            y = torch.stack(ys_).to(device)
            out = model(x)
            loss = loss_fn(out["out"], y) + 0.4 * loss_fn(out["aux"], y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss)
        if ep % log_every == 0 or ep == epochs - 1:
            print(f"    seg2d epoch {ep:3d}  loss {tot / max(n,1):.4f}", flush=True)
    model.eval()
    return model


@torch.no_grad()
def predict_map(model, img: np.ndarray, device: str = "cuda") -> np.ndarray:
    """Full-image per-pixel class prediction."""
    x = _to_tensor(img, device)[None]
    logits = model(x)["out"][0]
    return logits.argmax(0).to(torch.uint8).cpu().numpy()


# --------------------------------------------------------------------------
# lifting 2D predictions onto a 3D cloud
# --------------------------------------------------------------------------
def lift_votes(xyz: np.ndarray, frames_and_maps, depth_maps=None, z_tol: float = 0.5):
    """Majority vote of per-view 2D predictions over each 3D point.

    If `depth_maps` (rendered from the splatting model, same order) are given,
    a view only votes on a point when the point's depth matches the rendered
    depth at its pixel within `z_tol`, so occluded points do not receive the
    front surface's label.

    Returns (labels uint8 with 255 where unvoted, votes int16).
    """
    votes = np.zeros((len(xyz), NUM_CLASSES), np.int16)
    for k, (frame, lab_map) in enumerate(frames_and_maps):
        cam = frame.camera
        H, W = lab_map.shape
        pc = xyz @ frame.R.T + frame.tvec
        z = pc[:, 2]
        ok = z > 0.5
        u = np.round(cam.fx * pc[:, 0] / z + cam.cx).astype(np.int64)
        v = np.round(cam.fy * pc[:, 1] / z + cam.cy).astype(np.int64)
        ok &= (u >= 0) & (u < W) & (v >= 0) & (v < H)
        if depth_maps is not None:
            dm = depth_maps[k]
            zok = np.zeros(len(xyz), bool)
            zok[ok] = np.abs(dm[v[ok], u[ok]] - z[ok]) <= z_tol
            ok &= zok
        idx = np.nonzero(ok)[0]
        cls = lab_map[v[idx], u[idx]]
        valid = cls < NUM_CLASSES
        np.add.at(votes, (idx[valid], cls[valid].astype(np.int64)), 1)

    total = votes.sum(1)
    labels = np.where(total > 0, votes.argmax(1), IGNORE_LABEL).astype(np.uint8)
    return labels, total
