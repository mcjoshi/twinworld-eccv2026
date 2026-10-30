"""Zero-shot ADE20K votes as additional fusion features.

Our segmenter trains on 23 images per scene; the leaders' mIoU (0.374-0.376)
implies conversion no 23-image model reaches. A SegFormer trained on ADE20K's
20k images knows building, windowpane, road, grass, tree, car with no exposure
to our scenes at all - nothing to overfit, so it transfers to the withheld
scenes by construction. Its class posteriors are grouped to 8 super-classes and
appended as extra vote features; the tree learns the mapping to our 5 labels
(e.g. building-seen-from-above = roof, building-from-side = wall,
disambiguated by the geometric features it already has).

Gate: both folds beat baselines AND LOSO beats the slab_push incumbent 0.2451.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from twinworld.dataio import CLASS_NAMES, NUM_CLASSES, Scene
from twinworld.scoring import read_ply_xyz, score_semantics
from twinworld.semantics.classifier import (ClassifierConfig, fit, predict,
                                            thinned_features, transfer_labels)
from seg2d_lift import train_segmenter, predict_views
from seg2d_fusion import vote_features

V1 = Path("experiments/2026-08-08-pipeline-v1")
OUT = Path("experiments/2026-08-28-ade-fusion")
BASE = {"gold_coast/scene_009": 0.1529, "gold_coast/scene_010": 0.2067}
INCUMBENT = 0.2451

# ADE20K id -> 8 super-groups
GROUPS = {"building":0,"house":0,"skyscraper":0,"wall":0,
          "windowpane":1,"door":1,
          "road":2,"sidewalk":2,"path":2,"earth":2,"floor":2,
          "grass":3,"tree":3,"plant":3,"palm":3,"flower":3,
          "car":4,"truck":4,"bus":4,"van":4,"minibike":4,"bicycle":4,"boat":4,
          "sky":5,
          "person":6,"fence":6,"streetlight":6,"pole":6,"signboard":6,"awning":6,
          "water":7,"sea":7,"river":7,"swimming":7}
NG = 8

_ade = {}
def ade_maps(scene, device, max_side=None):
    """Per-view 8-group label maps from zero-shot ADE20K SegFormer-b2."""
    from transformers import SegformerForSemanticSegmentation
    if "m" not in _ade:
        m = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b2-finetuned-ade-512-512").to(device).eval()
        id2l = m.config.id2label
        lut = np.full(150, 7, np.uint8)   # default bucket: water/unknown -> 7
        for i, name in id2l.items():
            key = name.split(";")[0].strip().lower()
            for g, gid in GROUPS.items():
                if key.startswith(g): lut[int(i)] = gid; break
            else: lut[int(i)] = 6          # generic object
        _ade["m"], _ade["lut"] = m, lut
    m, lut = _ade["m"], _ade["lut"]
    mean = np.array([0.485,0.456,0.406],np.float32); std = np.array([0.229,0.224,0.225],np.float32)
    out = {}
    with torch.no_grad():
        for fr in scene.train_frames:
            if fr.image_path is None: continue
            im = np.asarray(Image.open(fr.image_path).convert("RGB"))
            H, W = im.shape[:2]
            if max_side and max(H, W) > max_side:
                scale = max_side / max(H, W)
                infer = np.asarray(Image.fromarray(im).resize(
                    (round(W * scale), round(H * scale)), Image.Resampling.BILINEAR
                ))
            else:
                infer = im
            x = ((infer.astype(np.float32)/255.0 - mean)/std).transpose(2,0,1)
            logits = m(pixel_values=torch.from_numpy(x[None]).to(device)).logits
            logits = F.interpolate(logits, size=(H,W), mode="bilinear", align_corners=False)
            out[fr.stem] = lut[logits.argmax(1)[0].cpu().numpy()]
    return out

def grouped_votes(scene, pts, maps, z_tol=0.30):
    votes = np.zeros((len(pts), NG), np.float32)
    for fr in scene.train_frames:
        pm = maps.get(fr.stem)
        if pm is None: continue
        H, W = pm.shape
        cam = pts @ fr.R.T + fr.t; z = cam[:,2]; front = z > 1e-6
        c = fr.camera
        u = np.full(len(pts),-1.0); v = np.full(len(pts),-1.0)
        u[front] = c.fx*cam[front,0]/z[front]+c.cx
        v[front] = c.fy*cam[front,1]/z[front]+c.cy
        px, py = np.round(u).astype(np.int64), np.round(v).astype(np.int64)
        ok = front & (px>=0)&(px<W)&(py>=0)&(py<H)
        idx = np.flatnonzero(ok)
        if not len(idx): continue
        flat = py[idx]*W+px[idx]
        nearest = np.full(W*H, np.inf, np.float32)
        np.minimum.at(nearest, flat, z[idx].astype(np.float32))
        vis = z[idx] <= nearest[flat]+z_tol
        keep, lab = idx[vis], pm.reshape(-1)[flat[vis]]
        votes[keep, lab] += 1.0
    tot = votes.sum(1, keepdims=True)
    return np.divide(votes, np.maximum(tot,1.0)).astype(np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=2)
    ap.add_argument("--tag", default="ade_push")
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
        amap = ade_maps(s, dev)
        av = grouped_votes(s, thin, amap)
        full = np.load(f"experiments/2026-08-27-slab-clouds/{s.scene_id}_push.npz")["xyz"].astype(np.float64)
        geo[s.key] = (thin, X, y, gt, gl, full, av)
        print(f"[{s.key}] ADE votes ready; groups mean {av.mean(0).round(3)}", flush=True)

    fold = {}
    for held in scenes:
        t0 = time.time()
        tr = [s for s in scenes if s.key != held.key][0]
        m, mu, sd = train_segmenter(tr, dev, 1500, 512, 6e-5, "nvidia/mit-b0")
        vf = {}
        for s in scenes:
            pm = predict_views(m, s, dev, mu, sd)
            vf[s.key] = vote_features(s, geo[s.key][0], pm)
        del m; torch.cuda.empty_cache()
        Xtr = np.hstack([geo[tr.key][1], vf[tr.key], geo[tr.key][6]])
        Xh  = np.hstack([geo[held.key][1], vf[held.key], geo[held.key][6]])
        model = fit(Xtr, geo[tr.key][2], cfg)
        thin, _, _, gt, gl, full, _ = geo[held.key]
        from scipy.spatial import cKDTree
        pred_thin = predict(model, Xh, n_jobs=1)
        _, nn = cKDTree(thin).query(full, k=5, workers=-1)
        lab = pred_thin[nn]
        vt = np.zeros((len(full), NUM_CLASSES), np.float32)
        for c in range(NUM_CLASSES): vt[:,c] = (lab==c).sum(1)
        pred = np.argmax(vt,1).astype(np.uint8)
        cnt = score_semantics(full, pred, gt, gl); iou = cnt.per_class_iou
        fold[held.key] = {"miou": round(float(cnt.miou),4),
                          "per_class": {CLASS_NAMES[i]: round(float(iou[i]),4) for i in range(NUM_CLASSES)}}
        print(f"[held {held.key}] mIoU {cnt.miou:.4f} (base {BASE[held.key]:.4f}) "
              + " ".join(f"{CLASS_NAMES[i]} {iou[i]:.3f}" for i in range(NUM_CLASSES))
              + f" ({time.time()-t0:.0f}s)", flush=True)

    loso = float(np.mean([f["miou"] for f in fold.values()]))
    both = all(fold[k]["miou"] > BASE[k] for k in BASE)
    v = "ADOPT" if both and loso > INCUMBENT else "REJECT"
    print(f"\n[{a.tag}] LOSO {loso:.4f} vs slab_push incumbent {INCUMBENT} ({loso-INCUMBENT:+.4f}) both: {both} -> {v}", flush=True)
    (OUT/f"{a.tag}.json").write_text(json.dumps({"tag":a.tag,"loso_miou":round(loso,4),
        "both_folds":both,"verdict":v,"folds":fold}, indent=2))

if __name__ == "__main__":
    main()
