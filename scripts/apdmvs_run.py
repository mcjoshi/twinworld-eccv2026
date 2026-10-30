"""Run APD-MVS on a scene and score it against our COLMAP PatchMatch baseline.

APD-MVS (CVPR 2023, MIT) replaces fixed square matching windows with adaptive
deformable patches, which is the published fix for exactly our failure profile:
on ETH3D it buys +30 points of completeness for -3 of accuracy over COLMAP, and
our F-score is recall-limited at almost the same precision/recall shape.

Plumbing notes, learned from the sources rather than the README:
- the shipped converter indexes `images[i + 1]`, assuming contiguous COLMAP
  image ids from 1; ours are arbitrary, so the model is rewritten with remapped
  ids first
- images are upscaled 2x with intrinsics scaled to match, consistent with the
  `loose_2x` baseline this is compared against
- the binary is `./APD <dense_folder> [gpu]` where dense_folder contains the
  converter's output; results land in `<dense_folder>/APD/APD.ply`
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from twinworld.dataio import Scene
from twinworld.scoring import read_ply_xyz, score_geometry

REPO = Path(__file__).resolve().parents[1]
APD = REPO / "third_party/APD-MVS"
WORK = Path("/data2/manish_joshi/twinworld/work/apdmvs")


def remap_model(scene: Scene, out: Path, scale: float) -> None:
    """COLMAP text model with contiguous ids 1..N and scaled intrinsics."""
    from PIL import Image

    out.mkdir(parents=True, exist_ok=True)
    (out / "images").mkdir(exist_ok=True)
    model = out / "dslr_calibration_undistorted"
    model.mkdir(exist_ok=True)

    frames = scene.train_frames
    id_map = {fr.image_id: i + 1 for i, fr in enumerate(frames)}

    cam = frames[0].camera
    with open(model / "cameras.txt", "w") as f:
        f.write(f"1 PINHOLE {round(cam.width*scale)} {round(cam.height*scale)} "
                f"{cam.fx*scale} {cam.fy*scale} {cam.cx*scale} {cam.cy*scale}\n")

    src_lines = [l for l in open(scene.train_sparse_dir / "images.txt")
                 if not l.startswith("#")]
    by_id = {}
    for i in range(0, len(src_lines), 2):
        p = src_lines[i].split()
        if p:
            by_id[int(p[0])] = (p, src_lines[i + 1])
    with open(model / "images.txt", "w") as f:
        for fr in frames:
            p, pts_line = by_id[fr.image_id]
            new_id = id_map[fr.image_id]
            f.write(f"{new_id} {' '.join(p[1:8])} 1 {p[9]}\n")
            # remap POINTS2D scaled? converter only uses point3D ids via tracks;
            # 2D coords in this line are unused by it, keep as-is
            f.write(pts_line)

    with open(scene.train_sparse_dir / "points3D.txt") as fi, \
         open(model / "points3D.txt", "w") as fo:
        for line in fi:
            if line.startswith("#") or not line.strip():
                continue
            p = line.split()
            head, track = p[:8], p[8:]
            kept = []
            for i in range(0, len(track) - 1, 2):
                iid = int(track[i])
                if iid in id_map:
                    kept += [str(id_map[iid]), track[i + 1]]
            if kept:
                fo.write(" ".join(head + kept) + "\n")

    for fr in frames:
        dst = out / "images" / fr.name
        if dst.exists():
            continue
        with Image.open(fr.image_path) as im:
            w, h = im.size
            im.resize((round(w * scale), round(h * scale)), Image.LANCZOS).save(dst, quality=98)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--scale", type=float, default=2.0)
    ap.add_argument("--gpu", type=int, default=3)
    ap.add_argument("--out", default="experiments/2026-08-11-apdmvs")
    ap.add_argument("--baseline", default="experiments/2026-08-08-pipeline-v1")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for key in args.scenes:
        s = Scene(*key.split("/"))
        ws = WORK / s.dataset / s.scene_id
        t0 = time.time()
        conv = ws / "conv"
        if not (conv / "APD" / "APD.ply").exists():
            if ws.exists():
                shutil.rmtree(ws)
            remap_model(s, ws, args.scale)
            # save_folder must differ from dense_folder: the converter deletes
            # save_folder/images before writing, which would destroy its own
            # source images if the two coincide
            r = subprocess.run(
                ["python3", str(APD / "colmap2mvsnet.py"), "--dense_folder", str(ws),
                 "--save_folder", str(conv), "--model_ext", ".txt"],
                capture_output=True, text=True)
            if r.returncode != 0:
                print(f"[{s.key}] converter FAILED:\n{r.stderr[-2000:]}", flush=True)
                continue
            print(f"[{s.key}] converted ({time.time()-t0:.0f}s); running APD", flush=True)
            r = subprocess.run([str(APD / "build/APD"), str(conv), str(args.gpu)],
                               capture_output=True, text=True)
            if r.returncode != 0 or not (conv / "APD" / "APD.ply").exists():
                print(f"[{s.key}] APD FAILED (rc={r.returncode}):\n"
                      f"{(r.stdout or '')[-800:]}\n{(r.stderr or '')[-800:]}", flush=True)
                continue

        xyz, _ = read_ply_xyz(conv / "APD" / "APD.ply")
        row = {"scene": s.key, "n_points": int(len(xyz)),
               "seconds": round(time.time() - t0, 1)}
        if s.has_gt_cloud:
            gt, _ = read_ply_xyz(s.gt_ply)
            r_ = score_geometry(xyz, gt, s.key)
            row |= {"fscore": r_.fscore, "f5": r_.f_scores[0.05],
                    "f10": r_.f_scores[0.10], "f20": r_.f_scores[0.20],
                    "p10": r_.precision[0.10], "r10": r_.recall[0.10]}
            print(f"[{s.key}] APD N={len(xyz)}  F={r_.fscore:.4f} "
                  f"(F@5 {r_.f_scores[0.05]:.3f} F@10 {r_.f_scores[0.10]:.3f} "
                  f"F@20 {r_.f_scores[0.20]:.3f}) P@10 {r_.precision[0.10]:.3f} "
                  f"R@10 {r_.recall[0.10]:.3f}  ({time.time()-t0:.0f}s)", flush=True)
        else:
            print(f"[{s.key}] APD N={len(xyz)} (no GT) ({time.time()-t0:.0f}s)", flush=True)
        rows.append(row)
        (out / "scores.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
