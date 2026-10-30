"""Score competing render directories against the held-out images.

The 26 Aug ablation measured MCMC at +0.0437 rendering_score on Gold Coast, but
that was a single run against a single run, while the submission ships 16-member
ensembles. Averaging already suppresses variance of the same magnitude - the
ensemble note records Gold Coast varying 0.5103 to 0.5386 across identical runs.
So the question is not whether MCMC beats one default run, but whether it beats
an ensemble at equal member count.

Scores every directory given, on the two Gold Coast scenes that have held-out
test images, and reports rendering_score per scene and averaged.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from twinworld.dataio import Scene
from twinworld.scoring import rendering_score, score_rendering

ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="+", required=True, help="run dirs under experiments/")
ap.add_argument("--scenes", nargs="+", default=["gold_coast/scene_009","gold_coast/scene_010"])
ap.add_argument("--gpu", type=int, default=2)
a = ap.parse_args()

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(a.gpu))
out = {}
for run in a.runs:
    root = Path(run) if run.startswith("experiments") else Path("experiments")/run
    per, ok = {}, True
    for key in a.scenes:
        s = Scene(*key.split("/"))
        d = root/s.dataset/s.scene_id/"gs"/"rgb"
        if not d.is_dir() or len(list(d.glob("*.png"))) < len(s.test_frames):
            print(f"  {run}: {key} incomplete, skipping run"); ok=False; break
        r = score_rendering(d, s.test_frames, device="cuda")
        per[key] = {"psnr": round(r.psnr,3), "ssim": round(r.ssim,4),
                    "lpips": round(r.lpips,4),
                    "rendering_score": round(rendering_score(r.psnr,r.ssim,r.lpips),4)}
    if not ok: continue
    m = float(np.mean([v["rendering_score"] for v in per.values()]))
    out[run] = {"per_scene": per, "mean_rendering_score": round(m,4)}
    print(f"\n=== {run} ===")
    for k,v in per.items(): print(f"  {k:24s} PSNR {v['psnr']:6.2f}  SSIM {v['ssim']:.4f}  LPIPS {v['lpips']:.4f}  score {v['rendering_score']:.4f}")
    print(f"  MEAN rendering_score {m:.4f}")
Path("experiments/2026-08-27-render-compare").mkdir(parents=True, exist_ok=True)
Path("experiments/2026-08-27-render-compare/results.json").write_text(json.dumps(out, indent=2))
