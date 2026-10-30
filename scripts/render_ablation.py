"""Which sparse-view Gaussian Splatting setup renders the held-out views best?

12 training views is roughly a fiftieth of what 3DGS is normally tuned for, so
the defaults are not a safe starting point: 15 000 iterations over 12 images is
1250 epochs, and unconstrained densification has nothing to stop it building
floaters in the space the training views never disambiguate.

The variables that matter here, and what each is testing:

  init         provided sparse points3D vs our dense MVS cloud
  iters        how long before the model starts memorising 12 images
  strategy     gradient-based densification vs MCMC's fixed capacity
  regularise   scale and opacity penalties against floaters

Scored on the development scenes' real held-out photos with the local scorer.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from twinworld.dataio import Scene
from twinworld.render.trainer import TrainConfig, render_frames, train_scene
from twinworld.scoring import rendering_score, score_rendering

BASE = TrainConfig(iters=7000, refine_stop_iter=5000)

VARIANTS: dict[str, tuple[TrainConfig, str]] = {
    "sparse_init":   (replace(BASE), "sparse"),
    "mvs_init":      (replace(BASE), "mvs"),
    "mvs_short":     (replace(BASE, iters=3000, refine_stop_iter=2200), "mvs"),
    "mvs_long":      (replace(BASE, iters=15000, refine_stop_iter=12000), "mvs"),
    "mvs_mcmc":      (replace(BASE, strategy="mcmc", cap_max=2_000_000), "mvs"),
    "mvs_reg":       (replace(BASE, scale_reg=0.05, opacity_reg=0.01), "mvs"),
    "mvs_randombg":  (replace(BASE, random_bg=True), "mvs"),
    # Gold Coast ships sparse init at 7000 iterations. Every longer schedule
    # previously tested was paired with MVS init, which loses badly on this
    # dataset (0.4413 against 0.5427), so "does Gold Coast want more iterations"
    # has never been asked in the configuration we actually ship. 3000 -> 7000
    # was worth +0.058, and 7000 is simply where the search stopped.
    "sparse_long":   (replace(BASE, iters=15000, refine_stop_iter=11000), "sparse"),
    "sparse_xlong":  (replace(BASE, iters=25000, refine_stop_iter=18000), "sparse"),
    "sparse_reg":    (replace(BASE, scale_reg=0.05, opacity_reg=0.01), "sparse"),
    "sparse_mcmc":   (replace(BASE, strategy="mcmc", cap_max=2_000_000), "sparse"),
    # TUM ships MVS init at 3000 iterations. MCMC was only ever tested there at
    # 7000 (mvs_mcmc), and only on Gold Coast in the 26 Aug sweep. TUM is FIVE of
    # the seven scored final-phase scenes against Gold Coast's two, so a gain
    # here is worth about 2.5x more than the same gain on Gold Coast.
    "tum_mcmc":      (replace(BASE, iters=3000, refine_stop_iter=2200,
                              strategy="mcmc", cap_max=2_000_000), "mvs"),
    "tum_mcmc_long": (replace(BASE, iters=7000, refine_stop_iter=5000,
                              strategy="mcmc", cap_max=3_000_000), "mvs"),
    "tum_base":      (replace(BASE, iters=3000, refine_stop_iter=2200), "mvs"),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", required=True)
    ap.add_argument("--scenes", nargs="*", default=["tum/scene_000", "tum/scene_002",
                                                    "gold_coast/scene_009"])
    ap.add_argument("--mvs-exp", default="experiments/2026-08-08-geometry-bakeoff/auto")
    ap.add_argument("--out", default="experiments/2026-08-08-render-ablation")
    ap.add_argument("--gpu", type=int, default=2)
    args = ap.parse_args()

    import os

    # Set by the launcher before Python starts. Setting it here would be after
    # `twinworld.render.trainer` has already imported torch, which is too late
    # to be reliable.
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        raise SystemExit(
            f"run with CUDA_VISIBLE_DEVICES={args.gpu} set in the environment, "
            "so the GPU choice is applied before torch initialises"
        )
    print(f"using CUDA_VISIBLE_DEVICES={visible}", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows_path = out / "scores.json"
    rows = json.loads(rows_path.read_text()) if rows_path.exists() else []

    for name in args.variants:
        cfg, init_kind = VARIANTS[name]
        for key in args.scenes:
            s = Scene(*key.split("/"))
            if not s.has_test_images:
                print(f"skip {key}: no held-out photos to score against", flush=True)
                continue
            if init_kind == "sparse":
                init = s.sparse_ply
            else:
                # accept both layouts: the ablation dirs are
                # <exp>/<dataset>/<scene>/fused.ply and the pipeline run dirs are
                # <run>/<dataset>/<scene>/mvs/fused.ply
                base = Path(args.mvs_exp) / s.dataset / s.scene_id
                init = next(
                    (c for c in (base / "mvs" / "fused.ply", base / "fused.ply") if c.exists()),
                    base / "fused.ply",
                )
            if not init.exists():
                print(f"skip {name}/{key}: init cloud {init} missing", flush=True)
                continue

            run_dir = out / name / s.dataset / s.scene_id
            t0 = time.time()
            summary = train_scene(s, init, run_dir, cfg, log_every=max(cfg.iters // 4, 1))
            render_frames(run_dir / "splats.pt", s.test_frames, run_dir / "rgb", cfg)
            r = score_rendering(run_dir / "rgb", s.test_frames)
            rs = rendering_score(r.psnr, r.ssim, r.lpips)
            row = {
                "variant": name, "scene": s.key, "init": init_kind,
                "psnr": r.psnr, "ssim": r.ssim, "lpips": r.lpips,
                "rendering_score": rs, "n_gaussians": summary["n_gaussians"],
                "iters": cfg.iters, "strategy": cfg.strategy,
                "seconds": round(time.time() - t0, 1),
                "config": asdict(cfg),
            }
            rows.append(row)
            rows_path.write_text(json.dumps(rows, indent=2))
            print(
                f"[{name}] {s.key}  PSNR {r.psnr:.2f}  SSIM {r.ssim:.4f}  "
                f"LPIPS {r.lpips:.4f}  -> render_score {rs:.4f}  "
                f"({summary['n_gaussians']} gauss, {time.time()-t0:.0f}s)",
                flush=True,
            )

    by_variant: dict[str, list[float]] = {}
    for r in rows:
        by_variant.setdefault(r["variant"], []).append(r["rendering_score"])
    print("\nmean rendering_score by variant:")
    for k, v in sorted(by_variant.items(), key=lambda kv: -float(np.mean(kv[1]))):
        print(f"  {k:16s} {np.mean(v):.4f}  (n={len(v)})")


if __name__ == "__main__":
    main()
