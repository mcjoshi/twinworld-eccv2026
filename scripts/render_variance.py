"""How much of a rendering-ablation difference is real?

Two runs of nominally the same configuration on Gold Coast scene_009 scored PSNR
20.28 and 18.21. Their configs differ only in `refine_stop_iter` (5000 against
5040) and they ended with a similar number of Gaussians, so a 2 dB gap is not
plausibly caused by the setting. Either densification is nondeterministic, or
three test frames per scene is simply too small a sample to resolve differences
of this size, or both.

That matters because every rendering conclusion in this repo comes from
single-run comparisons on the same three frames. This measures the noise floor
so those conclusions can be read with the right error bars:

  repeat   same seed, same config, run twice -> pure run-to-run nondeterminism
  seeds    different seeds -> nondeterminism plus initialisation sampling

A difference smaller than this spread is not evidence of anything.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from twinworld.dataio import Scene
from twinworld.render.trainer import TrainConfig, render_frames, train_scene
from twinworld.scoring import rendering_score, score_rendering


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--run", default="experiments/2026-08-08-pipeline-v1")
    ap.add_argument("--out", default="experiments/2026-08-08-render-variance")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--iters", type=int, default=7000)
    ap.add_argument("--init", default="sparse", choices=["sparse", "mvs"])
    ap.add_argument("--vary-seed", action="store_true")
    args = ap.parse_args()

    s = Scene(*args.scene.split("/"))
    init = (s.sparse_ply if args.init == "sparse"
            else Path(args.run) / s.dataset / s.scene_id / "mvs" / "fused.ply")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for i in range(args.repeats):
        seed = i if args.vary_seed else 0
        cfg = TrainConfig(iters=args.iters, refine_stop_iter=int(args.iters * 0.72), seed=seed)
        d = out / f"{s.scene_id}_{args.init}_{'seed' if args.vary_seed else 'rep'}{i}"
        t0 = time.time()
        summary = train_scene(s, init, d, cfg, log_every=args.iters)
        render_frames(d / "splats.pt", s.test_frames, d / "rgb", cfg)
        r = score_rendering(d / "rgb", s.test_frames)
        sc = rendering_score(r.psnr, r.ssim, r.lpips)
        rows.append({"i": i, "seed": seed, "psnr": r.psnr, "ssim": r.ssim,
                     "lpips": r.lpips, "rendering_score": sc,
                     "n_gaussians": summary["n_gaussians"],
                     "per_frame": r.per_frame})
        print(f"  run {i} (seed {seed}): PSNR {r.psnr:.2f}  SSIM {r.ssim:.4f}  "
              f"LPIPS {r.lpips:.4f}  -> {sc:.4f}  ({summary['n_gaussians']} gauss, "
              f"{time.time()-t0:.0f}s)", flush=True)
        (out / f"{s.scene_id}_{args.init}.json").write_text(json.dumps(rows, indent=2))

    p = np.array([r["psnr"] for r in rows])
    sc = np.array([r["rendering_score"] for r in rows])
    print(f"\n{s.key} / {args.init} init / {args.iters} iters, "
          f"{'varying seed' if args.vary_seed else 'identical config'}:")
    print(f"  PSNR            mean {p.mean():.2f}  sd {p.std(ddof=1):.2f}  "
          f"range {p.min():.2f}-{p.max():.2f}")
    print(f"  rendering_score mean {sc.mean():.4f}  sd {sc.std(ddof=1):.4f}  "
          f"range {sc.min():.4f}-{sc.max():.4f}")
    print(f"  -> differences below about {2*sc.std(ddof=1):.4f} of rendering_score "
          f"are within noise")


if __name__ == "__main__":
    main()
