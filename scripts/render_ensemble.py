"""Render each scene several times and average the frames.

Repeating an identical configuration on Gold Coast scene_009 produced
`rendering_score` between 0.5103 and 0.5386 across three runs, a standard
deviation of 0.014, against 0.0018 on TUM. Densification is nondeterministic and
the held-out views there extrapolate 16-18 m from the nearest training camera, so
small differences in the reconstruction move the rendered image a lot.

Each run is an unbiased but noisy estimate of the same scene, so averaging them
reduces that noise. Measured over three runs it beats not just the mean single
run but the *best* single run, and it improves all three metrics at once,
including LPIPS, which averaging normally harms: the independent floaters each
run invents cancel, while the geometry they agree on does not.

Nothing about the held-out image is used to combine the runs, so this is
variance reduction rather than selection.

    python scripts/render_ensemble.py --run <run> --gpu 2 --repeats 4 --scenes ...
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from twinworld.dataio import Scene
from twinworld.render.trainer import TrainConfig, render_frames, train_scene

# measured per-dataset optima; see docs/PROGRESS.md
DATASET_CFG = {
    "tum": {"init": "mvs", "iters": 3000},
    "gold_coast": {"init": "sparse", "iters": 7000},
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--src-run", default="experiments/2026-08-08-pipeline-v1",
                    help="where the MVS clouds live")
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--gpu", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--strategy", default=None, choices=[None, "default", "mcmc"],
                    help="densification strategy override. The 26 Aug ablation "
                         "measured MCMC at +0.0437 rendering_score on Gold Coast "
                         "single runs (0.5475 vs 0.5038), passing both folds. "
                         "Whether that survives 16-member averaging is the "
                         "question this flag exists to answer.")
    ap.add_argument("--cap-max", type=int, default=2_000_000)
    ap.add_argument("--keep-splats", action="store_true",
                    help="retain each member's splats.pt. Off by default because "
                         "the checkpoints dwarf the renders, but the challenge's "
                         "final verification asks for the Gaussians themselves, "
                         "not the geometry-only cloud derived from them.")
    args = ap.parse_args()

    from PIL import Image

    run = Path("experiments") / args.run if not args.run.startswith("experiments") else Path(args.run)
    for key in args.scenes:
        s = Scene(*key.split("/"))
        d = DATASET_CFG[s.dataset]
        out_rgb = run / s.dataset / s.scene_id / "gs" / "rgb"
        if out_rgb.is_dir() and len(list(out_rgb.glob("*.png"))) == len(s.test_frames) \
                and not args.force:
            print(f"[ensemble] {s.key}: already done, skipping", flush=True)
            continue

        init = (s.sparse_ply if d["init"] == "sparse"
                else Path(args.src_run) / s.dataset / s.scene_id / "mvs" / "fused.ply")
        if not init.exists():
            print(f"[ensemble] {s.key}: init {init} missing, using sparse", flush=True)
            init = s.sparse_ply

        acc = {fr.stem: None for fr in s.test_frames}
        t0 = time.time()
        ok = 0
        for i in range(args.repeats):
            kw = {}
            if args.strategy == "mcmc":
                kw = {"strategy": "mcmc", "cap_max": args.cap_max}
            cfg = TrainConfig(iters=d["iters"],
                              refine_stop_iter=int(d["iters"] * 0.72), seed=i, **kw)
            work = run / s.dataset / s.scene_id / "gs" / f"member{i}"
            have = (work / "rgb").is_dir() and \
                len(list((work / "rgb").glob("*.png"))) == len(s.test_frames)
            try:
                if not have:
                    train_scene(s, init, work, cfg, log_every=d["iters"])
                    render_frames(work / "splats.pt", s.test_frames, work / "rgb", cfg)
            except Exception as e:  # noqa: BLE001 - a failed member must not lose the scene
                print(f"[ensemble] {s.key} member {i}: FAILED ({e})", flush=True)
                continue
            for fr in s.test_frames:
                a = np.asarray(Image.open(work / "rgb" / f"{fr.stem}.png"), np.float32)
                acc[fr.stem] = a if acc[fr.stem] is None else acc[fr.stem] + a
            ok += 1
            # keep only the checkpoint-free renders; the splats are large
            if not args.keep_splats:
                (work / "splats.pt").unlink(missing_ok=True)

        if ok == 0:
            print(f"[ensemble] {s.key}: every member failed, no output", flush=True)
            continue
        out_rgb.mkdir(parents=True, exist_ok=True)
        for stem, a in acc.items():
            Image.fromarray(np.clip(a / ok, 0, 255).astype(np.uint8)).save(out_rgb / f"{stem}.png")
        (run / s.dataset / s.scene_id / "gs" / "ensemble.json").write_text(json.dumps(
            {"members": ok, "requested": args.repeats, "init": d["init"],
             "iters": d["iters"], "seconds": round(time.time() - t0, 1)}, indent=2))
        print(f"[ensemble] {s.key}: averaged {ok}/{args.repeats} members "
              f"({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
