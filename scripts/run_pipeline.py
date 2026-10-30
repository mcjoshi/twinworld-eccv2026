"""End-to-end pipeline: MVS -> Gaussian Splatting -> semantics -> submission.

Stages are independent and resumable: each writes its artefacts under the run
directory and is skipped if those artefacts already exist, so a crashed or
extended run continues rather than restarting.

    python scripts/run_pipeline.py --stage mvs    --gpu 2 --scenes tum/scene_004 ...
    python scripts/run_pipeline.py --stage render --gpu 3 --scenes ...
    python scripts/run_pipeline.py --stage semantics
    python scripts/run_pipeline.py --stage package --require all
    python scripts/run_pipeline.py --stage score            # dev scenes only
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from twinworld.dataio import ALL_SCENES, DEV_SCENES, FINAL_SCENES, Scene, dev_scenes

RUN_ROOT = Path("experiments")


def parse_scenes(spec: list[str] | None, which: str) -> list[Scene]:
    if spec:
        return [Scene(*k.split("/")) for k in spec]
    table = {"all": ALL_SCENES, "dev": DEV_SCENES, "final": FINAL_SCENES}
    return [Scene(d, s) for d, s in table[which]]


def scene_dir(run: Path, s: Scene) -> Path:
    return run / s.dataset / s.scene_id


def env_capture() -> dict:
    import torch

    def sh(cmd):
        try:
            return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()
        except Exception:  # noqa: BLE001
            return "?"

    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": sh("nvidia-smi --query-gpu=name --format=csv,noheader | head -1"),
        "colmap": sh(f"{os.environ.get('COLMAP_BIN', 'colmap')} -h 2>&1 | head -1"),
        "git_rev": sh("git rev-parse --short HEAD 2>/dev/null || echo uncommitted"),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


# --------------------------------------------------------------------------
def stage_mvs(run: Path, scenes: list[Scene], args) -> None:
    from twinworld.geometry.colmap_mvs import MvsConfig, run_mvs
    from scripts.mvs_ablation import VARIANTS

    for s in scenes:
        # Gold Coast is never scored on geometric precision, only on how much
        # ground truth sits within 10 cm of a submitted point, so it gets the
        # coverage-maximal config while TUM gets the F-score-balanced one.
        variant = args.mvs_variant_gc if s.dataset == "gold_coast" else args.mvs_variant_tum
        cfg = VARIANTS[variant]
        out = scene_dir(run, s) / "mvs"
        if (out / "fused.ply").exists() and not args.force:
            print(f"[mvs] {s.key}: already done, skipping", flush=True)
            continue
        t0 = time.time()
        # One scene failing must not cost the other twelve: this runs unattended
        # and a partial set of clouds still packages, with the sparse cloud as a
        # per-scene fallback.
        try:
            # Tag the workspace by process, not by GPU. `run_mvs` clears its
            # workspace before starting, so two pipeline jobs sharing a tag
            # delete each other's depth maps mid-run. That already happened
            # once, when an older job on the same GPU outlived the kill meant to
            # stop it (the kill hit its shell wrapper, not the Python process).
            summary = run_mvs(s, out, cfg, gpu_index=args.gpu,
                              tag=f"pipe{args.gpu}_{os.getpid()}")
        except Exception as e:  # noqa: BLE001
            print(f"[mvs] {s.key}: FAILED after {time.time()-t0:.0f}s: {e}", flush=True)
            continue
        print(f"[mvs] {s.key}: variant={variant} {summary['n_points']} points "
              f"in {time.time()-t0:.0f}s", flush=True)


def stage_render(run: Path, scenes: list[Scene], args) -> None:
    from twinworld.render.trainer import TrainConfig, train_scene, render_frames

    for s in scenes:
        # Initialisation is a per-dataset decision, and the two datasets want
        # opposite things. Measured on held-out photos:
        #   TUM scene_000     sparse init 0.5432, dense MVS init 0.5946
        #   Gold Coast s009   sparse init 0.5427, dense MVS init 0.4413
        #   Gold Coast s010   sparse init 0.5543
        # The Gold Coast clouds come from the coverage-maximal config: 23-31 M
        # points, deliberately noisy, spread over hundreds of metres. That is the
        # right cloud to *submit* and the wrong cloud to seed Gaussians with,
        # so the two uses are decoupled.
        use_sparse = (s.dataset == "gold_coast") if args.gc_init == "sparse" else False
        iters = args.iters
        cfg = TrainConfig(iters=iters, strategy=args.strategy,
                          init_points=args.init_points,
                          refine_stop_iter=int(iters * 0.72))
        sd = scene_dir(run, s)
        init = s.sparse_ply if use_sparse else sd / "mvs" / "fused.ply"
        if not init.exists():
            init = s.sparse_ply
            print(f"[render] {s.key}: no MVS cloud, falling back to sparse points3D", flush=True)
        out = sd / "gs"
        if (out / "rgb").is_dir() and len(list((out / "rgb").glob("*.png"))) == len(s.test_frames) \
                and not args.force:
            print(f"[render] {s.key}: already done, skipping", flush=True)
            continue
        t0 = time.time()
        try:
            if not (out / "splats.pt").exists() or args.force:
                print(f"[render] {s.key}: init={init.name} ({'sparse' if init == s.sparse_ply else 'mvs'}), "
                      f"{cfg.iters} iters", flush=True)
                train_scene(s, init, out, cfg, log_every=args.log_every)
            render_frames(out / "splats.pt", s.test_frames, out / "rgb", cfg)
        except Exception as e:  # noqa: BLE001
            print(f"[render] {s.key}: FAILED after {time.time()-t0:.0f}s: {e}", flush=True)
            continue
        print(f"[render] {s.key}: done in {time.time()-t0:.0f}s", flush=True)


def stage_semantics(run: Path, scenes: list[Scene], args) -> None:
    from twinworld.semantics.classifier import (
        ClassifierConfig, fit_final, leave_one_scene_out, predict, propagate,
        thinned_features,
    )

    cfg = ClassifierConfig(max_feature_points=args.sem_budget)
    gc_dev = [Scene("gold_coast", s) for s in ("scene_009", "scene_010")]
    clouds = {s.key: scene_dir(run, s) / "mvs" / args.gc_cloud for s in gc_dev}
    missing = [k for k, p in clouds.items() if not p.exists()]
    if missing:
        raise SystemExit(f"semantics needs MVS clouds for {missing}; run --stage mvs first")

    print("[semantics] leave-one-scene-out validation", flush=True)
    cv = leave_one_scene_out(gc_dev, clouds, cfg, run / "semantics")
    print(f"[semantics] mIoU {cv['miou_mean']:.4f} (fold spread {cv['miou_spread']:.4f})", flush=True)

    print("[semantics] fitting on both labelled scenes", flush=True)
    model = fit_final(gc_dev, clouds, cfg)
    import joblib

    (run / "semantics").mkdir(parents=True, exist_ok=True)
    joblib.dump(model, run / "semantics" / "model.joblib")

    targets = [s for s in scenes if s.dataset == "gold_coast"]
    for s in targets:
        ply = scene_dir(run, s) / "mvs" / args.gc_cloud
        if not ply.exists():
            print(f"[semantics] {s.key}: {args.gc_cloud} missing, skipping", flush=True)
            continue
        out = scene_dir(run, s) / "semantics" / "labels.npz"
        if out.exists() and not args.force:
            print(f"[semantics] {s.key}: already done, skipping", flush=True)
            continue
        t0 = time.time()
        thin, X, _, xyz = thinned_features(s, ply, cfg)
        lab = propagate(thin, predict(model, X, n_jobs=cfg.n_jobs), xyz)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, xyz=xyz.astype(np.float32), labels=lab)
        hist = np.bincount(lab, minlength=5)[:5]
        print(
            f"[semantics] {s.key}: {len(lab)} pts, class hist {hist.tolist()} "
            f"({time.time()-t0:.0f}s)", flush=True
        )


def _artefacts(run: Path, scenes: list[Scene], voxel_override: dict | None = None,
               gc_cloud: str = "fused.ply", tum_cloud: str = "fused.ply"):
    from twinworld.package import SceneArtefacts, submission_cloud

    voxel_override = voxel_override or {}
    out = []
    for s in scenes:
        sd = scene_dir(run, s)
        rgb = sd / "gs" / "rgb"
        sem = sd / "semantics" / "labels.npz"
        if s.needs_semantics and sem.exists():
            d = np.load(sem)
            xyz, labels = d["xyz"].astype(np.float64), d["labels"]
            v = voxel_override.get(s.dataset)
            if v:
                from twinworld.scoring import voxel_downsample_indices

                keep = voxel_downsample_indices(xyz, v)
                xyz, labels = xyz[keep], labels[keep]
            # labels were produced on the submission cloud, so this is already
            # cropped and voxelised; re-applying either would break the pairing
        else:
            ply = sd / "mvs" / (gc_cloud if s.needs_semantics else tum_cloud)
            if not ply.exists():
                print(f"  ! {s.key}: no point cloud, using sparse points3D")
                ply = s.sparse_ply
            # `submission_cloud` applies the per-dataset resolution and drops the
            # far-field tail. Going through it here is what keeps the packaged
            # cloud identical to the one semantics and scoring were computed on.
            xyz, _ = submission_cloud(s, ply, voxel=voxel_override.get(s.dataset),
                                      cache=Path(ply).parent / f"{Path(ply).stem}_subcloud.npz")
            # Fallback labels only, when the semantics stage has not run. A
            # constant label scores IoU on that class alone, so the majority
            # class is the least bad constant: wall is 56 percent of Gold Coast
            # points and yields mIoU around 0.11, against 0.004 for "other".
            labels = np.full(len(xyz), 1, np.uint8) if s.needs_semantics else None
            if s.needs_semantics:
                print(f"  ! {s.key}: no semantic labels, falling back to constant 'wall'")
        print(f"  {s.key}: {len(xyz)} points at the dataset's submission resolution")
        out.append(SceneArtefacts(scene=s, rgb_dir=rgb, xyz=xyz, labels=labels))
    return out


def stage_package(run: Path, scenes: list[Scene], args) -> None:
    from twinworld.package import build_submission

    arts = _artefacts(run, scenes, getattr(args, 'voxel_override', None),
                      getattr(args, 'gc_cloud', 'fused.ply'),
                      getattr(args, 'tum_cloud', 'fused.ply'))
    zip_path = Path(args.out or (Path("submissions") / f"{run.name}.zip"))
    # downsample=None: `submission_cloud` has already applied the per-dataset
    # voxel, and re-applying the 2 cm default here would undo it for Gold Coast
    rep = build_submission(arts, zip_path, downsample=None, require=args.require)
    print(rep)
    (run / "submission_report.json").write_text(
        json.dumps({"zip": str(zip_path), "ok": rep.ok, "stats": rep.stats,
                    "errors": rep.errors, "warnings": rep.warnings}, indent=2)
    )
    if rep.ok:
        print(f"\nUPLOAD THIS FILE:  {zip_path.resolve()}")


def stage_score(run: Path, scenes: list[Scene], args) -> None:
    """Score locally on the dev scenes, mirroring the official aggregation."""
    from twinworld.scoring import (
        SemanticCounts, final_score, read_ply_xyz, rendering_score,
        score_geometry, score_rendering,
    )

    scenes = [s for s in scenes if s.is_dev]
    rows, geo, sem = [], [], SemanticCounts()
    ps, ss, lp = [], [], []
    for s in scenes:
        sd = scene_dir(run, s)
        row = {"scene": s.key}
        rgb = sd / "gs" / "rgb"
        if rgb.is_dir() and len(list(rgb.glob("*.png"))) == len(s.test_frames):
            r = score_rendering(rgb, s.test_frames)
            row |= {"psnr": r.psnr, "ssim": r.ssim, "lpips": r.lpips}
            ps.append(r.psnr), ss.append(r.ssim), lp.append(r.lpips)

        sem_f = sd / "semantics" / "labels.npz"
        if s.needs_semantics and sem_f.exists():
            d = np.load(sem_f)
            pred_xyz, pred_lab = d["xyz"].astype(np.float64), d["labels"]
        elif s.needs_semantics:
            print(f"  ! {s.key}: no semantic labels, its mIoU is excluded from the "
                  f"pooled score (run --stage semantics)", flush=True)
            pred_xyz = pred_lab = None
        else:
            ply = sd / "mvs" / "fused.ply"
            pred_xyz = read_ply_xyz(ply)[0] if ply.exists() else None
            pred_lab = None

        if pred_xyz is not None and s.has_gt_cloud:
            if s.dataset == "tum":
                gt, _ = read_ply_xyz(s.gt_ply)
                g = score_geometry(pred_xyz, gt, s.key)
                row |= {"f5": g.f_scores[0.05], "f10": g.f_scores[0.10],
                        "f20": g.f_scores[0.20], "fscore": g.fscore}
                geo.append(g.fscore)
            else:
                from twinworld.scoring import score_semantics

                gt, gl = read_ply_xyz(s.gt_ply, with_classification=True)
                if pred_lab is not None:
                    c = score_semantics(pred_xyz, pred_lab, gt, gl)
                    sem.add(c)
                    row |= {"scene_miou": c.miou,
                            "scene_miou_ceiling": c.miou_ceiling,
                            "unmatched_frac": round(c.n_unmatched / max(c.n_total, 1), 4)}
        rows.append(row)
        print("  " + json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                                 for k, v in row.items()}), flush=True)

    r_s = rendering_score(np.mean(ps), np.mean(ss), np.mean(lp)) if ps else 0.0
    g_s = float(np.mean(geo)) if geo else 0.0
    s_s = sem.miou if sem.n_total else 0.0
    summary = {
        "rows": rows,
        "rendering_score": r_s,
        "geometry_score": g_s,
        "semantic_score": s_s,
        "final_score": final_score(r_s, g_s, s_s),
        "psnr": float(np.mean(ps)) if ps else None,
        "ssim": float(np.mean(ss)) if ss else None,
        "lpips": float(np.mean(lp)) if lp else None,
        "semantic_per_class_iou": [None if np.isnan(v) else round(float(v), 4)
                                   for v in sem.per_class_iou] if sem.n_total else None,
        "semantic_per_class_coverage": [None if np.isnan(v) else round(float(v), 4)
                                        for v in sem.per_class_coverage] if sem.n_total else None,
        "semantic_score_ceiling": sem.miou_ceiling if sem.n_total else None,
        "note": "local scorer, dev scenes only; not the official score",
    }
    (run / "local_scores.json").write_text(json.dumps(summary, indent=2))
    print(
        f"\nLOCAL (dev):  rendering {r_s:.4f} + geometry {g_s:.4f} + semantic {s_s:.4f} "
        f"= {summary['final_score']:.4f} / 3.0"
    )
    if sem.n_total:
        print(
            f"  semantic ceiling from coverage alone: {sem.miou_ceiling:.4f} "
            f"(achieved {s_s:.4f}); per-class coverage "
            f"{np.round(sem.per_class_coverage, 3).tolist()}"
        )


STAGES = {
    "mvs": stage_mvs, "render": stage_render, "semantics": stage_semantics,
    "package": stage_package, "score": stage_score,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=sorted(STAGES))
    ap.add_argument("--run", default="2026-08-08-pipeline-v1")
    ap.add_argument("--scenes", nargs="*", default=None)
    ap.add_argument("--which", default="all", choices=["all", "dev", "final"])
    ap.add_argument("--gpu", type=int, default=2)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--mvs-variant-tum", default="loose")
    ap.add_argument("--mvs-variant-gc", default="coverage")
    # 12 training views: 7000 iterations is already ~580 epochs. 15000 was the
    # 3DGS default carried over from setups with 50x more views.
    # 3000, not 7000: measured on TUM scene_000, 3000 iterations scores
    # 0.6099 against 0.5946 at 7000, and takes 94 s against 232 s. Twelve
    # views make 7000 iterations roughly 580 epochs, which is memorisation.
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--strategy", default="default")
    ap.add_argument("--sem-budget", type=int, default=4_000_000,
                    help="uniform feature budget; 4M is the adopted config")
    ap.add_argument("--tum-cloud", default="fused.ply",
                    help="which TUM cloud to submit (fused.ply or fused_union.ply)")
    ap.add_argument("--gc-cloud", default="fused.ply",
                    help="which Gold Coast cloud to label and submit (fused.ply or fused_gs.ply)")
    ap.add_argument("--gc-init", default="sparse", choices=["sparse", "mvs"],
                    help="what to seed Gold Coast Gaussians from")
    ap.add_argument("--init-points", type=int, default=1_000_000)
    ap.add_argument("--log-every", type=int, default=2000)
    ap.add_argument("--require", default="all", choices=["all", "dev", "final", "none"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--profile", default="full", choices=["full", "lite", "mini", "v5safe", "v5best", "v6mid", "v6safe", "v7safe", "v7best", "v7mini", "v8", "v8mid", "v9", "v9mid", "v10safe", "v11"],
                    help="submission resolution: full, lite, or mini (for diagnosing failures)")
    args = ap.parse_args()

    from twinworld.package import (
        SUBMISSION_VOXEL_LITE, SUBMISSION_VOXEL_MINI,
        SUBMISSION_VOXEL_V5_BEST, SUBMISSION_VOXEL_V5_SAFE,
        SUBMISSION_VOXEL_V6_MID, SUBMISSION_VOXEL_V6_SAFE,
        SUBMISSION_VOXEL_V7_BEST, SUBMISSION_VOXEL_V7_MINI, SUBMISSION_VOXEL_V7_SAFE,
        SUBMISSION_VOXEL_V8, SUBMISSION_VOXEL_V8_MID,
        SUBMISSION_VOXEL_V9, SUBMISSION_VOXEL_V9_MID, SUBMISSION_VOXEL_V10_SAFE, SUBMISSION_VOXEL_V11,
    )

    args.voxel_override = {"full": {}, "lite": SUBMISSION_VOXEL_LITE,
                           "mini": SUBMISSION_VOXEL_MINI,
                           "v5safe": SUBMISSION_VOXEL_V5_SAFE,
                           "v5best": SUBMISSION_VOXEL_V5_BEST,
                           "v6mid": SUBMISSION_VOXEL_V6_MID,
                           "v6safe": SUBMISSION_VOXEL_V6_SAFE,
                           "v7safe": SUBMISSION_VOXEL_V7_SAFE,
                           "v7best": SUBMISSION_VOXEL_V7_BEST,
                           "v7mini": SUBMISSION_VOXEL_V7_MINI,
                           "v8": SUBMISSION_VOXEL_V8,
                           "v8mid": SUBMISSION_VOXEL_V8_MID,
                           "v9": SUBMISSION_VOXEL_V9,
                           "v9mid": SUBMISSION_VOXEL_V9_MID,
                           "v10safe": SUBMISSION_VOXEL_V10_SAFE,
                           "v11": SUBMISSION_VOXEL_V11}[args.profile]
    run = RUN_ROOT / args.run
    run.mkdir(parents=True, exist_ok=True)
    (run / "env.json").write_text(json.dumps(env_capture(), indent=2))
    (run / f"args_{args.stage}.json").write_text(json.dumps(vars(args), indent=2))

    scenes = parse_scenes(args.scenes, args.which)
    print(f"== stage {args.stage} on {len(scenes)} scene(s) -> {run}", flush=True)
    STAGES[args.stage](run, scenes, args)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    main()
