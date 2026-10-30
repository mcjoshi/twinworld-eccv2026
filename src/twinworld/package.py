"""Submission assembly and validation.

A rejected submission burns a day's quota and cannot be undone, so nothing is
written to a zip that has not first been checked against the format rules:
exact frame stems, exact reference resolution, PNG, valid finite PLY with the
required properties and no extras.
"""

from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .dataio import ALL_SCENES, DEV_SCENES, FINAL_SCENES, IGNORE_LABEL, NUM_CLASSES, Scene
from .scoring import VOXEL_SIZE, voxel_downsample, voxel_downsample_indices

VALID_LABELS = set(range(NUM_CLASSES)) | {IGNORE_LABEL}

# Submission resolution per dataset, chosen from what each metric can see.
# TUM is scored on an F-score computed after a 2 cm voxel downsample, so 2 cm is
# exactly right there. Gold Coast is scored only on whether ground truth has a
# submitted point within 10 cm, and geometric precision is never scored, so its
# resolution is set by the match radius rather than by the voxel. Measured on
# scene_009: going from 2 cm to 6 cm costs 0.004 of the semantic ceiling
# (0.1751 -> 0.1710) and removes 38 percent of the points.
SUBMISSION_VOXEL = {"tum": 0.02, "gold_coast": 0.06}

# A second, coarser profile. Scoring runs on the organiser's single machine
# (queue "twinworld-local-3070") under a 600 s execution limit, and the dominant
# cost is a nearest-neighbour query of 8.8 M ground-truth points against *our*
# cloud, so our point count directly drives their runtime. This profile halves
# the Gold Coast clouds for a measured 0.02 of semantic ceiling.
SUBMISSION_VOXEL_LITE = {"tum": 0.04, "gold_coast": 0.12}

# Deliberately small, for diagnosing a scorer that fails with an empty log.
# Measured cost on the development scenes: TUM F-score 0.5336 -> 0.5157,
# Gold Coast semantic ceiling 0.1709 -> 0.0986. Worth it only to establish
# that the submission pipeline works end to end at all.
SUBMISSION_VOXEL_MINI = {"tum": 0.06, "gold_coast": 0.20}

# For the fused (stereo + splatting-depth) Gold Coast clouds, which are denser.
# "safe" targets the size class that has already been accepted by the scorer;
# "best" targets the best score we can build. Measured ceilings on scene_009:
# 10 cm 0.2914, 20 cm 0.1882, against 0.0986 for the stereo-only cloud at 20 cm,
# so even the safe profile keeps most of the fusion gain.
SUBMISSION_VOXEL_V5_SAFE = {"tum": 0.06, "gold_coast": 0.20}
SUBMISSION_VOXEL_V5_BEST = {"tum": 0.04, "gold_coast": 0.10}

# For the interpolated-view fused clouds (v6), which are denser at every voxel.
# The achieved-mIoU curve has a cliff between 14 and 20 cm (0.308 -> 0.209
# train-fitted), so "mid" holds 14 cm and pays the size; "safe" accepts the
# cliff to stay in the size class the scorer has already accepted.
SUBMISSION_VOXEL_V6_MID = {"tum": 0.06, "gold_coast": 0.14}
SUBMISSION_VOXEL_V6_SAFE = {"tum": 0.06, "gold_coast": 0.20}

# v7: TUM clouds are now APD-union (10-12M pts/scene at 2 cm), so TUM needs its
# own thinning to stay inside the scorer's envelope (864 MB accepted, 1.51 GB
# failed, mechanism believed to be the 600 s timeout). Measured cost of TUM
# 4 cm on the loose clouds was -0.007 F; assumed similar on union.
SUBMISSION_VOXEL_V7_SAFE = {"tum": 0.04, "gold_coast": 0.20}
SUBMISSION_VOXEL_V7_BEST = {"tum": 0.02, "gold_coast": 0.14}
# at the proven-accepted size envelope (~0.88 GB); TUM 6 cm costs ~-0.018 F vs 2 cm
SUBMISSION_VOXEL_V7_MINI = {"tum": 0.06, "gold_coast": 0.20}

# v8: the packaging discovery of 13 Aug. Gold Coast at 20 cm loses half its
# reachable ground truth outright, because one point per 20 cm cell leaves a
# ground-truth point at the edge of the scorer's 10 cm match radius: measured
# ceiling 0.2577 -> 0.4207 (scene_009) and 0.2786 -> 0.4686 (scene_010) going to
# 10 cm. TUM returns to the 2 cm the F-score is computed at (mean F 0.5937 over
# the four ground-truth scenes, against 0.549 at 6 cm).
SUBMISSION_VOXEL_V8 = {"tum": 0.02, "gold_coast": 0.10}
SUBMISSION_VOXEL_V8_MID = {"tum": 0.03, "gold_coast": 0.12}

# v9: pushes Gold Coast to 6 cm. The coverage ceiling rises 0.4207 -> 0.4333
# (scene_009) and 0.4686 -> 0.4847 (scene_010), and the leave-one-scene-out
# semantic figure rises 0.1798 -> 0.1846 - real but small, because past 10 cm the
# spacing is already well inside the scorer's 10 cm match radius and the residual
# error is classification rather than coverage. The dev-visible gain is far
# larger (~+0.05) because dev semantics is train-fit. Costs ~47 M points on the
# two dev scenes, which pushes the scored total to ~126 M against the 108 M
# proven to complete in 600 s, so this profile MUST be dev-tested before use.
SUBMISSION_VOXEL_V9 = {"tum": 0.02, "gold_coast": 0.06}
SUBMISSION_VOXEL_V9_MID = {"tum": 0.02, "gold_coast": 0.08}

# v10safe: sized after v8-final FAILED on the final phase with 108.6 M scored
# points, while 108.4 M had passed on the dev phase. The final scorer is more
# expensive than dev - 7 scenes instead of 6, and Gold Coast 011's ground truth
# is likely larger than 009/010's - so the envelope must be re-established from
# below. This sits ~28 percent under the known failure.
#
# Allocation follows the measured exchange rates: Gold Coast 20 -> 14 cm buys
# +0.054 held-out mIoU for ~12 M points, while TUM 4 -> 3 cm buys only +0.010 F
# for ~8 M, so the budget goes to Gold Coast and TUM stays at 4 cm.
SUBMISSION_VOXEL_V10_SAFE = {"tum": 0.04, "gold_coast": 0.14}

# v11: the allocation the failure taught us. v8 spent 49.5 M points on TUM at
# 2 cm and failed at 108.6 M total. TUM 2 -> 4 cm costs only 0.023 of F-score
# (0.5937 -> 0.5709 measured) and frees 19.5 M points, which buys Gold Coast back
# from 14 cm to 10 cm - worth 0.0134 of held-out mIoU. TUM at 4 cm is also the
# packaging that actually scored F 0.627 on the final scenes on 20 Aug, so its
# geometry is measured rather than assumed.
SUBMISSION_VOXEL_V11 = {"tum": 0.04, "gold_coast": 0.10}






class SubmissionError(Exception):
    pass


@dataclass
class SceneArtefacts:
    """What one scene contributes to a submission."""

    scene: Scene
    rgb_dir: Path
    xyz: np.ndarray
    labels: np.ndarray | None = None


@dataclass
class ValidationReport:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def __str__(self) -> str:
        lines = [f"{'VALID' if self.ok else 'INVALID'}: {self.stats}"]
        lines += [f"  ERROR   {e}" for e in self.errors]
        lines += [f"  warning {w}" for w in self.warnings]
        return "\n".join(lines)


# Coordinates are snapped to a per-dataset grid before writing. Snapped float32
# mantissas compress far better under deflate (measured: zip ratio 0.92 -> 0.70
# at 1 cm), and the displacement is bounded by half the grid diagonal: 0.43 cm
# for TUM's 5 mm grid against a 5 cm finest F-score threshold, 0.87 cm for Gold
# Coast's 1 cm grid against a 10 cm match radius.
QUANTISE_GRID = {"tum": 0.005, "gold_coast": 0.01}


def _ply_bytes(xyz: np.ndarray, labels: np.ndarray | None, grid: float | None = None) -> bytes:
    from plyfile import PlyData, PlyElement

    if grid:
        xyz = np.round(np.asarray(xyz) / grid) * grid
    fields = [("x", "f4"), ("y", "f4"), ("z", "f4")]
    if labels is not None:
        fields.append(("classification", "u1"))
    vertex = np.empty(len(xyz), dtype=fields)
    a = np.asarray(xyz, dtype=np.float32)
    vertex["x"], vertex["y"], vertex["z"] = a[:, 0], a[:, 1], a[:, 2]
    if labels is not None:
        vertex["classification"] = np.asarray(labels).astype(np.uint8)
    buf = io.BytesIO()
    PlyData([PlyElement.describe(vertex, "vertex")], text=False, byte_order="<").write(buf)
    return buf.getvalue()


def validate_scene(art: SceneArtefacts, downsample: float | None = VOXEL_SIZE) -> ValidationReport:
    from PIL import Image

    rep = ValidationReport(ok=True)
    s = art.scene

    # --- images: one per test pose, exact stem, exact resolution, PNG
    for fr in s.test_frames:
        p = Path(art.rgb_dir) / f"{fr.stem}.png"
        if not p.exists():
            rep.errors.append(f"{s.key}: missing render {p.name}")
            continue
        try:
            with Image.open(p) as im:
                if im.format != "PNG":
                    rep.errors.append(f"{s.key}/{p.name}: format {im.format}, expected PNG")
                if im.size != (fr.camera.width, fr.camera.height):
                    rep.errors.append(
                        f"{s.key}/{p.name}: size {im.size}, expected "
                        f"{(fr.camera.width, fr.camera.height)}"
                    )
        except Exception as e:  # noqa: BLE001 - surface any unreadable file
            rep.errors.append(f"{s.key}/{p.name}: unreadable ({e})")

    extra = {p.stem for p in Path(art.rgb_dir).glob("*.png")} - {fr.stem for fr in s.test_frames}
    if extra:
        rep.warnings.append(f"{s.key}: {len(extra)} render(s) not in the reference set, dropped")

    # --- point cloud
    xyz = np.asarray(art.xyz)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        rep.errors.append(f"{s.key}: point cloud shape {xyz.shape}, expected (N, 3)")
    elif len(xyz) == 0:
        rep.errors.append(f"{s.key}: empty point cloud")
    elif not np.isfinite(xyz).all():
        rep.errors.append(f"{s.key}: point cloud contains non-finite coordinates")

    if s.needs_semantics:
        if art.labels is None:
            rep.errors.append(f"{s.key}: gold_coast scene requires a classification property")
        else:
            bad = set(np.unique(art.labels).tolist()) - VALID_LABELS
            if bad:
                rep.errors.append(f"{s.key}: invalid class values {sorted(bad)}")
            if len(art.labels) != len(xyz):
                rep.errors.append(f"{s.key}: {len(art.labels)} labels for {len(xyz)} points")
    elif art.labels is not None:
        rep.warnings.append(f"{s.key}: labels supplied for a TUM scene, they will be dropped")

    rep.ok = not rep.errors
    rep.stats = {"frames": len(s.test_frames), "points": int(len(xyz))}
    return rep


def build_submission(
    artefacts: list[SceneArtefacts],
    out_zip: Path,
    downsample: float | None = VOXEL_SIZE,
    require: str = "all",
) -> ValidationReport:
    """Validate everything, then write the zip. Nothing is written if invalid.

    `require` is "all", "dev", "final" or "none": which scenes must be present.
    The scorer voxel-downsamples at 2 cm, so submissions are downsampled to the
    same resolution, which costs no score and keeps uploads small.
    """
    out_zip = Path(out_zip)
    report = ValidationReport(ok=True)
    by_key = {a.scene.key: a for a in artefacts}

    required = {
        "all": ALL_SCENES, "dev": DEV_SCENES, "final": FINAL_SCENES, "none": [],
    }[require]
    for d, sid in required:
        if f"{d}/{sid}" not in by_key:
            report.errors.append(f"missing scene {d}/{sid}")

    per_scene = {}
    for art in artefacts:
        r = validate_scene(art, downsample)
        report.errors += r.errors
        report.warnings += r.warnings
        per_scene[art.scene.key] = r.stats

    report.ok = not report.errors
    if not report.ok:
        report.stats = {"scenes": len(artefacts), "written": False}
        return report

    out_zip.parent.mkdir(parents=True, exist_ok=True)
    total_frames = 0
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for art in artefacts:
            s = art.scene
            for fr in s.test_frames:
                src = Path(art.rgb_dir) / f"{fr.stem}.png"
                zf.write(src, f"{s.dataset}/{s.scene_id}/rgb/{fr.stem}.png")
                total_frames += 1

            xyz, labels = np.asarray(art.xyz), art.labels
            if not s.needs_semantics:
                labels = None
            if downsample:
                if labels is None:
                    xyz = voxel_downsample(xyz, downsample)
                else:
                    xyz, labels = voxel_downsample(xyz, downsample, np.asarray(labels))
            zf.writestr(
                f"{s.dataset}/{s.scene_id}/3D_point_cloud/point_cloud.ply",
                _ply_bytes(xyz, labels, grid=QUANTISE_GRID.get(s.dataset)),
            )
            per_scene[s.key]["points_submitted"] = int(len(xyz))

    report.stats = {
        "scenes": len(artefacts),
        "frames": total_frames,
        "zip_mb": round(out_zip.stat().st_size / 1e6, 1),
        "per_scene": per_scene,
        "written": True,
    }
    return report


def inspect_zip(path: Path) -> dict:
    """Summarise an existing submission zip, for checking what was uploaded."""
    from plyfile import PlyData

    out: dict = {"path": str(path), "mb": round(Path(path).stat().st_size / 1e6, 1), "scenes": {}}
    with zipfile.ZipFile(path) as zf:
        for name in zf.namelist():
            parts = name.split("/")
            if len(parts) < 3:
                continue
            key = f"{parts[0]}/{parts[1]}"
            sc = out["scenes"].setdefault(key, {"rgb": 0, "ply": None})
            if parts[2] == "rgb" and name.endswith(".png"):
                sc["rgb"] += 1
            elif name.endswith(".ply"):
                with zf.open(name) as f:
                    v = PlyData.read(io.BytesIO(f.read()))["vertex"]
                sc["ply"] = {
                    "n": len(v.data),
                    "props": [p.name for p in v.properties],
                }
    return out


# --------------------------------------------------------------------------
def submission_cloud(scene: Scene, fused_ply, voxel: float | None = None, cache=None):
    """The exact cloud that will be submitted for a scene, with its colours.

    Everything downstream works on this: semantic features, semantic labels, the
    local score, and the zip. Computing it once and sharing it is what keeps
    those four from drifting apart, and it is also a large speedup: the 2x Gold
    Coast clouds are 23-31 M points but collapse to a fraction of that at the
    2 cm voxel size the scorer uses, so features computed on the raw cloud are
    mostly work on points that never reach the scorer.

    Two steps, in order:
      robust_core   drop COLMAP's far-field tail, which cannot help either
                    metric because both crop to the ground-truth bounding box
      voxel         `SUBMISSION_VOXEL` for the dataset: 2 cm on TUM because the
                    F-score is computed at 2 cm, 6 cm on Gold Coast because only
                    a 10 cm match radius is ever asked about there

    Returns (xyz float64, rgb uint8 or None).
    """
    from plyfile import PlyData

    from .geometry.densify import robust_core

    if voxel is None:
        voxel = SUBMISSION_VOXEL[scene.dataset]
    if cache is not None:
        # the voxel size is part of the identity of the cached cloud
        cache = Path(cache)
        cache = cache.with_name(f"{cache.stem}_v{voxel}{cache.suffix}")
    if cache is not None and Path(cache).exists():
        d = np.load(cache)
        return d["xyz"].astype(np.float64), (d["rgb"] if "rgb" in d else None)

    v = PlyData.read(str(fused_ply))["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
    names = {pr.name for pr in v.properties}
    rgb = (
        np.stack([v["red"], v["green"], v["blue"]], axis=1).astype(np.uint8)
        if {"red", "green", "blue"} <= names
        else None
    )
    keep = robust_core(xyz, scene.train_frames)
    xyz, rgb = xyz[keep], (rgb[keep] if rgb is not None else None)
    idx = voxel_downsample_indices(xyz, voxel)
    xyz, rgb = xyz[idx], (rgb[idx] if rgb is not None else None)

    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        if rgb is None:
            _atomic_savez(cache, xyz=xyz.astype(np.float32))
        else:
            _atomic_savez(cache, xyz=xyz.astype(np.float32), rgb=rgb)
    return xyz, rgb
