"""Tests for the parts a wrong answer would silently corrupt a submission.

The expensive parts (MVS, training) are not tested here; what is tested is
everything that decides whether a submission is accepted and how it is scored.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pytest

from twinworld import dataio
from twinworld.dataio import ALL_SCENES, Scene, qvec2rotmat, rotmat2qvec
from twinworld.package import SceneArtefacts, build_submission, validate_scene
from twinworld.scoring import (
    crop_to_bbox, read_ply_xyz, score_geometry, score_semantics, voxel_downsample,
    write_ply_xyz, rendering_score, final_score,
)

DATA_PRESENT = (dataio.DATA_ROOT / "Data_TUM" / "scene_000").is_dir()
needs_data = pytest.mark.skipif(not DATA_PRESENT, reason="dataset not downloaded")


# --------------------------------------------------------------------------
# dataio
# --------------------------------------------------------------------------
@needs_data
@pytest.mark.parametrize("dataset,scene_id", ALL_SCENES)
def test_frame_stems_match_reference_images(dataset, scene_id):
    """Frame stems and resolutions must come from the reference poses, for all
    13 scenes. A mismatch here is an `incomplete_submission` rejection."""
    s = Scene(dataset, scene_id)
    assert len(s.test_frames) == 3
    stems = [f.stem for f in s.test_frames]
    assert len(set(stems)) == len(stems), "duplicate frame stems"
    for fr in s.test_frames:
        assert fr.camera.model == "PINHOLE"
        assert fr.camera.width > 0 and fr.camera.height > 0
        if fr.image_path is not None:
            from PIL import Image

            with Image.open(fr.image_path) as im:
                assert im.size == (fr.camera.width, fr.camera.height)


@needs_data
def test_dev_scenes_have_gt_and_final_do_not():
    for s in dataio.dev_scenes():
        assert s.has_gt_cloud and s.has_test_images
    for s in dataio.final_scenes():
        assert not s.has_gt_cloud and not s.has_test_images


def test_qvec_rotmat_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(20):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        if q[0] < 0:
            q = -q
        assert np.allclose(q, rotmat2qvec(qvec2rotmat(q)), atol=1e-8)


@needs_data
def test_camera_centre_is_consistent_with_projection():
    """A camera projects its own centre to a zero-depth point, and the frustum
    test must accept a point straight ahead of the camera."""
    s = Scene("tum", "scene_000")
    fr = s.train_frames[0]
    forward = fr.R.T @ np.array([0.0, 0.0, 1.0])
    ahead = fr.center + forward * 30.0
    assert dataio._frustum_mask(ahead[None], [fr], 0.0)[0]
    behind = fr.center - forward * 30.0
    assert not dataio._frustum_mask(behind[None], [fr], 0.0)[0]


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------
def _plane_grid(n=200, spacing=0.05):
    a = np.arange(n) * spacing
    x, y = np.meshgrid(a, a)
    return np.stack([x.ravel(), y.ravel(), np.zeros(x.size)], axis=1)


def test_identical_clouds_score_one():
    gt = _plane_grid()
    r = score_geometry(gt.copy(), gt)
    assert r.fscore == pytest.approx(1.0)


def test_offset_beyond_largest_threshold_scores_zero():
    """A structured surface shifted 30 cm along its own normal must miss all
    three thresholds. Random clouds cannot test this: they have neighbours
    everywhere."""
    gt = _plane_grid()
    pred = gt + np.array([0.0, 0.0, 0.30])
    r = score_geometry(pred, gt)
    assert r.fscore == pytest.approx(0.0, abs=1e-9)


def test_offset_between_thresholds():
    gt = _plane_grid()
    pred = gt + np.array([0.0, 0.0, 0.07])
    r = score_geometry(pred, gt)
    assert r.f_scores[0.05] == pytest.approx(0.0, abs=1e-9)
    assert r.f_scores[0.10] == pytest.approx(1.0)
    assert r.f_scores[0.20] == pytest.approx(1.0)


def test_empty_prediction_scores_zero():
    assert score_geometry(np.empty((0, 3)), _plane_grid()).fscore == 0.0


def test_points_outside_the_gt_bbox_are_free():
    """The scorer crops to the GT bbox + 20 cm, so surplus points far away must
    not cost precision. This is what makes generous reconstruction safe on the
    withheld final scenes."""
    gt = _plane_grid()
    junk = np.random.default_rng(0).uniform(100, 200, size=(50_000, 3))
    assert score_geometry(np.vstack([gt, junk]), gt).fscore == pytest.approx(1.0)


def test_points_just_inside_the_padding_do_count():
    gt = _plane_grid()
    near = gt[:100] + np.array([0.0, 0.0, 0.15])  # inside the 20 cm padding
    r = score_geometry(np.vstack([gt, near]), gt)
    assert r.precision[0.05] < 1.0


def test_voxel_downsample_is_idempotent_and_bounded():
    rng = np.random.default_rng(0)
    xyz = rng.uniform(0, 1, (100_000, 3))
    a = voxel_downsample(xyz, 0.02)
    b = voxel_downsample(a, 0.02)
    assert len(a) == len(b)
    assert len(a) <= len(xyz)
    d = np.linalg.norm(a[:, None, :] - a[None, : min(50, len(a)), :], axis=-1)
    assert np.isfinite(d).all()


def test_voxel_downsample_carries_labels():
    xyz = _plane_grid(50, 0.05)
    labels = np.arange(len(xyz), dtype=np.uint8) % 5
    x2, l2 = voxel_downsample(xyz, 0.02, labels)
    assert len(x2) == len(l2)


def test_semantics_perfect_and_worst_cases():
    xyz = _plane_grid(60)
    lab = (np.arange(len(xyz)) % 5).astype(np.uint8)
    assert score_semantics(xyz, lab, xyz, lab).miou == pytest.approx(1.0)

    far = score_semantics(xyz + 100.0, lab, xyz, lab)
    assert far.miou == pytest.approx(0.0)
    assert far.n_unmatched == len(xyz)


def test_semantics_ignores_255_in_ground_truth():
    xyz = _plane_grid(40)
    lab = (np.arange(len(xyz)) % 5).astype(np.uint8)
    gt = lab.copy()
    gt[: len(gt) // 2] = 255
    c = score_semantics(xyz, lab, xyz, gt)
    assert c.n_total == len(xyz) - len(xyz) // 2
    assert c.miou == pytest.approx(1.0)


def test_semantics_unmatched_points_count_against_their_true_class():
    """A GT region with no prediction within 10 cm must reduce its own class's
    IoU rather than vanish from the metric.

    The two halves are placed far apart, not split from one contiguous grid:
    adjacent rows of a 5 cm grid are inside the 10 cm matching radius, so a
    contiguous split leaves a matched band along the seam.
    """
    covered = _plane_grid(40)
    orphan = _plane_grid(40) + np.array([1000.0, 0.0, 0.0])
    xyz = np.vstack([covered, orphan])
    lab = np.zeros(len(xyz), np.uint8)

    c = score_semantics(covered, lab[: len(covered)], xyz, lab)
    assert c.n_unmatched == len(orphan)
    assert c.per_class_iou[0] == pytest.approx(len(covered) / len(xyz), rel=1e-6)


def test_score_formulas():
    assert rendering_score(50.0, 1.0, 0.0) == pytest.approx(1.0)
    assert rendering_score(100.0, 1.0, 0.0) == pytest.approx(1.0)  # PSNR capped
    assert rendering_score(25.0, 0.5, 0.5) == pytest.approx(0.5)
    assert final_score(1.0, 1.0, 1.0) == pytest.approx(3.0)


def test_ply_roundtrip_uses_submission_dtypes(tmp_path):
    xyz = np.random.default_rng(0).uniform(-10, 10, (1000, 3))
    lab = np.random.default_rng(0).integers(0, 5, 1000).astype(np.uint8)
    p = tmp_path / "pc.ply"
    write_ply_xyz(p, xyz, lab)
    from plyfile import PlyData

    v = PlyData.read(str(p))["vertex"]
    assert [pr.name for pr in v.properties] == ["x", "y", "z", "classification"]
    back, back_lab = read_ply_xyz(p, with_classification=True)
    assert np.allclose(back, xyz, atol=1e-5)
    assert (back_lab == lab).all()


# --------------------------------------------------------------------------
# packaging
# --------------------------------------------------------------------------
def _fake_renders(tmp_path: Path, s: Scene, size=None, fmt="PNG") -> Path:
    from PIL import Image

    d = tmp_path / s.scene_id / "rgb"
    d.mkdir(parents=True, exist_ok=True)
    for fr in s.test_frames:
        wh = size or (fr.camera.width, fr.camera.height)
        Image.fromarray(np.zeros((wh[1], wh[0], 3), np.uint8)).save(
            d / f"{fr.stem}.{'png' if fmt == 'PNG' else 'jpg'}", format=fmt
        )
    return d


@needs_data
def test_validator_rejects_wrong_resolution(tmp_path):
    s = Scene("tum", "scene_000")
    art = SceneArtefacts(s, _fake_renders(tmp_path, s, size=(800, 600)),
                         np.zeros((10, 3)))
    rep = validate_scene(art)
    assert not rep.ok and any("size" in e for e in rep.errors)


@needs_data
def test_validator_rejects_missing_frame(tmp_path):
    s = Scene("tum", "scene_000")
    d = _fake_renders(tmp_path, s)
    next(d.glob("*.png")).unlink()
    rep = validate_scene(SceneArtefacts(s, d, np.zeros((10, 3))))
    assert not rep.ok and any("missing render" in e for e in rep.errors)


@needs_data
def test_validator_rejects_gold_coast_without_labels(tmp_path):
    s = Scene("gold_coast", "scene_009")
    rep = validate_scene(SceneArtefacts(s, _fake_renders(tmp_path, s), np.zeros((10, 3))))
    assert not rep.ok and any("classification" in e for e in rep.errors)


@needs_data
def test_validator_rejects_invalid_class_values(tmp_path):
    s = Scene("gold_coast", "scene_009")
    rep = validate_scene(
        SceneArtefacts(s, _fake_renders(tmp_path, s), np.zeros((10, 3)),
                       np.full(10, 7, np.uint8))
    )
    assert not rep.ok and any("invalid class" in e for e in rep.errors)


@needs_data
def test_validator_rejects_non_finite_points(tmp_path):
    s = Scene("tum", "scene_000")
    xyz = np.zeros((10, 3))
    xyz[3, 1] = np.nan
    rep = validate_scene(SceneArtefacts(s, _fake_renders(tmp_path, s), xyz))
    assert not rep.ok and any("non-finite" in e for e in rep.errors)


@needs_data
def test_invalid_submission_writes_nothing(tmp_path):
    s = Scene("tum", "scene_000")
    art = SceneArtefacts(s, _fake_renders(tmp_path, s, size=(10, 10)), np.zeros((10, 3)))
    out = tmp_path / "sub.zip"
    rep = build_submission([art], out, require="none")
    assert not rep.ok
    assert not out.exists(), "a rejected submission must not leave a zip behind"


@needs_data
def test_build_submission_matches_the_official_dummy_layout(tmp_path):
    """The starting kit's dummy generator is the reference for the zip layout."""
    scenes = [Scene("tum", "scene_000"), Scene("gold_coast", "scene_009")]
    arts = []
    for s in scenes:
        n = 500
        xyz = np.random.default_rng(0).uniform(-5, 5, (n, 3))
        lab = np.random.default_rng(0).integers(0, 5, n).astype(np.uint8) if s.needs_semantics else None
        arts.append(SceneArtefacts(s, _fake_renders(tmp_path, s), xyz, lab))

    out = tmp_path / "sub.zip"
    rep = build_submission(arts, out, require="none")
    assert rep.ok, rep
    with zipfile.ZipFile(out) as zf:
        names = set(zf.namelist())
    for s in scenes:
        assert f"{s.dataset}/{s.scene_id}/3D_point_cloud/point_cloud.ply" in names
        for fr in s.test_frames:
            assert f"{s.dataset}/{s.scene_id}/rgb/{fr.stem}.png" in names
    assert not any(n.startswith("submission/") for n in names), "no extra top-level folder"


@needs_data
def test_require_all_flags_missing_scenes(tmp_path):
    s = Scene("tum", "scene_000")
    art = SceneArtefacts(s, _fake_renders(tmp_path, s), np.zeros((10, 3)))
    rep = build_submission([art], tmp_path / "x.zip", require="all")
    assert not rep.ok
    assert sum("missing scene" in e for e in rep.errors) == len(ALL_SCENES) - 1


def test_miou_ceiling_equals_mean_per_class_coverage():
    """The ceiling must bound the achievable mIoU, and must be reached when the
    labelling of the covered part is perfect."""
    covered = _plane_grid(40)
    orphan = _plane_grid(40) + np.array([1000.0, 0.0, 0.0])
    xyz = np.vstack([covered, orphan])
    lab = np.zeros(len(xyz), np.uint8)
    lab[len(covered):] = 1  # the uncovered half is a different class

    c = score_semantics(covered, lab[: len(covered)], xyz, lab)
    cov = c.per_class_coverage
    assert cov[0] == pytest.approx(1.0)   # fully covered
    assert cov[1] == pytest.approx(0.0)   # entirely missing
    assert c.miou_ceiling == pytest.approx(0.5)
    assert c.miou <= c.miou_ceiling + 1e-9
    assert c.miou == pytest.approx(0.5)   # perfect labels on the covered class


def test_miou_never_exceeds_its_ceiling_under_random_labels():
    rng = np.random.default_rng(0)
    gt_xyz = rng.uniform(0, 5, (20_000, 3))
    gt_lab = rng.integers(0, 5, 20_000).astype(np.uint8)
    keep = rng.random(20_000) < 0.4
    pred_lab = rng.integers(0, 5, int(keep.sum())).astype(np.uint8)
    c = score_semantics(gt_xyz[keep], pred_lab, gt_xyz, gt_lab)
    assert c.miou <= c.miou_ceiling + 1e-9
