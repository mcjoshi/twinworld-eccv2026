"""COLMAP parsing and the scene registry.

Single source of truth for frame IDs, resolutions and poses. Everything that
produces a submission artefact must go through `Scene`, so the frame stems and
image sizes in a submission can never drift from the reference data.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data"

# submission dataset name -> dataset directory name
DATASETS = {"tum": "Data_TUM", "gold_coast": "Data_Goldcoast"}

TUM_DEV = [f"scene_{i:03d}" for i in range(0, 4)]
TUM_FINAL = [f"scene_{i:03d}" for i in range(4, 9)]
GC_DEV = ["scene_009", "scene_010"]
GC_FINAL = ["scene_011", "scene_012"]

DEV_SCENES = [("tum", s) for s in TUM_DEV] + [("gold_coast", s) for s in GC_DEV]
FINAL_SCENES = [("tum", s) for s in TUM_FINAL] + [("gold_coast", s) for s in GC_FINAL]
ALL_SCENES = DEV_SCENES + FINAL_SCENES

CLASS_NAMES = {0: "ground", 1: "wall", 2: "roof", 3: "window", 4: "other", 255: "ignore"}
NUM_CLASSES = 5
IGNORE_LABEL = 255


@dataclass(frozen=True)
class Camera:
    camera_id: int
    model: str
    width: int
    height: int
    params: np.ndarray

    @property
    def fx(self) -> float:
        return float(self.params[0])

    @property
    def fy(self) -> float:
        return float(self.params[1]) if self.model == "PINHOLE" else float(self.params[0])

    @property
    def cx(self) -> float:
        return float(self.params[2]) if self.model == "PINHOLE" else float(self.params[1])

    @property
    def cy(self) -> float:
        return float(self.params[3]) if self.model == "PINHOLE" else float(self.params[2])

    @property
    def K(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]], dtype=np.float64
        )


@dataclass(frozen=True)
class Frame:
    """One posed view. `R`/`t` map world points to camera: x_cam = R @ x_world + t."""

    image_id: int
    name: str
    qvec: np.ndarray
    tvec: np.ndarray
    camera: Camera
    image_path: Path | None

    @property
    def stem(self) -> str:
        """The frame_id the scorer matches on: the images.txt NAME column's stem."""
        return Path(self.name).stem

    @property
    def R(self) -> np.ndarray:
        return qvec2rotmat(self.qvec)

    @property
    def t(self) -> np.ndarray:
        return self.tvec

    @property
    def center(self) -> np.ndarray:
        """Camera centre in world coordinates."""
        return -self.R.T @ self.tvec

    @property
    def world_to_cam(self) -> np.ndarray:
        M = np.eye(4)
        M[:3, :3] = self.R
        M[:3, 3] = self.tvec
        return M


def qvec2rotmat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
        ]
    )


def rotmat2qvec(R: np.ndarray) -> np.ndarray:
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = (
        np.array(
            [
                [Rxx - Ryy - Rzz, 0, 0, 0],
                [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
                [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
                [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz],
            ]
        )
        / 3.0
    )
    vals, vecs = np.linalg.eigh(K)
    qvec = vecs[[3, 0, 1, 2], np.argmax(vals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def read_cameras_text(path: Path) -> dict[int, Camera]:
    cameras = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = line.split()
            cid = int(p[0])
            cameras[cid] = Camera(
                camera_id=cid,
                model=p[1],
                width=int(p[2]),
                height=int(p[3]),
                params=np.array([float(v) for v in p[4:]]),
            )
    return cameras


def read_images_text(path: Path, cameras: dict[int, Camera]) -> list[Frame]:
    """Each image occupies two lines; the second (POINTS2D) line is skipped."""
    with open(path) as f:
        lines = [ln for ln in f if not ln.startswith("#")]
    frames = []
    for i in range(0, len(lines), 2):
        p = lines[i].split()
        if not p:
            continue
        frames.append(
            Frame(
                image_id=int(p[0]),
                name=p[9],
                qvec=np.array([float(v) for v in p[1:5]]),
                tvec=np.array([float(v) for v in p[5:8]]),
                camera=cameras[int(p[8])],
                image_path=None,
            )
        )
    frames.sort(key=lambda fr: fr.name)
    return frames


class Scene:
    """One challenge scene, addressed by (dataset, scene_id).

    `dataset` is the submission-level name ("tum" / "gold_coast"), which is also
    the top-level folder name inside a submission zip.
    """

    def __init__(self, dataset: str, scene_id: str, data_root: Path = DATA_ROOT):
        if dataset not in DATASETS:
            raise ValueError(f"unknown dataset {dataset!r}, expected one of {list(DATASETS)}")
        self.dataset = dataset
        self.scene_id = scene_id
        self.root = data_root / DATASETS[dataset] / scene_id
        if not self.root.is_dir():
            raise FileNotFoundError(f"scene directory not found: {self.root}")

    def __repr__(self) -> str:
        return f"Scene({self.dataset}/{self.scene_id})"

    @property
    def key(self) -> str:
        return f"{self.dataset}/{self.scene_id}"

    # --- poses -----------------------------------------------------------
    def _frames(self, split: str) -> list[Frame]:
        sparse = self.root / split / "sparse" / "0"
        cameras = read_cameras_text(sparse / "cameras.txt")
        frames = read_images_text(sparse / "images.txt", cameras)
        img_dir = self.root / split / "images"
        out = []
        for fr in frames:
            path = img_dir / fr.name
            out.append(
                Frame(
                    fr.image_id, fr.name, fr.qvec, fr.tvec, fr.camera, path if path.exists() else None
                )
            )
        return out

    @functools.cached_property
    def train_frames(self) -> list[Frame]:
        return self._frames("train")

    @functools.cached_property
    def test_frames(self) -> list[Frame]:
        return self._frames("test")

    @property
    def train_images_dir(self) -> Path:
        return self.root / "train" / "images"

    @property
    def train_sparse_dir(self) -> Path:
        return self.root / "train" / "sparse" / "0"

    @property
    def test_sparse_dir(self) -> Path:
        return self.root / "test" / "sparse" / "0"

    # --- ground truth ----------------------------------------------------
    @property
    def gt_ply(self) -> Path | None:
        p = self.root / "3d_gt" / "point_cloud.ply"
        return p if p.exists() else None

    @property
    def has_gt_cloud(self) -> bool:
        return self.gt_ply is not None

    @property
    def has_test_images(self) -> bool:
        return all(fr.image_path is not None for fr in self.test_frames)

    @property
    def is_dev(self) -> bool:
        return (self.dataset, self.scene_id) in DEV_SCENES

    @property
    def needs_semantics(self) -> bool:
        return self.dataset == "gold_coast"

    # --- derived geometry ------------------------------------------------
    @property
    def sparse_ply(self) -> Path:
        return self.train_sparse_dir / "points3D.ply"

    def train_frustum_mask(self, xyz: np.ndarray, margin_px: float = 0.0) -> np.ndarray:
        """Boolean mask: is each world point inside at least one train-view frustum?

        Measured on the six dev scenes, train frustums contain 97.2-100% of GT
        points, which is why this is the region rule for the withheld final
        scenes (design spec section 3.1.3).
        """
        return _frustum_mask(xyz, self.train_frames, margin_px)

    def test_frustum_mask(self, xyz: np.ndarray, margin_px: float = 0.0) -> np.ndarray:
        return _frustum_mask(xyz, self.test_frames, margin_px)


def _frustum_mask(xyz: np.ndarray, frames: list[Frame], margin_px: float) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float64)
    seen = np.zeros(len(xyz), dtype=bool)
    for fr in frames:
        cam = xyz @ fr.R.T + fr.t
        z = cam[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = fr.camera.fx * cam[:, 0] / z + fr.camera.cx
            v = fr.camera.fy * cam[:, 1] / z + fr.camera.cy
        seen |= (
            (z > 0)
            & (u >= -margin_px)
            & (u < fr.camera.width + margin_px)
            & (v >= -margin_px)
            & (v < fr.camera.height + margin_px)
        )
    return seen


def all_scenes(data_root: Path = DATA_ROOT) -> list[Scene]:
    return [Scene(d, s, data_root) for d, s in ALL_SCENES]


def dev_scenes(data_root: Path = DATA_ROOT) -> list[Scene]:
    return [Scene(d, s, data_root) for d, s in DEV_SCENES]


def final_scenes(data_root: Path = DATA_ROOT) -> list[Scene]:
    return [Scene(d, s, data_root) for d, s in FINAL_SCENES]


def get_scene(scene_id: str, data_root: Path = DATA_ROOT) -> Scene:
    """Look up a scene by bare id, e.g. "scene_004"."""
    for d, s in ALL_SCENES:
        if s == scene_id:
            return Scene(d, s, data_root)
    raise KeyError(scene_id)
