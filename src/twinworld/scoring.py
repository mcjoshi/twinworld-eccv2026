"""Local reimplementation of the official TwinWorld scorer.

The official `scoring_v3` program is not downloadable (`make_programs_available`
is false on the competition), so this is written from the published Evaluation
page and calibrated against the first real submission score. Local numbers are
never reported as official; the calibration gap is recorded in
`docs/PROGRESS.md`.

Spec being implemented:

  rendering (all scenes)
    PSNR, SSIM (Wang et al., Gaussian-weighted), LPIPS (VGG)
  geometry (TUM only)
    crop GT and submission to the GT bounding box + 20 cm, voxel-downsample at
    2 cm, then precision/recall F-scores at 5/10/20 cm; report the mean
  semantics (Gold Coast only)
    each GT point takes the label of the nearest submitted point within 10 cm,
    unmatched GT points count as errors; per-class IoU pooled over scenes;
    mIoU is the unweighted mean over the 5 classes

  rendering_score = mean(PSNR / 50 capped at 1, SSIM, 1 - LPIPS)
  geometry_score  = mean(F@5, F@10, F@20)
  semantic_score  = mIoU 3D
  Final Score     = rendering_score + geometry_score + semantic_score
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .dataio import IGNORE_LABEL, NUM_CLASSES

BBOX_PADDING = 0.20  # metres
VOXEL_SIZE = 0.02  # metres
F_THRESHOLDS = (0.05, 0.10, 0.20)  # metres
SEMANTIC_MATCH_RADIUS = 0.10  # metres
PSNR_CAP = 50.0


# --------------------------------------------------------------------------
# point cloud helpers
# --------------------------------------------------------------------------
def read_ply_xyz(path: Path, with_classification: bool = False):
    """Read a PLY's vertex element. Returns (xyz float64, labels uint8 or None)."""
    from plyfile import PlyData

    v = PlyData.read(str(path))["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
    labels = None
    if with_classification:
        names = {p.name for p in v.properties}
        for cand in ("classification", "class", "label", "scalar_Classification"):
            if cand in names:
                labels = np.asarray(v[cand]).astype(np.int32)
                break
    return xyz, labels


def write_ply_xyz(path: Path, xyz: np.ndarray, labels: np.ndarray | None = None) -> None:
    """Write the exact submission PLY format: binary little-endian, float32 xyz,
    optional uchar classification, nothing else."""
    from plyfile import PlyData, PlyElement

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [("x", "f4"), ("y", "f4"), ("z", "f4")]
    if labels is not None:
        fields.append(("classification", "u1"))
    vertex = np.empty(len(xyz), dtype=fields)
    xyz = np.asarray(xyz, dtype=np.float32)
    vertex["x"], vertex["y"], vertex["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    if labels is not None:
        vertex["classification"] = np.asarray(labels).astype(np.uint8)
    PlyData([PlyElement.describe(vertex, "vertex")], text=False, byte_order="<").write(str(path))


def voxel_downsample_indices(xyz: np.ndarray, voxel: float = VOXEL_SIZE) -> np.ndarray:
    """Indices of the kept points: one per occupied voxel, nearest its centre.

    Returned as indices so any per-point attribute (colour, labels, normals) can
    be carried through the same selection.
    """
    if len(xyz) == 0:
        return np.empty(0, dtype=np.int64)
    keys = np.floor(np.asarray(xyz) / voxel).astype(np.int64)
    _, inv = np.unique(keys, axis=0, return_inverse=True)
    centres = (keys + 0.5) * voxel
    d = np.linalg.norm(xyz - centres, axis=1)
    order = np.lexsort((d, inv))
    first = np.zeros(len(order), dtype=bool)
    first[0] = True
    first[1:] = inv[order][1:] != inv[order][:-1]
    keep = order[first]
    keep.sort()
    return keep


def voxel_downsample(xyz: np.ndarray, voxel: float = VOXEL_SIZE, labels: np.ndarray | None = None):
    """Keep one point per occupied voxel: the point closest to the voxel centre.

    Deterministic, and matches Open3D's `voxel_down_sample` closely enough for
    F-scores at 5 cm and coarser. Implemented directly so the behaviour is
    explicit and label carrying is well defined.
    """
    if len(xyz) == 0:
        return (xyz, labels) if labels is not None else xyz
    keys = np.floor(xyz / voxel).astype(np.int64)
    _, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    # pick, per voxel, the point nearest that voxel's centre
    centres = (keys + 0.5) * voxel
    d = np.linalg.norm(xyz - centres, axis=1)
    order = np.lexsort((d, inv))
    first = np.zeros(len(order), dtype=bool)
    first[0] = True
    first[1:] = inv[order][1:] != inv[order][:-1]
    keep = order[first]
    keep.sort()
    if labels is not None:
        return xyz[keep], labels[keep]
    return xyz[keep]


def crop_to_bbox(xyz: np.ndarray, lo: np.ndarray, hi: np.ndarray, labels: np.ndarray | None = None):
    m = np.all((xyz >= lo) & (xyz <= hi), axis=1)
    if labels is not None:
        return xyz[m], labels[m]
    return xyz[m]


def _kdtree(xyz: np.ndarray):
    from scipy.spatial import cKDTree

    return cKDTree(xyz)


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------
@dataclass
class GeometryResult:
    scene: str
    f_scores: dict[float, float]
    precision: dict[float, float]
    recall: dict[float, float]
    n_gt: int
    n_pred: int

    @property
    def fscore(self) -> float:
        return float(np.mean([self.f_scores[t] for t in F_THRESHOLDS]))


def score_geometry(pred_xyz: np.ndarray, gt_xyz: np.ndarray, scene: str = "") -> GeometryResult:
    lo = gt_xyz.min(axis=0) - BBOX_PADDING
    hi = gt_xyz.max(axis=0) + BBOX_PADDING

    gt = voxel_downsample(crop_to_bbox(gt_xyz, lo, hi))
    pred = voxel_downsample(crop_to_bbox(pred_xyz, lo, hi))

    f_scores, precision, recall = {}, {}, {}
    if len(pred) == 0 or len(gt) == 0:
        for t in F_THRESHOLDS:
            f_scores[t] = precision[t] = recall[t] = 0.0
        return GeometryResult(scene, f_scores, precision, recall, len(gt), len(pred))

    d_pred, _ = _kdtree(gt).query(pred, workers=-1)  # pred -> nearest gt : precision
    d_gt, _ = _kdtree(pred).query(gt, workers=-1)  # gt -> nearest pred : recall
    for t in F_THRESHOLDS:
        p = float((d_pred < t).mean())
        r = float((d_gt < t).mean())
        precision[t], recall[t] = p, r
        f_scores[t] = 0.0 if (p + r) == 0 else 2 * p * r / (p + r)
    return GeometryResult(scene, f_scores, precision, recall, len(gt), len(pred))


# --------------------------------------------------------------------------
# semantics
# --------------------------------------------------------------------------
@dataclass
class SemanticCounts:
    """Confusion accumulators pooled over scenes, as the spec requires."""

    intersection: np.ndarray = field(default_factory=lambda: np.zeros(NUM_CLASSES, np.int64))
    union: np.ndarray = field(default_factory=lambda: np.zeros(NUM_CLASSES, np.int64))
    # per-class counts of ground-truth points, and of those with any submitted
    # point within the match radius. Their ratio is the per-class ceiling on IoU.
    gt_per_class: np.ndarray = field(default_factory=lambda: np.zeros(NUM_CLASSES, np.int64))
    matched_per_class: np.ndarray = field(default_factory=lambda: np.zeros(NUM_CLASSES, np.int64))
    n_unmatched: int = 0
    n_total: int = 0

    def add(self, other: "SemanticCounts") -> None:
        self.intersection += other.intersection
        self.union += other.union
        self.gt_per_class += other.gt_per_class
        self.matched_per_class += other.matched_per_class
        self.n_unmatched += other.n_unmatched
        self.n_total += other.n_total

    @property
    def per_class_coverage(self) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            cov = self.matched_per_class / self.gt_per_class
        return np.where(self.gt_per_class > 0, cov, np.nan)

    @property
    def miou_ceiling(self) -> float:
        """The best mIoU any labelling of this point cloud could reach.

        An unmatched ground-truth point can never enter its class's intersection
        but always enters its union, so `IoU_c <= coverage_c`. This separates a
        classification problem from a reconstruction-coverage problem.
        """
        cov = self.per_class_coverage
        return float(np.nanmean(cov)) if np.any(~np.isnan(cov)) else 0.0

    @property
    def per_class_iou(self) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            iou = self.intersection / self.union
        return np.where(self.union > 0, iou, np.nan)

    @property
    def miou(self) -> float:
        iou = self.per_class_iou
        # a class absent from the GT contributes nothing; mean over present classes
        return float(np.nanmean(iou)) if np.any(~np.isnan(iou)) else 0.0


def score_semantics(
    pred_xyz: np.ndarray,
    pred_labels: np.ndarray,
    gt_xyz: np.ndarray,
    gt_labels: np.ndarray,
    radius: float = SEMANTIC_MATCH_RADIUS,
) -> SemanticCounts:
    """Transfer predicted labels to GT points by nearest neighbour within `radius`.

    Unmatched GT points count as errors: they are added to the union of their own
    true class but never to any intersection, exactly as an incorrect prediction
    would be, without polluting another class's union.
    """
    valid = gt_labels != IGNORE_LABEL
    gt_xyz, gt_labels = gt_xyz[valid], gt_labels[valid]

    counts = SemanticCounts(n_total=len(gt_xyz))
    for c in range(NUM_CLASSES):
        counts.gt_per_class[c] = int((gt_labels == c).sum())
    if len(gt_xyz) == 0:
        return counts
    if len(pred_xyz) == 0:
        counts.union[:] = counts.gt_per_class
        counts.n_unmatched = len(gt_xyz)
        return counts

    dist, idx = _kdtree(pred_xyz).query(gt_xyz, workers=-1)
    matched = dist <= radius
    transferred = np.full(len(gt_xyz), -1, dtype=np.int64)
    transferred[matched] = pred_labels[idx[matched]]
    # a predicted `ignore` is not a valid class prediction, treat it as unmatched
    transferred[transferred == IGNORE_LABEL] = -1
    counts.n_unmatched = int((transferred < 0).sum())

    for c in range(NUM_CLASSES):
        gt_c = gt_labels == c
        pr_c = transferred == c
        counts.intersection[c] = int((gt_c & pr_c).sum())
        counts.union[c] = int((gt_c | pr_c).sum())
        counts.matched_per_class[c] = int((gt_c & (transferred >= 0)).sum())
    return counts


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------
def _gaussian_window(size: int = 11, sigma: float = 1.5):
    import torch

    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    return (g[:, None] @ g[None, :]).unsqueeze(0).unsqueeze(0)


def ssim(img1, img2, window_size: int = 11):
    """The 3DGS/Wang et al. Gaussian-weighted SSIM. Inputs are (C, H, W) in [0, 1]."""
    import torch
    import torch.nn.functional as F

    C = img1.shape[-3]
    window = _gaussian_window(window_size).expand(C, 1, window_size, window_size).to(img1)
    pad = window_size // 2
    mu1 = F.conv2d(img1, window, padding=pad, groups=C)
    mu2 = F.conv2d(img2, window, padding=pad, groups=C)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=C) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=C) - mu1_mu2
    C1, C2 = 0.01**2, 0.03**2
    m = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return m.mean()


def psnr(img1, img2):
    import torch

    mse = ((img1 - img2) ** 2).flatten(1).mean(1, keepdim=True)
    return (20 * torch.log10(1.0 / torch.sqrt(mse))).mean()


class LpipsVgg:
    """Lazily constructed so importing this module never touches the GPU."""

    _model = None

    @classmethod
    def get(cls, device="cuda"):
        if cls._model is None:
            import lpips

            cls._model = lpips.LPIPS(net="vgg").to(device).eval()
        return cls._model


def load_image(path: Path):
    import torch
    from PIL import Image

    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


@dataclass
class RenderResult:
    scene: str
    psnr: float
    ssim: float
    lpips: float
    n_frames: int
    per_frame: dict[str, dict[str, float]] = field(default_factory=dict)


def score_rendering(pred_dir: Path, gt_frames, device: str = "cuda") -> RenderResult:
    """`gt_frames` is a list of `dataio.Frame` with real image paths."""
    import torch

    model = LpipsVgg.get(device)
    ps, ss, lp, per = [], [], [], {}
    for fr in gt_frames:
        pred_path = Path(pred_dir) / f"{fr.stem}.png"
        if not pred_path.exists():
            raise FileNotFoundError(f"missing render {pred_path}")
        gt = load_image(fr.image_path).to(device)[None]
        pr = load_image(pred_path).to(device)[None]
        if pr.shape != gt.shape:
            raise ValueError(f"{pred_path}: shape {tuple(pr.shape)} != reference {tuple(gt.shape)}")
        with torch.no_grad():
            p = float(psnr(pr, gt))
            s = float(ssim(pr, gt))
            # LPIPS expects [-1, 1]
            l = float(model(pr * 2 - 1, gt * 2 - 1).mean())
        ps.append(p), ss.append(s), lp.append(l)
        per[fr.stem] = {"psnr": p, "ssim": s, "lpips": l}
    return RenderResult("", float(np.mean(ps)), float(np.mean(ss)), float(np.mean(lp)), len(ps), per)


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------
def rendering_score(psnr_v: float, ssim_v: float, lpips_v: float) -> float:
    return float(np.mean([min(psnr_v / PSNR_CAP, 1.0), ssim_v, 1.0 - lpips_v]))


def final_score(render_s: float, geom_s: float, sem_s: float) -> float:
    return float(render_s + geom_s + sem_s)
