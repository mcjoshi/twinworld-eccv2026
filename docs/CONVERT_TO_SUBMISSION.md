# Converting the reconstruction outputs into the Codabench submission

This note accompanies the released reconstruction outputs and describes how they
become `v20-mid-final.zip` (Codabench submission **910133**, final score
**1.4837**, md5 `88a3a3cbc51b7d6744c66210e73dbf53`).

## Checksums, and why the provided archive has a different one

The zip uploaded to Codabench has md5 `88a3a3cbc51b7d6744c66210e73dbf53`. The
copy released here, `submission/v20-mid-final.zip`, has md5
`c99ffd0abd55c4c7dfb629788c074b99`.

They differ in ZIP container metadata only - an archive stores a modification
timestamp per entry, so rebuilding the same payload never reproduces the same
bytes. The payload is identical: 2 253 MB in both, thirteen scenes in both, and
the same point count in every scored cloud -

    gold_coast/scene_011   142.6 M      tum/scene_004    7.4 M
    gold_coast/scene_012    83.3 M      tum/scene_005   11.7 M
                                        tum/scene_006   10.0 M
                                        tum/scene_007   10.0 M
                                        tum/scene_008   10.5 M

The reason a rebuild exists at all is worth stating plainly: the released copy
was produced by running the two commands in section 2 below against the released
checkpoints. Regenerating the submission *is* the reproducibility check, and
running those commands again will reproduce this archive's payload.

The submission has two kinds of content per scene: rendered PNGs for the test
poses, and a point cloud PLY. They come from different branches.

```
Gaussians (splats.pt, 8 per scene)  --render_frames-->  per-member PNGs
                                    --pixel average-->  submitted PNGs

MVS cloud + semantic labels         --sheets-------->   submitted Gold Coast PLY
MVS cloud                           --2 cm voxel---->   submitted TUM PLY
```

## 1. Gaussians to submitted renders

Each scene has eight Gaussian models, `member{0..7}/splats.pt`, trained with
`seed = i` and otherwise identical configuration (recorded in each member's
`train_summary.json`). Rendering a member and averaging is what
`scripts/render_ensemble.py` does; the equivalent standalone form is:

```python
import numpy as np
from PIL import Image
from twinworld.dataio import Scene
from twinworld.render.trainer import TrainConfig, render_frames

s = Scene("gold_coast", "scene_011")
cfg = TrainConfig(iters=7000, refine_stop_iter=5040)   # tum: iters=3000
acc = None
for i in range(8):
    render_frames(f"member{i}/splats.pt", s.test_frames, f"member{i}/rgb", cfg)
    for fr in s.test_frames:
        a = np.asarray(Image.open(f"member{i}/rgb/{fr.stem}.png"), np.float32)
        acc = a if acc is None else acc + a
# submitted frame = mean over the eight members
Image.fromarray(np.clip(acc / 8, 0, 255).astype(np.uint8)).save("rgb/<stem>.png")
```

Averaging happens in image space, on the rendered frames, not in Gaussian
parameter space. Nothing about the held-out image is used to combine members, so
this is variance reduction rather than selection.

`per_member_renders/` in the released archive contains each member's frames, and
`averaged_renders/` contains the averaged result that was submitted.

**On bit-exactness.** gsplat's densification uses non-deterministic GPU atomics,
so re-running training from the same seed produces a statistically equivalent but
not bit-identical model. The released `splats.pt` files were regenerated from the
recorded seeds and configurations after the challenge closed, because the
original pipeline deleted per-member checkpoints once their frames had been
averaged (`render_ensemble.py` did this by design -- the checkpoints are far
larger than the renders). The renders they produce match the submitted frames to
within run-to-run variance, which on Gold Coast is a rendering-score standard
deviation of about 0.014 and on TUM about 0.0018.

Measured directly: averaging the eight regenerated members of `tum/scene_007`
and comparing against the frames actually submitted gives

    DJI_20241217100100_0036_D.png   PSNR 29.01 dB   mean|diff| 5.16/255
    DJI_20241217100321_0015_D.png   PSNR 29.41 dB   mean|diff| 5.18/255
    DJI_20241217100339_0024_D.png   PSNR 30.51 dB   mean|diff| 3.30/255

so the regenerated ensemble agrees with the submission to about 29.6 dB. The
residual is the densification nondeterminism described above, not a difference
in configuration.

## 2. Point clouds to submitted PLYs

**TUM (geometry term).** The fused MVS cloud voxel-downsampled to 2 cm. No
labels. The 2 cm figure is the scorer's own matching grid, not a tuned value.

**Gold Coast (semantic term).** The labelled cloud is thickened perpendicular to
its surface: every point emits copies at `0`, `-12 cm` and `+12 cm` along its
local surface normal, each carrying its parent's label, after the parents have
been coarsened to a 15 cm voxel.

Both are produced by the two commands below. They need
`{scene}_xl_v18labeled.npz` and `{scene}_xl_normals.npz` from the semantic
checkpoint archive.

```bash
# Gold Coast: coarsen parents to 15 cm, emit sheets at 0, -12, +12 cm
python scripts/build_sheet_clouds.py \
    --offsets 0.0 -0.12 0.12 --parent-voxel 0.15 --tag mid

# assemble the zip: sheeted Gold Coast + TUM at 2 cm + renders
python scripts/swap_gc_clouds.py \
    --tag mid --tum-voxel 0.02 \
    --src submissions/v18-final.zip \
    --out submissions/v20-mid-final.zip
```

`swap_gc_clouds.py` copies every entry of the source zip verbatim except the two
scored Gold Coast clouds and the five scored TUM clouds, which it rewrites. That
is why the rendering metrics of the final submission are identical to those of
the earlier `v18` submission: the PNG bytes are the same files.

**Important:** nothing may voxel-downsample the cloud *after* sheeting. The sheet
offsets (12 cm) are smaller than the parent voxel (15 cm), so a downsample
applied afterwards would merge each sheet back into its parent and silently
undo the operation. `build_submission(..., downsample=None)` is required, and the
PLY writer quantises to a 1 cm grid without deduplication, which leaves 12 cm
offsets intact.

## 3. Verifying the result

```bash
python -c "
import zipfile; z=zipfile.ZipFile('submissions/v20-mid-final.zip')
n=[x for x in z.namelist() if x.endswith('point_cloud.ply')]
print(len(n),'scenes; testzip', z.testzip())"
```

Expect 13 scenes and `testzip None`. Scored point counts are 142.6 M for
`gold_coast/scene_011`, 83.3 M for `gold_coast/scene_012`, and 7.4 / 11.7 / 10.0
/ 10.0 / 10.5 M for `tum/scene_004..008`. Development scenes are decimated to a
1 m voxel because the final phase does not score them.
