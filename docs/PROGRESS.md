# TwinWorld - Progress Log

Replaces Linear for this project (decided 2026-08-08).
Markers: `[done]` `[next]` `[blocked]`.

## 2026-08-08

- [done] Analysed Codabench competition 17661. Mirrored all pages, terms, phase and leaderboard
  metadata to `docs/competition/competition_17661.json`, plus the starting kit and
  `render_test_poses.py`.
- [done] Downloaded the full dataset (1.4 GB, 297 files) to `/data2/manish_joshi/twinworld/data`,
  symlinked as `data/`.
- [done] Characterised the data. Key finding: this is a **sparse-view** problem, 12 train views per
  TUM scene and 13-24 per Gold Coast scene, 3 test views each, 13 scenes total.
- [done] Measured the load-bearing facts, recorded in the design spec section 3.1:
  train-view frustums cover 97.2-100% of GT points on all six dev scenes (test frustums only
  87.2-100%); Gold Coast class balance is wall 56 / roof 21 / window 17 / ground 3 / other 2 percent;
  TUM test views interpolate (2.3 m to nearest train camera) while Gold Coast test views extrapolate
  (16-18 m).
- [done] Design spec written and approved:
  `docs/superpowers/specs/2026-08-08-twinworld-challenge-design.md`.
- [done] Decisions: compete to win; user registers on Codabench and uploads zips; tracking in-repo
  instead of Linear; GPUs 2 and 3 only; large artefacts under `/data2/manish_joshi`.
- [next] Phase 0: local scorer, dummy submission, package validator. Then Phase 1 geometry bake-off.

### Deadlines

- Development phase closes **20 Aug 2026** (3 submissions/day).
- Final phase **20 Aug - 01 Sep 2026** (1/day, 15 max).
- Workshop 08 Sep 2026.

## 2026-08-08 (evening, autonomous session)

- [done] Infrastructure complete: `dataio` (COLMAP parsing, scene registry, frustum tests),
  `scoring` (local reimplementation of the official metrics), `package` (submission builder plus
  validator that refuses to write a rejectable zip), `geometry/colmap_mvs`, `render/trainer`
  (gsplat), `semantics` (features + LightGBM classifier). 37 tests pass.
- [done] Toolchain: COLMAP 3.13 with CUDA into `/data2/manish_joshi/envs/colmap`; gsplat 1.5.3.
  The system `nvcc` is 11.5 and cannot parse this machine's gcc-11 headers, so `env.sh` pins
  `CUDA_HOME=/usr/local/cuda-12.3`, which compiles gsplat's kernels cleanly. **Source `env.sh`
  before any gsplat run.**
- [done] Dummy submission built and inspected: `submissions/dummy_official.zip`, 13 scenes,
  39 frames, 211 MB. Ready to upload as the scorer calibration point.
- [done] Fixed a real bug caught mid-run: two MVS configs shared one COLMAP workspace and were
  overwriting each other's depth maps. `run_mvs` now takes a `tag` that separates workspaces.

### Geometry bake-off, TUM scene_000

| Variant | Points | F@5 | F@10 | F@20 | mean F | P@10 | R@10 |
|---|---|---|---|---|---|---|---|
| provided `points3D.ply` | 27,949 | 0.0177 | 0.1278 | 0.4416 | **0.1957** | 0.751 | 0.070 |
| MVS, global depth range | 653,041 | 0.1835 | 0.5526 | 0.7195 | **0.4852** | 0.721 | 0.448 |
| MVS, per-image depth range | 651,909 | 0.1840 | 0.5534 | 0.7194 | **0.4856** | 0.721 | 0.449 |
| MVS, per-image + loose fusion | 1,501,782 | 0.1960 | 0.5353 | 0.6960 | **0.4758** | 0.615 | 0.474 |

- MVS is 2.5x the provided-sparse floor, so geometry rides on classical MVS rather than needing a
  learned-prior pipeline. That was the question Phase 1 existed to answer.
- Depth-range handling barely matters (0.4852 vs 0.4856), which is worth recording as a negative
  result: the obvious knob is not the bottleneck.
- Loose fusion buys recall (0.448 -> 0.474) but loses more precision (0.721 -> 0.615), so it is a
  net loss. **`auto` is the variant to carry forward.**
- **Recall is the bottleneck**: GT has 4.67 M voxels at 2 cm against our 0.42 M. 12 near-nadir views
  cannot observe the facades that the ground-truth scan contains.
- [next] Diagnose whether the missing recall is observable-but-filtered or genuinely unobserved,
  since that decides whether to push MVS density or accept the ceiling.

### Key insight: the semantic score is hard-capped by geometric coverage

Each ground-truth point takes the label of the nearest submitted point within 10 cm, and unmatched
points count as errors in their own class. So if a fraction `f_c` of class `c`'s ground-truth points
have any submitted point within 10 cm, then `IoU_c <= f_c`, and therefore

    semantic_score = mIoU <= mean_c (coverage of class c at 10 cm)

Our TUM recall at 10 cm is 0.33-0.47. If Gold Coast coverage is similar, the semantic score is
capped near 0.45 no matter how good the classifier is.

The consequence is that **TUM and Gold Coast want different point clouds**:

- **TUM** is scored on precision *and* recall (F-score), so its cloud is a balance.
- **Gold Coast** is never scored on geometric precision. Surplus points cost nothing there, so its
  cloud should be coverage-maximal: aggressive hole filling, generous trimming.

This is why Poisson densification is being evaluated separately per dataset rather than as one
global setting.

### Second bug: the far-field tail of COLMAP fusion

COLMAP's `stereo_fusion` leaves a small tail of very distant points. On TUM scene_000 that tail is
0.1 percent of the cloud but stretches the bounding box from 145 m to **9405 m**. Two consequences,
both measured rather than suspected:

- Poisson reconstruction grids the bounding box, so the first densification run produced 116 usable
  points instead of millions. The failure looked like a bad algorithm choice and was actually a
  bounding-box problem.
- The frustum test does not remove it. Those points sit inside the training frustums, just very far
  away, so `train_frustum_mask` kept all 651,909 points and the extent stayed at 9.4 km. At
  initialisation their k-nearest-neighbour scales are of the same order, which would have created
  Gaussians the size of the scene.

`robust_core()` keeps points within 3x the median distance from the camera rig. On scene_000 that
keeps 95.6 percent of points and brings the extent to 145 m, with 1.9x headroom over the furthest
ground-truth point. Per-axis percentile cropping was tried first and rejected: on every development
scene it cut inside the ground-truth bounding box.

Per-axis percentile crops were measured as unsafe, so distance from the camera rig is used instead.
Nothing is lost either way, because densification runs on the core and its output is unioned back
with the full raw cloud.

### Production run 2026-08-08-pipeline-v1

TUM uses the `loose` MVS config; Gold Coast uses `coverage`, which trades geometric precision (never
scored there) for the coverage that caps the semantic score.
Render uses 7000 iterations rather than the 3DGS default of 15000-30000: 12 training views make
7000 iterations already about 580 epochs.

Both the MVS and render stages now catch per-scene failures and continue, because this runs
unattended and a partial set of clouds still packages, with the provided sparse cloud as the
per-scene fallback.

### Third bug: densification ordering in the training loop

My training loop called `strategy.step_post_backward` before `optimizer.step()`. gsplat's own
trainer does the opposite, and the opposite is correct: `step_post_backward` grows and prunes
Gaussians, which rebuilds the parameter tensors and their Adam state, so stepping afterwards applies
gradients computed for the previous set of Gaussians to a different set.

Caught by reading gsplat's `simple_trainer.py` rather than by a failing test, because the wrong
order still trains and still lowers the loss. It was found before any production rendering had run,
so no results are affected.

### MVS bake-off, final: four TUM development scenes

| Variant | s000 | s001 | s002 | s003 | mean |
|---|---|---|---|---|---|
| provided sparse | 0.1957 | - | - | - | - |
| `auto` (per-image depth) | 0.4856 | 0.4103 | 0.5334 | 0.4692 | **0.4746** |
| `loose` (per-image depth + permissive fusion) | 0.4758 | 0.4833 | 0.4993 | 0.4462 | **0.4761** |

The two configurations are within noise of each other on the mean and swap rank per scene:
`auto` wins scene_002 on precision (P@10 0.82 vs 0.67), `loose` wins scene_001 on recall.
`loose` is carried forward for TUM because its recall advantage compounds with densification, and
because Gold Coast needs a coverage-biased configuration anyway.

### Autonomous run structure

The whole run is chained so nothing waits on interactive polling:

```
GPU 2:  MVS tum/scene_000-008   ->  render tum/scene_000-008
GPU 3:  MVS gold_coast/009-012  ->  render gold_coast/009-012
        (chain_render.sh waits on the MVS process, then renders what it built)

then:   semantics -> local score on dev scenes -> validated zip
        (chain_finish.sh waits on rendered frames appearing, with a 300 minute
         deadline after which it proceeds anyway, because a partial submission
         that exists beats a complete one that never got built)
```

`env.sh` had to be made safe under `set -u`: the chain scripts run with it, and a bare
`$LD_LIBRARY_PATH` on a machine that has never set it aborts the script before anything runs.

### Semantic features: added signed offset from the local plane

The wall/window boundary is the weakest part of the 5-class problem, and window is 16-18 percent of
Gold Coast points and a fifth of the unweighted mIoU. The original feature set separated the two
only by colour, which is fragile under changing illumination.

Added `planeoffset_k{16,48,128}`: the point's displacement along its local surface normal relative
to its neighbourhood centroid. At the larger neighbourhood sizes the fitted plane is the facade, so a
recessed window sits at a systematically negative offset while the wall around it sits near zero.
Planarity cannot express this, since a window pane and the wall around it are both planar.

Normals are oriented towards the mean training camera centre rather than by the up vector. Up-based
orientation is arbitrary on walls, where the normal is perpendicular to up, which would have made
the sign meaningless for exactly the class the feature was added for.

Verified on a synthetic facade with a 15 cm recessed patch: window mean -0.036, wall mean +0.002.
Feature count went from 26 to 29, and the feature cache now invalidates on the feature names rather
than only the point count, so a cache written before the change cannot feed the model wrong columns.

## Gold Coast: the semantic score is coverage-limited, and badly

Measured the first Gold Coast reconstruction (scene_009, `coverage` config, 8.0 M points) against
its ground truth. The result changes the picture for that dataset entirely.

**Per-class coverage at 10 cm, which upper-bounds per-class IoU:**

| Class | share of GT | coverage @10 cm | median distance to our nearest point |
|---|---|---|---|
| ground | 3.1% | 0.192 | 0.29 m |
| wall | 56.7% | **0.017** | 0.96 m |
| roof | 22.0% | 0.396 | 0.13 m |
| window | 16.5% | **0.005** | 0.86 m |
| other | 1.8% | 0.090 | 0.42 m |

**mIoU ceiling = 0.14.** No classifier can beat that on this reconstruction, so the semantic score
is currently a reconstruction problem, not a classification one.

### It is not a visibility limit

The obvious explanation would be that near-nadir drone views cannot see facades. Measured, that is
false: the incidence angle between each ground-truth surface normal and the best-placed training
camera has a median of **41 degrees for walls and 43 for windows**, with 100 percent of points below
70 degrees. These surfaces are geometrically well-placed for stereo.

Nor is it a registration error. A global misalignment would displace roofs too, and roofs sit at a
median 0.13 m with 89 percent within 50 cm. Only the vertical surfaces are displaced, by about a
metre. Our cloud does contain vertical surface (16.4 percent of in-bbox points against 27.1 percent
in the ground truth), so walls are being reconstructed, just in the wrong place. That is the
signature of the classic nadir-MVS facade artefact, where a facade is smeared along the ray from the
roof edge down to the ground rather than reconstructed as a vertical plane.

### A configuration mistake found in the process

The `coverage` config lowered `filter_min_triangulation_angle` to 0.5 degrees. A point triangulated
at half a degree has effectively unbounded depth uncertainty: 1 percent of the resulting cloud landed
beyond 3.5 km and the furthest point was 1.2e10 m from the camera rig. Reverted to 1.0. Loosening
the photometric and fusion filters buys coverage; loosening the triangulation angle only buys noise
at arbitrary depth. Packaging now also drops the far-field tail, which can never help either metric
since both are computed inside the ground-truth bounding box.

- [next] Compare `loose` against the corrected `coverage` on scene_009 before committing the other
  three Gold Coast scenes to either.

## Image resolution was the binding constraint on TUM geometry

Ground sampling distance is about 3.4 cm/px on TUM (36 m altitude, f = 1053 px) while the ground
truth is a 2 cm scan, so the hypothesis was that native resolution, not MVS tuning, caps the F-score.
Upsampling the images 2x before undistortion, with intrinsics scaled to match, confirms it:

| scene_000 | points | F@5 | F@10 | F@20 | mean F | P@10 | R@10 | runtime |
|---|---|---|---|---|---|---|---|---|
| `loose` (native) | 1.50 M | 0.196 | 0.535 | 0.696 | 0.4758 | 0.615 | 0.474 | 401 s |
| `loose_2x` | 5.85 M | **0.289** | **0.597** | 0.715 | **0.5336** | 0.652 | **0.551** | 1616 s |

F@5 improves by 47 percent and recall by 16 percent, for 4x the runtime. This is the largest single
improvement found so far, and it is worth more than every MVS filter setting combined: the whole
`auto` vs `loose` sweep moved the mean by 0.002, and resolution moves it by 0.058.

TUM production switched to `loose_2x`. The native-resolution clouds are kept as
`tum/<scene>/mvs_1x/fused.ply` so a complete submission remains buildable if the 2x pass does not
finish for every scene.

### Fourth bug: workspace collision between concurrent pipeline jobs

`run_mvs` clears its workspace before starting, and the workspace tag was `pipe{gpu}`. Two pipeline
jobs on the same GPU therefore deleted each other's depth maps mid-run, which surfaced as
`patch_match_stereo` exiting with SIGTERM 14 seconds in.

The proximate cause was that an earlier `kill` had hit the job's shell wrapper rather than the Python
process, so a job believed to be stopped was still running. The tag is now
`pipe{gpu}_{os.getpid()}`, which makes the collision impossible regardless of how process cleanup
goes. This is the same class of bug as the first one of the session, fixed properly this time:
a shared mutable workspace keyed on something that is not unique per writer.

## 2x reconstruction across all scenes

All thirteen scenes reconstructed at 2x image resolution:

| Scene set | config | points per scene | runtime |
|---|---|---|---|
| TUM scene_005-008 | `loose_2x` | 7.3 - 9.1 M | ~720 s |
| Gold Coast scene_009-012 | `coverage_2x` | 23.5 - 31.3 M | 700 - 1700 s |
| TUM scene_000-004 | `loose_2x` | in progress | |

Native-resolution clouds are preserved at `<scene>/mvs_1x/fused.ply` as a fallback.

### A size problem this creates

At 2 cm voxel resolution these clouds imply a submission of roughly 2 GB, most of it outside the
ground-truth bounding box and discarded by the scorer before anything is computed. The bounding box
is unknown for the final-phase scenes, so any crop has to be defined by camera geometry rather than
by the ground truth. `scripts/crop_analysis.py` measures the candidates on the development scenes,
where the cost in F-score can actually be checked:

- `train_frustum` - the current rule
- `test_frustum` and a margined variant - the held-out views are aimed at the region the ground truth
  was cropped around, and cover 87-100 percent of ground-truth points
- `gt_bbox_oracle` - not usable, included only to show how much is theoretically droppable

A rule is only adopted if it costs almost no F-score on **every** development scene, since on the
final scenes there is no way to check.

## Facade extrusion: modelling the walls instead of reconstructing them

The Gold Coast deficit is entirely in vertical surfaces. Roofs land within 13 cm of the ground truth
while walls are about a metre off, and walls plus windows are 73 percent of the points and 40 percent
of the unweighted mIoU. That asymmetry is exploitable: a building's walls are the vertical surfaces
directly beneath its roof outline, and the roof outline is the part we reconstruct well.

`src/twinworld/geometry/facade.py` rasterises the cloud into a horizontal height grid, marks cells
whose top sits above the local terrain as building, finds every boundary where a building cell meets
a cell at least 1.5 m lower, and emits a vertical sheet of points from the neighbour's top up to the
building's top. That covers both outer walls, where the neighbour is ground, and internal walls
between roof levels.

### Cell size is the parameter that decides whether this can score at all

The extruded wall is placed on a cell boundary while the true edge lies somewhere inside the cell
that straddles it, so the placement error is uniform over one cell. Against a 10 cm semantic match
radius that is fatal at coarse cells. Measured on a synthetic 20 x 20 x 12 m building deliberately
placed off-grid:

| cell | inward shift | median error | within 10 cm |
|---|---|---|---|
| 0.30 m | none | 0.230 m | 0.015 |
| 0.15 m | none | 0.080 m | 0.507 |
| 0.10 m | none | 0.070 m | 0.505 |
| 0.15 m | half a cell | **0.005 m** | **1.000** |
| 0.10 m | half a cell | **0.020 m** | **1.000** |

Shifting the emitted sheet half a cell back towards the building centres the error instead of
leaving it one-sided, and takes the synthetic case from half the points matching to all of them.

The first synthetic test was degenerate and nearly hid this: the building was axis-aligned on exact
grid boundaries, so the quantisation error was an artefact of the test rather than a property of the
method. Re-running with an off-grid building is what exposed the real behaviour.

- [next] Measure on the two labelled Gold Coast scenes. Only worth keeping if the ceiling actually
  moves; a model of where facades ought to be is not evidence that they are there.

### Facade extrusion: measured on Gold Coast scene_009

| Cloud | points | ceiling | ground | wall | roof | window | other |
|---|---|---|---|---|---|---|---|
| MVS only | 29.8 M | 0.1751 | 0.264 | 0.024 | 0.454 | 0.007 | 0.126 |
| + extrusion, 0.10 m cells | 70.1 M | **0.2000** | 0.278 | 0.027 | 0.518 | 0.010 | 0.168 |
| + extrusion, 0.15 m cells | 71.1 M | 0.1983 | 0.280 | 0.029 | 0.504 | 0.012 | 0.167 |
| + extrusion, 0.30 m cells | 69.8 M | 0.1902 | 0.275 | 0.030 | 0.472 | 0.014 | 0.162 |

**The idea works, but not for the reason it was built.** The ceiling improves by 0.025, and finer
cells are better as predicted. But the gain is almost entirely in roof (+0.064) and other (+0.042),
while wall moves 0.024 -> 0.027 and window 0.007 -> 0.010. Those two classes were the entire
motivation.

So the extruded sheets are not landing on the true facades. The most likely reason is that they are
extruded beneath *our* roof outline, and if our roof outline is itself smeared outward, or if the
real building has eaves that overhang the wall below, the sheet sits off the wall plane by more than
the 10 cm match radius. The synthetic test could not have caught this: it had a roof whose outline
was exactly the wall plane.

**A flaw in this measurement:** every configuration hit the 40 M point cap, so all four were
truncated, and the truncation drops whole boundary segments in iteration order rather than sampling
uniformly. The comparison between cell sizes is therefore only suggestive. Cost is also high:
doubling the cloud to 70 M points for +0.025 ceiling is a poor trade against upload size.

**Not shipping this in v1.** It is a real effect and worth the paper, but the honest reading is that
it mostly adds points near surfaces we already cover. The next question, if there is time, is where
ground-truth wall points actually sit relative to our roof outline, which is what would say whether
the offset is an overhang, an outline error, or something else.

## Submission resolution is a per-dataset decision, and it was wrong

Both datasets were being submitted at a 2 cm voxel. That is right for TUM, where the F-score is
computed after a 2 cm downsample, and wrong for Gold Coast, where the only question ever asked is
whether a ground-truth point has *some* submitted point within 10 cm. Coverage is a property of
spacing, not density, so beyond a certain spacing extra points buy literally nothing.

Measured on scene_009, sweeping the voxel and reading off the semantic ceiling:

| voxel | points | approx size | ceiling |
|---|---|---|---|
| 2 cm | 29.8 M | 387 MB | 0.1751 |
| 4 cm | 24.6 M | 320 MB | 0.1737 |
| **6 cm** | **18.5 M** | **241 MB** | **0.1710** |
| 8 cm | 14.2 M | 185 MB | 0.1668 |
| 12 cm | 9.3 M | 121 MB | 0.1552 |

6 cm removes 38 percent of the points for 0.004 of ceiling, so `SUBMISSION_VOXEL` is now 2 cm for
TUM and 6 cm for Gold Coast. The same reasoning applies to the extruded facades, which were being
emitted every 4 cm: at 0.04 m spacing one scene produced 121 M facade points, roughly six times finer
than a 10 cm match radius can distinguish.

This is the third time the same mistake has appeared in a different guise: doing work at a resolution
the metric cannot see. The others were computing semantic features on the raw 30 M-point fusion
output rather than the submitted cloud, and submitting points outside the ground-truth bounding box
that the scorer discards before it computes anything.

### Crop rules, measured on TUM

| rule | scene_000 kept / delta F | scene_001 kept / delta F |
|---|---|---|
| `train_frustum` | 100.0% / +0.0000 | 100.0% / +0.0000 |
| `test_frustum` | 92.7% / +0.0001 | 84.0% / **-0.0203** |
| `test_margin200` | 98.2% / +0.0000 | 96.1% / +0.0000 |
| `gt_bbox_oracle` | 46.7% / +0.0000 | 77.0% / +0.0000 |

`test_frustum` is not safe: it costs 0.02 F-score on scene_001, which is exactly the kind of
per-scene failure that would be invisible on the final scenes. `test_margin200` is free but saves
little. TUM scenes are 66-90 MB each and are not the size problem, so no crop is adopted for them.

### Facade density: the earlier comparison was truncated

The first facade sweep capped output at 40 M points, and every configuration hit the cap, so all
four were truncated by dropping whole boundary segments in iteration order. Re-running scene_009
uncapped:

| cloud | points | size | ceiling |
|---|---|---|---|
| MVS at 2 cm | 29.8 M | 387 MB | 0.1751 |
| MVS + facades at 4 cm spacing | 150.6 M | 1958 MB | **0.2167** |

So facades are worth more than the capped run suggested (+0.042 rather than +0.025), and the cap was
the thing hiding it. 4 cm spacing is still six times finer than a 10 cm match radius needs, which is
why the union is 2 GB for one scene.

**A ceiling is not a score.** The extruded points are synthetic and have no colour of their own,
while about half the classifier's features are colour, and a wrongly-labelled point that sits closer
to a ground-truth point than a correctly-labelled one actively steals the match. So facades are only
adopted if leave-one-scene-out *achieved* mIoU improves, not the ceiling.
`scripts/gc_v2_semantics.py` measures exactly that, giving extruded points the colour of their
nearest reconstructed point.

## Rendering: where the score actually goes, and one thing not worth building

First production rendering numbers, TUM scene_000: PSNR 16.90, SSIM 0.698, LPIPS 0.276, giving
`rendering_score` 0.587. The three terms are 0.338, 0.698, 0.724, so **PSNR is by far the weakest**,
which is a consequence of the metric dividing by 50: PSNR 25 would still only score 0.5.

Correcting a single global gain on the first test frame lifted its PSNR from 16.38 to 18.23, which
looked like a large, cheap win from per-image appearance modelling. It is not, and the check that
showed this is worth recording:

| scene | train mean | test mean | render mean | render/test |
|---|---|---|---|---|
| tum/scene_000 | 0.4127 | 0.4082 | 0.4440 | 1.088 |
| tum/scene_001 | 0.4351 | 0.4401 | 0.4391 | 0.998 |
| tum/scene_002 | 0.4207 | 0.4226 | 0.4269 | 1.010 |
| tum/scene_003 | 0.4145 | 0.4141 | 0.4138 | 0.999 |
| gold_coast/scene_009 | 0.3436 | 0.3583 | 0.3252 | 0.908 |
| gold_coast/scene_010 | 0.3560 | 0.3639 | 0.3577 | 0.983 |

The renders already sit at the training set's average exposure, and that average already matches the
test set's. The apparent 1.85 dB was **per-frame variance**, not a bias: the individual ratios run
from 0.84 to 1.20 in both directions within a single scene. Appearance modelling factors out
per-image exposure during training and then renders at the canonical average, which is exactly where
we already are, so it would move nothing. Not built.

Gold Coast is a different problem: a global gain there changes PSNR by 0.01 dB, so its error is
structural rather than photometric, consistent with test views that extrapolate 16-18 m from the
nearest training camera.

### Initialisation, measured

| variant | scene | PSNR | SSIM | LPIPS | rendering_score |
|---|---|---|---|---|---|
| sparse `points3D` init | tum/scene_000 | 16.55 | 0.645 | 0.347 | 0.5432 |
| dense MVS init | tum/scene_000 | 17.81 | 0.700 | 0.273 | **0.5946** |

Dense initialisation is worth +0.051 of `rendering_score` on its own, which is the sparse-view claim
the design predicted and the largest single rendering effect measured so far.

## Rendering configuration is per-dataset in two ways, not one

Held-out-photo scores on the development scenes:

| | TUM | Gold Coast |
|---|---|---|
| initialisation | dense MVS (0.5946 vs 0.5432 sparse) | **sparse** (0.5427 vs 0.4413 dense) |
| iterations | **3000** (0.6099 vs 0.5946 at 7000) | **7000** (0.5427 vs 0.4850 at 3000) |

Both axes flip between the datasets, and each flip has a reason:

- **Initialisation.** The Gold Coast clouds are the coverage-maximal ones: 23-31 M points,
  deliberately noisy, spread over hundreds of metres. That is the right cloud to submit and the wrong
  cloud to seed Gaussians from. Submitted geometry and rendering initialisation are now decoupled.
- **Iterations.** TUM has 12 views whose held-out frames interpolate 2.3 m from a training camera, so
  it overfits quickly. Gold Coast has 13-24 views whose held-out frames extrapolate 16-18 m, so it
  needs longer to resolve geometry it barely observes.

Assuming one setting for both datasets, which is what the first production run did, was wrong on
every axis for one dataset or the other.

Pooled over the six development scenes:

| run | PSNR | SSIM | LPIPS | rendering_score |
|---|---|---|---|---|
| v1 (MVS init, 7000 everywhere) | 18.04 | 0.6876 | 0.3216 | 0.5756 |
| v2 (per-dataset init, 3000 everywhere) | 19.66 | 0.7281 | 0.3512 | 0.5900 |

Note that v2 gained 1.6 dB of PSNR and 0.04 of SSIM while **losing** 0.03 of LPIPS: the sparse
initialisation produces a blurrier reconstruction that pixel metrics like and the perceptual metric
does not. Since the challenge weights all three equally, that is still a net gain, but it is the kind
of trade that a single-metric evaluation would have hidden.

## Ensembling independent runs, and why it works here

Repeating an identical configuration three times, same seed, same everything:

| scene | PSNR range | rendering_score sd | 2 sd band |
|---|---|---|---|
| tum/scene_000 | 19.10 - 19.28 | 0.0018 | 0.004 |
| gold_coast/scene_009 | 18.06 - 19.76 | 0.0141 | 0.028 |

Gold Coast is about eight times noisier. Densification is nondeterministic, and its held-out views
extrapolate 16-18 m from the nearest training camera, so a small change in the reconstruction moves
the rendered image a lot. This also explains a 2 dB discrepancy chased earlier between two runs whose
configs differed only in `refine_stop_iter` 5000 against 5040: that was noise, not the setting.

**Read against this noise floor, the rendering conclusions hold, with one weaker than the others:**

| conclusion | effect | noise band | verdict |
|---|---|---|---|
| TUM: dense over sparse init | +0.051 | 0.004 | solid |
| TUM: 3000 over 7000 iterations | +0.015 | 0.004 | solid |
| Gold Coast: sparse over dense init | +0.101 | 0.028 | solid |
| Gold Coast: 7000 over 3000 iterations | +0.058 | 0.028 | real, but only 2x the noise |

**Averaging independent runs turns that noise into score.** Each run is an unbiased but noisy
estimate of the same scene, so the mean is a better estimate than any single run, and nothing about
the held-out image is used to combine them.

| run | PSNR | SSIM | LPIPS | rendering_score |
|---|---|---|---|---|
| v1 (one run, MVS init, 7000 both datasets) | 18.04 | 0.6876 | 0.3216 | 0.5756 |
| v2 (one run, per-dataset init and iterations) | 19.51 | 0.7299 | 0.3236 | 0.5988 |
| **v3 (4-member ensemble)** | **20.77** | **0.7640** | **0.2973** | **0.6274** |

v3 improves on every development scene, and unusually it improves LPIPS as well as the pixel
metrics. Averaging normally blurs and costs perceptual quality; here the floaters each run invents
are independent and cancel, while the geometry the runs agree on survives. The total gain from v1 is
+0.052 of `rendering_score`, which is larger than any single configuration change measured.

### Ensemble size

| members | pooled dev rendering_score |
|---|---|
| 1 (v2) | 0.5988 |
| 4 (v3) | 0.6274 |
| 8 (v4) | 0.6309 |

Diminishing as expected, since the variance of a mean falls as 1/N: the first three members buy
0.029 and the next four buy 0.0035. Eight is where this stops being worth the GPU time.

### A pathology worth naming: LightGBM threading under load

The semantics stage appeared to hang three separate times, once for over three hours on a single
fold. It was not the data size and not the model size. On this machine, with a load average around
270 while the GPUs are busy, LightGBM's OpenMP barriers spin-wait and contend:

| n_jobs | fit, 20k points, 50 estimators |
|---|---|
| 1 | 1.1 s |
| 4 | 3.3 s |

More threads made it strictly slower, and `n_jobs=-1` on 96 cores made it effectively
non-terminating; the process was holding 151 threads. The stage is now single-threaded.

Two other things were wrong in the same place and were fixed with it. The model was specified with
600 estimators over 5 classes, which is 3000 trees, while only about 70 000 points per scene fall
within the label-transfer radius of any ground truth: it was far larger than the signal supports, and
predicting it over millions of points was most of the cost. And features were being computed on the
full submitted cloud rather than a thinned copy, which cannot change any label at a 10 cm match
radius.

The general lesson is the same one that appeared three times elsewhere in this session: **work done
at a resolution, or a capacity, that the metric cannot see is pure cost.**

## Semantics: achieved score, and how much of the ceiling it reaches

Leave-one-scene-out over the two labelled Gold Coast scenes, classifier trained on our own
reconstruction with labels transferred from the ground truth:

| held out | mIoU | ceiling | ground | wall | roof | window | other |
|---|---|---|---|---|---|---|---|
| scene_009 | 0.0886 | 0.1709 | 0.039 | 0.014 | 0.387 | 0.002 | 0.002 |
| scene_010 | 0.1282 | 0.2155 | 0.146 | 0.027 | 0.459 | 0.005 | 0.004 |
| **mean** | **0.1084** | 0.1932 | | | | | |

With only two labelled scenes this is a two-fold estimate and the spread between folds is 0.040,
which is a third of the mean. The number should be read as "about 0.11, plus or minus a lot".

**Reading it against the per-class coverage tells the real story.** Dividing achieved IoU by the
coverage that bounds it:

| class | fraction of its ceiling reached |
|---|---|
| roof | 87% and 90% |
| wall | 61% and 63% |
| window | 29% and 13% |
| ground | 15% and 57% |
| other | 2% |

Roof is nearly saturated: its remaining error is coverage, not classification. Window and other are
where classification itself fails, and the reason is visible in the training data. Only about 26 000
of 1.5 million classified points per scene fall within 15 cm of any ground-truth point, and those are
roof-dominated: scene_009 contributes 351 window examples and 544 "other" examples. The classifier
cannot learn classes it has a few hundred examples of.

So the semantic score is bounded twice over: by coverage at roughly 0.19, and by having almost no
supervision for three of the five classes. Both come from the same root cause, that the
reconstruction only lands near the ground truth on roofs.

## Final local score, and one number in it that is not honest

`experiments/2026-08-08-pipeline-v4/local_scores.json`:

```
LOCAL (dev):  rendering 0.6309 + geometry 0.5462 + semantic 0.1898 = 1.3669 / 3.0
```

**The semantic term of 0.1898 is optimistically biased and must not be quoted as an expectation.**
The model that produced it was fitted on scene_009 and scene_010 and then evaluated on those same two
scenes, because they are the only labelled Gold Coast scenes in existence. It is a training-set
score. The tell is that it lands at 0.1898 against a coverage ceiling of 0.1962, which is 97 percent
of the maximum: the classifier has memorised the two scenes it was shown.

The unbiased estimate is the leave-one-scene-out number, **0.1084**, where each fold is evaluated on
a scene the model never saw. The gap between 0.1084 and 0.1898 is the size of the optimism.

| component | honest estimate for the final phase | training-set reading |
|---|---|---|
| rendering | 0.631 | 0.631 |
| geometry (TUM) | 0.546 | 0.546 |
| semantics (Gold Coast) | **0.108** | 0.190 |
| **total** | **~1.29** | 1.37 |

Rendering and geometry are honest: both are measured against held-out photos and ground-truth clouds
that no part of the pipeline was fitted to. Only the semantic term has this problem, and it has it
because the challenge provides exactly two labelled scenes.

### Per-scene geometry, 2x MVS

| scene | F@5 | F@10 | F@20 | mean F |
|---|---|---|---|---|
| tum/scene_000 | 0.2887 | 0.5972 | 0.7149 | 0.5336 |
| tum/scene_001 | 0.3431 | 0.5618 | 0.6853 | 0.5301 |
| tum/scene_002 | 0.4861 | 0.6365 | 0.6869 | 0.6032 |
| tum/scene_003 | 0.3543 | 0.5528 | 0.6468 | 0.5179 |
| **mean** | | | | **0.5462** |

Against 0.196 for submitting the provided sparse cloud unchanged, and 0.476 before the 2x resolution
change.

## Decomposing the semantic score: both halves are broken, and coverage is the harder one

Three measurements, each isolating one failure:

| what is measured | mIoU | what it holds fixed |
|---|---|---|
| **achieved**, leave-one-scene-out on our reconstruction | **0.108** | nothing; this is the real number |
| **coverage ceiling**, perfect labels on our reconstruction | **0.196** | classification made perfect |
| **classification ceiling**, our classifier on ground-truth geometry | **0.346** | geometry made perfect |

Neither term alone explains the result. Fixing classification perfectly would reach 0.196; fixing
geometry perfectly would reach 0.346. The achieved 0.108 is 55 percent of the coverage cap, so there
is real headroom in classification too, but **coverage is the binding constraint**, and it is the
lower of the two ceilings.

Per-class, on ground-truth geometry (so this is pure classification quality, two folds):

| class | IoU |
|---|---|
| roof | 0.671, 0.506 |
| wall | 0.384, 0.573 |
| window | 0.307, 0.315 |
| ground | 0.176, 0.364 |
| other | 0.081, 0.089 |

Even with perfect geometry the classifier only reaches 0.31 on window and 0.09 on "other". So the
window problem is not purely a coverage problem: the feature set genuinely struggles to separate a
recessed glazed panel from the wall around it, and "other" is a catch-all with no coherent
signature. That is worth knowing before investing further in facade reconstruction, which would lift
coverage but leave these two classes weak.

These runs took 4-8 hours each, entirely because of the LightGBM threading pathology described above.
Re-running them with the single-threaded configuration would take minutes.

## Resolution: where the sensor runs out

| resolution | points | F@5 | F@10 | F@20 | mean F | recall@10 | runtime |
|---|---|---|---|---|---|---|---|
| native | 1.5 M | 0.196 | 0.535 | 0.696 | 0.4758 | 0.474 | 401 s |
| 2x | 5.9 M | 0.289 | 0.597 | 0.715 | 0.5336 | 0.551 | 830 s |
| 4x | 21.9 M | 0.328 | 0.616 | 0.718 | **0.5539** | 0.560 | 3673 s |

Native to 2x buys 0.058; 2x to 4x buys 0.020 for four times the cost again. That is the shape
expected once upsampling passes the information the sensor actually captured: the images are 1600 px
wide, so 2x already puts effective sampling below the 2 cm ground truth and 4x is interpolating.
Recall is nearly flat between 2x and 4x (0.551 to 0.560) while F@5 still moves (0.289 to 0.328), so
what 4x buys is precision on surfaces already found, not new coverage.

## Render ablation, complete

All on TUM scene_000 and Gold Coast scene_009, against noise bands of 0.004 and 0.028 respectively:

| variant | TUM | Gold Coast |
|---|---|---|
| 3000 iterations, dense init (chosen for TUM) | **0.6099** | 0.4405 |
| 7000 iterations, sparse init (chosen for Gold Coast) | 0.5432 | **0.5427** |
| MCMC densification | 0.6093 | 0.4648 |
| scale and opacity regularisation | 0.6016 | 0.4760 |
| random background | 0.5971 | 0.4391 |
| 15000 iterations | 0.5874 | 0.4155 |

Nothing beats the per-dataset choices already made. MCMC ties on TUM within noise and loses on Gold
Coast. Regularisation and random-background augmentation, both standard sparse-view remedies, are
neutral to harmful here. 15000 iterations is the worst variant on both datasets, which is the
overfitting story again: more optimisation against twelve views makes the held-out views worse.

# 2026-08-10

## The radiance field holds the geometry stereo threw away

PatchMatch stereo needs photometric correspondence. On Gold Coast it finds it on roofs and largely
fails on facades: wall coverage at 10 cm is 0.023 against 0.443 for roofs. But those facades are not
unobserved, their median incidence angle to the best-placed training camera is 41 degrees, and the
splatting model renders the training views at 25-35 dB. So the model must contain that geometry even
though correspondence search could not find it.

Rendering the trained model's expected depth at the training poses and unprojecting it recovers
exactly that. Measured on scene_009:

| cloud | ceiling | ground | wall | roof | window | other |
|---|---|---|---|---|---|---|
| MVS only | 0.1709 | 0.259 | 0.023 | 0.443 | 0.007 | 0.123 |
| GS depth only | 0.2268 | 0.118 | 0.107 | 0.490 | 0.117 | 0.302 |
| **MVS + GS depth** | **0.3202** | 0.332 | **0.125** | 0.653 | **0.121** | 0.370 |

The semantic ceiling nearly doubles, and the gains land precisely on the classes stereo failed:
**wall 5.4x and window 17x**. The two sources are complementary rather than redundant, which is why
the union beats either alone by a wide margin.

### Gold Coast only, and why

The same union costs F-score on TUM: 0.5336 to 0.5193. It raises recall there too (0.551 to 0.618)
but precision falls further (0.652 to 0.537), and TUM is scored on an F-score while Gold Coast is
scored purely on whether ground truth has *some* submitted point within 10 cm. The right cloud is
different for the two datasets because the two metrics ask different questions. This is the fourth
axis on which the datasets want opposite treatment.

### The multi-view agreement filter had to be dropped, and that is worth stating plainly

Requiring two training views to independently place a surface in the same spot, which is the
consistency check stereo applies during matching, removes most of the benefit: ceiling 0.320 falls
to 0.223, and requiring three views takes it to 0.199. The surfaces at issue are exactly the ones
few views resolve confidently, so the filter removes the geometry it is meant to validate.

Running without it means these points are noisier than stereo output and some are certainly wrong.
They are still a real estimate from a model fitted to real photographs, not invented surface, and
their unreliability is precisely why they are not used on TUM, where wrong points are penalised. The
honest framing for the paper is that Gold Coast's metric has no precision term, so a
recall-maximising reconstruction is the correct response to it, and that this is a property of the
metric worth criticising rather than a property of the scene.

## v5: stereo fused with splatting depth

| run | rendering | geometry | semantic | total |
|---|---|---|---|---|
| v4 (stereo only) | 0.6309 | 0.5462 | 0.1898 | 1.3669 |
| **v5 (stereo + GS depth on Gold Coast)** | 0.6309 | 0.5462 | **0.2791** | **1.4562** |

The semantic term rises by 0.089 with no change to rendering or geometry, and the achieved score is
now 83 percent of its ceiling (0.2791 of 0.3352) against 55 percent before. Held-out
leave-one-scene-out also improves, 0.1084 to 0.1480, so this is not only the training-set effect.

### Gap-filling on TUM: a clean negative

The hypothesis was that GS depth hurts TUM only where it duplicates surfaces stereo already has, so
keeping only points far from any stereo point should add recall without costing precision. Measured
on scene_000, every variant is worse than stereo alone, and worse than the full union:

| cloud | F | P@10 | R@10 |
|---|---|---|---|
| stereo only | 0.5336 | 0.652 | 0.551 |
| + all GS depth | 0.4633 | 0.411 | 0.672 |
| + GS depth >0.1 m from stereo | 0.4426 | 0.395 | 0.657 |
| + GS depth >0.2 m from stereo | 0.4304 | 0.395 | 0.600 |
| + GS depth >0.4 m from stereo | 0.4335 | 0.417 | 0.569 |

The restriction makes it worse, not better, so the GS points that were helping were the ones *near*
existing stereo surface, and the ones far from it are largely wrong. GS depth is Gold Coast only, and
the question is settled rather than left open.

### Interpolated viewpoints: a large gain that needs a harder test

Rendering depth from viewpoints interpolated between adjacent training cameras raises the Gold Coast
ceiling from 0.3025 to 0.4333 on scene_009, doubling wall coverage and more than doubling window.

That gain is larger than anything else measured, which is reason for suspicion rather than
celebration. A coverage metric with no precision term rewards simply having more points near
surfaces, so a ceiling increase alone cannot distinguish genuine reconstruction from dilution. The
discriminating test is whether **achieved** mIoU tracks it, since that additionally requires the
labels on those points to be correct. That test is running.

# 2026-08-11

## Official confirmation: v5-safe scored 1.325, first place

| # | owner | final | psnr | ssim | lpips | fscore | miou |
|---|---|---|---|---|---|---|---|
| 1 | manish | **1.325** | 20.82 | 0.770 | 0.294 | 0.521 | 0.174 |
| 2 | yruichen | 1.100 | 18.35 | 0.614 | 0.410 | 0.452 | 0.125 |

Local prediction was 1.33 with mIoU 0.173; official is 1.325 with 0.174. The scorer calibration
holds exactly, and the splatting-depth fusion is now externally verified: same renders, same
resolution as the failed-then-mini submissions, only the clouds changed, and mIoU went 0.110 to
0.174. Also learned: 864 MB is accepted, so the size failure threshold lies between 864 MB and
1.51 GB.

## v6 (interpolated views) passes the honest test

Held-out leave-one-scene-out on the interpolated-view clouds, paired by fold:

| fold | v5 fused | v6 interpolated |
|---|---|---|
| held out scene_009 | 0.1236 | **0.1449** |
| held out scene_010 | 0.1725 | **0.1985** |
| mean | 0.1480 | **0.1717** |

Both folds improve in the same direction by similar amounts, which is stronger evidence than the
mean against a fold spread of 0.05. The ceiling gain was not dilution; the extra viewpoints add
genuinely labellable surface.

## The binding constraint has flipped

The paper claim "coverage caps the semantic score" was true for stereo-only clouds (achieved 0.108
of ceiling 0.196). With v6 clouds the ceiling is 0.43-0.48 and achieved is 0.17, or **37 percent of
ceiling: classification is now the bottleneck**, specifically window (IoU 0.045-0.050 against
coverage 0.25-0.27) and other. This is why the next investment is a pretrained 2D segmenter lifted
to 3D rather than more geometry.

## Overfitting audit, per the user's directive

Two checks, both recorded:

1. **The interpolate/extrapolate mechanism transfers to the final scenes.** Measured on the
   published final-scene poses: TUM final test views sit 0.9-9.3 m from their nearest training
   camera (interpolation, matching dev's 1.8-11.8), Gold Coast final 16.0-17.1 m (extrapolation,
   matching dev's 17.7). The per-dataset rendering configs rest on this mechanism, not on
   leaderboard feedback. Caveat: tum/scene_006 has one 19 m test view, so TUM is not uniformly
   interpolative.
2. **Per-scene configuration selection on final scenes is impossible** (no GT, no test photos), so
   every choice is per-dataset and mechanism-backed. The leaderboard has only ever been used to
   confirm decisions already made on held-out dev evidence, never to choose between configurations.

The remaining known dev-fit risks, stated plainly: the semantic classifier is trained on two scenes
(mitigated by paired-fold LOSO validation and next by pretrained 2D backbones); the GC iteration
count (7000 vs 3000) cleared its noise band by only 2x.

## seg2d: an honest negative under the pre-registered rule

The 2D-segmentation lift (DeepLabV3 fine-tuned on GT projected into training views, lifted by
occlusion-gated multi-view voting) was adopted only if held-out LOSO beat 0.1717 with **both folds
improving**. Result: fold 009 improved 0.1449 -> 0.1843, fold 010 regressed 0.1985 -> 0.1639, mean
0.1741. Split folds, rule says keep LightGBM. Likely cause: 40 epochs on ~20 images of a single
scene overfits that scene's appearance; the backbone's priors do not compensate for a train set of
one building style. Recorded because the discipline matters as much as the result: the rule was set
before the run, and it held.

## Coordinate quantisation: 26 percent smaller submissions for free

Snapping submitted coordinates to a per-dataset grid (5 mm TUM, 1 cm Gold Coast) before writing
makes the float32 mantissas compressible: zip ratio 0.92 -> 0.70, v6-safe 1.23 GB -> 0.91 GB.
Worst-case displacement is half the grid diagonal, 0.43/0.87 cm against 5 cm and 10 cm metric
scales. Verified from the written zip: coordinates on-grid, pooled dev semantic 0.2090 (vs 0.2091
unquantised - the difference is one point in the fourth decimal).

## Submission policy change (user decision, 2026-08-11)

No further development-phase uploads. Rationale: the local scorer is calibrated exactly (two
consecutive predictions matched the official score), the development ranking has no official
weight, and each upload broadcasts our achievable score to competitors. All improvement is
validated locally (LOSO with pre-registered adoption rules); the final phase gets the polished
version. Open operational risk to close before 20 Aug: the scorer's size limit is only bracketed
(864 MB accepted, 1.51 GB failed with an empty log). Options: one dress-rehearsal dev upload of
the final-form zip, or cap the final submission at 864 MB.

## 16-member Gold Coast ensembles: adopted

Paired comparison (first 8 members shared): scene_009 0.5575 -> 0.5621, scene_010 0.5857 -> 0.5898.
Both scenes agree, matching the 1/N variance prediction. Adopted for Gold Coast; TUM stays at 8,
where run variance is 8x smaller and the marginal member buys nothing measurable.

A process slip worth recording: the v6 run directory's `gs/rgb` was scaffolded as a symlink into
v4, so the 16-member averaging wrote *through* the link and overwrote v4's Gold Coast averages.
First comparison came out byte-identical, which is what exposed it. The 8-member results were
recomputed from the stored member renders (all members are kept precisely so ensembles can be
reconstructed), and run directories now get real directories, not symlinks, for anything a stage
writes into. Historical zips are unaffected.

## TSDF fusion of ensemble depth: killed early on a decisive negative

The hypothesis was that a TSDF volume would do for geometry what frame averaging did for rendering:
cancel independent member errors. The single-member control already refutes the premise on TUM
scene_000: the extracted surface scores F 0.183 with precision 0.117 *inside the GT bounding box*,
against 0.534 for MVS, and notably worse than the raw unprojected depth it was built from (P@10
0.230). TSDF interpolation manufactures phantom surface in the truncation band around noisy depth,
so integrating depth this way adds geometry rather than cancelling noise. Eight-member integration
was not worth hours of contended CPU to confirm against a 5x precision deficit; killed and recorded.
Geometry is closed unless a qualitatively different idea appears. The remaining headroom is in
semantics, which now gets the machine undivided.

## Dense-train sampling: rejected, and the failure mode matters more than the number

Thirty times more labelled training points (26k -> ~800k per scene) made the classifier
dramatically *worse*: LOSO 0.1717 -> 0.0863, with fold 009 collapsing to 0.0400. Both folds
regressed; the pre-registered gate rejected it instantly.

The mechanism is the project's oldest lesson wearing a new coat. The training features were
computed on a label-dense subset (everything within 15 cm of ground truth, plus a 500k background
sample), while inference features are computed on a uniformly thinned cloud. The neighbourhood
features - k-nearest eigenfeatures, local extent, density - are sample-density-dependent, so the
training and inference feature distributions diverged, and the model learned statistics of a
sampling pattern rather than of surfaces. "Train on what you submit" was the founding principle of
this semantics pipeline, and this experiment violated it in the name of more data.

A supervision increase remains desirable, but it has to come from labelling more of the *uniform*
cloud (larger uniform feature budget), not from resampling around the labels. Running total:
three adoptions, four rejections, every rejection caught by a gate set before the run.

## Research survey, 2026-08-11

A five-agent SOTA survey covering low-label 3D semantics, sparse-view extrapolated NVS, dense
geometry, and competition intelligence is written up at `docs/research/2026-08-11-sota-survey.md`,
ending in a proposed five-day plan ordered by expected impact per day.

## Research-driven cycle: first two verdicts

**uniform-4M supervision: ADOPTED.** Raising the shared uniform feature budget from 1.5M to 4M
points (labelled subset ~26k -> 58-70k, identical sampling on both sides) lifts LOSO from 0.1717
to **0.1846**, both folds improving (0.1449 -> 0.1567, 0.1985 -> 0.2125). This is what dense-train
was trying to buy, obtained without the distribution shift that sank it.

**APD-MVS: ADOPTED for TUM, pending full-scene confirmation.** The survey's top geometry
recommendation (CVPR23, adaptive patch deformation, MIT) transfers from ETH3D to our aerial data:

| scene | COLMAP loose_2x | APD-MVS | P@10 | R@10 |
|---|---|---|---|---|
| tum/scene_000 | 0.5336 | **0.5774** | 0.652 -> 0.781 | 0.551 -> 0.564 |
| tum/scene_002 | 0.6032 | **0.6406** | 0.760 -> 0.875 | 0.547 -> 0.541 |

Precision and recall both rise; mean +0.04 F on the two scenes tested. Geometry, which had been
declared closed after four failed internal ideas, was reopened by external evidence - the survey
paid for itself with this one row. Build cost: four small fights (sm_80, shared Boost, conda
libstdc++ rpath, numpy 2.x asscalar, and a converter that deletes its own input images when
dense_folder == save_folder), all recorded in scripts/apdmvs_run.py.

The survey's own leverage table (docs/research/2026-08-11-sota-survey.md section 0) now orders all
remaining work: other +0.108, roof +0.059, ground +0.054, window +0.042, wall +0.023, with the
decision-rule sweep in flight against the rare-class operating-point diagnosis.

## Notion archive created (2026-08-11)

Durable off-VM archive at Notion project "TwinWorld Challenge @ ECCV 2026": Journal (full history
mirror), Design spec, Submissions & scores, Rules & contamination blocklist, Research library,
Handoff. Major journal updates get mirrored there from now on.

## Decision-rule sweep: rejected cleanly

| weighting + rule | LOSO |
|---|---|
| none + argmax | 0.1764 |
| none + iou_bias | 0.1800 |
| sqrt + argmax | 0.1833 |
| sqrt + iou_bias | 0.1776 |
| **balanced + argmax (incumbent)** | **0.1846** |
| balanced + iou_bias | 0.1702 |

The survey's operating-point hypothesis did not survive the gate: an IoU-optimal bias tuned on one
building does not transfer to another (helps `none`, hurts `sqrt` and `balanced`), and the existing
`balanced+argmax` is the best of all six cells. With one training scene per fold, post-hoc rule
tuning is one more thing to overfit. Kept as-is; the negative and its mechanism go in the paper.

## Final-window verdicts

**Roof-plane feature: REJECTED, overriding the script's own "ADOPT" banner.** The check script
compared against a stale baseline (0.1717); against the true incumbent (0.1846) both folds are
worse (0.1535/0.2030 vs 0.1567/0.2125) and the "other" IoU the feature was designed for *halved*
(0.040->0.018, 0.060->0.013). Perfect synthetic separation (plant +1.5, roof 0, wall -6, ground
-12) did not survive contact with real smeared clouds - the roof-level estimate is too noisy there.
Reverted to 29 features; implementation retained in features.py as a documented negative.

**APD union loose_2x: ADOPTED for TUM.** The two stereo methods are complementary:

| scene | loose_2x | APD | union |
|---|---|---|---|
| tum/000 | 0.5336 | 0.5774 | **0.5905** |
| tum/001 | 0.5301 | 0.5281 | **0.5732** |

(002/003 computing.) v7 TUM geometry = per-scene union at 2 cm.

## Union complete: adopted on 4/4 scenes

| scene | loose_2x | APD | union |
|---|---|---|---|
| tum/000 | 0.5336 | 0.5774 | **0.5905** |
| tum/001 | 0.5301 | 0.5281 | **0.5732** |
| tum/002 | 0.6032 | 0.6406 | **0.6502** |
| tum/003 | 0.5179 | 0.5452 | **0.5607** |
| mean | 0.5462 | 0.5728 | **0.5937** |

The union beats both components on every scene: the two PatchMatch variants make partially
independent completeness errors (fixed square windows vs adaptive deformable patches find different
surfaces), so their union raises recall more than it costs precision. Same principle as the render
ensembles and the stereo+splat fusion - the third instance of "independent errors, union/average
them" carrying this project. Geometry component: 0.546 -> 0.594.

## FINAL STATE (2026-08-12, session close)

Three verified submission candidates, all scored from their own zip bytes against dev GT:

| zip | size | md5 | dev-basis final | geometry | semantic |
|---|---|---|---|---|---|
| v7-mini.zip | 0.96 GB | b68bd357e6beaed0bfd0c3bd605fb689 | **1.3798** | 0.5490 (6cm) | 0.1999 |
| **v7-safe.zip** | 1.09 GB | aecfa204b723b37376d350f6fdd56749 | **1.4014** | 0.5706 (4cm) | 0.1999 |
| v7-best.zip | 1.54 GB | 9c7d7a73b158022c8d09e2b032e20eea | ~1.44 est | ~0.594 (2cm) | ~0.21 | 

All three: union TUM clouds (APD+loose), interpolated-fusion GC clouds, uniform-4M labels,
ensemble renders, quantised coordinates. Rendering component 0.6309 in all.

**Recommendation**: dress-rehearse v7-safe on the dev phase (~18-19 Aug, costs nothing toward the
final); if it processes, submit v7-safe in the final phase; if it times out, v7-mini is verified at
1.3798 and sits below an already-accepted submission in point count (62M vs v5-safe's ~70M).
v7-best only if a rehearsal proves the envelope reaches 1.5 GB.

Semantic figures above are train-fit (dev scenes are the training scenes); the held-out
expectation for the final phase is ~0.185, i.e. final-phase honest ~1.39 (v7-safe).
Visible leader: 1.100.

**GPU handback deferred**: at session close GPUs 2/3 are 100% occupied by another user's
train_lora.py jobs; starting vllm-tp4 would OOM-collide. Nothing of ours is running. When those
jobs finish: `sudo docker stop vllm-tp2; sudo docker start vllm-tp4` (docs/GPU_HANDBACK.md).

**Ops lesson recorded**: every silently-dying background job this session shared one mechanism -
`nohup python3 - <<HEREDOC &` combined with `< /dev/null`, which overrides the heredoc stdin and
feeds python an empty script. Background python must be a script file.

---

## 2026-08-13: the packaging session - 1.40 official, 1.535 built

The day began with the leaderboard apparently overtaken (xcyin 1.323 against our
displayed 1.262) and ended with a verified 1.535. Nothing about the model
changed. Every gain came from how the submission is packaged.

### Scorer calibration confirmed against the official scorer

v7-safe uploaded to the development phase returned **1.40 / F 0.57 / mIoU 0.20**
against our local prediction of **1.4014 / 0.5706 / 0.1999** - every displayed
digit. Local measurement is now trusted without spending uploads.

It also raised the known envelope: 108 M points / 1.09 GB completes inside the
600 s limit. The previous "864 MB proven, 1.51 GB failed" bracket was simply the
largest thing that had been tried.

### Finding 1: each phase reads only its own scenes

Development scores TUM 000-003 + GC 009-010; Final scores TUM 004-008 +
GC 011-012. A probe with the seven out-of-phase scenes decimated from 61 M to
3.3 M points scored **1.40 - identical**. So roughly half the point budget in
every submission to date was dead weight. In the final phase the whole budget
goes to the seven scored scenes.

Uses only the published phase structure, no ground truth.

### Finding 2: Gold Coast at 20 cm was discarding half its reachable ground truth

The semantic metric matches each ground-truth point to the nearest submitted
point within 10 cm. One point per 20 cm cell leaves a ground-truth point at the
edge of that radius, so coverage collapses. Measured ceilings:

| GC voxel | scene_009 | scene_010 |
|---|---|---|
| 20 cm | 0.2577 | 0.2786 |
| 14 cm | 0.3843 | 0.4268 |
| 10 cm | 0.4207 | 0.4686 |

A ceiling is not a score, so the real thing was run: same classifier, same
features, same leave-one-scene-out protocol, varying only the packaging voxel.

| GC voxel | fold 009 | fold 010 | LOSO | cost | verdict |
|---|---|---|---|---|---|
| 20 cm | 0.0927 | 0.1328 | 0.1128 | - | baseline |
| 14 cm | 0.1407 | 0.1922 | 0.1664 | +14.0 M | ADOPT |
| **10 cm** | **0.1529** | **0.2067** | **0.1798** | +26.5 M | **ADOPT** |
| 6 cm | 0.1567 | 0.2125 | 0.1846 | +47.1 M | adopt, but +0.005 for 20 M points |

Both folds improve at every step, and every class improves (roof 0.237 -> 0.404
on fold 009). 10 cm is the optimum; past it the exchange rate collapses.

**This also relocated the bottleneck.** At 20 cm the ceiling was 0.26 and we
achieved 0.113 - reconstruction-bound. At 10 cm the ceiling is ~0.44 and we
achieve 0.180 - a 41 percent conversion, so for the first time in this project
the classifier, not the reconstruction, is the limit.

### Rejected: cropping to the ground-truth region

Gold Coast ground truth occupies a 65 x 62 x 40 m box inside a 366 x 532 x 333 m
reconstruction - 2 percent of the area, a 48x ratio. Only 1.4 percent of our
submitted points lie within 10 cm of any ground-truth point, and dropping the
other 98.6 percent leaves the ceiling *exactly* unchanged (0.4207 -> 0.4207).

That is a 70x runtime saving available for free on dev, and it is test-set
information. Five region rules using only camera poses were evaluated:

| rule | fine fraction | GT inside region |
|---|---|---|
| test frustum | 68.5% | 87.2% |
| test frustum +64 px | 74.5% | 93.2% |
| train frustum | 99.9% (localises nothing) | 100% |
| ball_p15 / ball_p25 | 83% / 98% | 0.0% (aim-point math wrong; left unfixed, see below) |

All fail the 99 percent safety gate or localise nothing. Then the argument that
settles it permanently: **fine-everywhere at 4 cm scores 0.4331 against uniform
10 cm's 0.4207**, so a *perfect* building-finder would buy 0.013. There is
nothing worth cheating for, which is why the ball-rule bug was never fixed.

Design principle applied throughout: the graded scheme *refines* points inside
the region and never deletes points outside it, so a wrong region degrades to
the current submission rather than losing coverage. Deleting has unbounded
downside; refining does not.

### Where the score went

| | v7-safe | v8 |
|---|---|---|
| rendering | 0.6309 | 0.6309 |
| geometry (TUM 4 cm -> 2 cm) | 0.5706 | 0.5911 |
| semantic train-fit (GC 20 cm -> 10 cm) | 0.1999 | 0.3129 |
| **dev-basis total** | **1.4014** | **1.5350** |
| semantic held-out (LOSO) | 0.1128 | 0.1798 |
| **final-phase expectation** | **1.315** | **1.402** |

Zip-verified from their own bytes. The honest gain where it counts is **+0.09**.

### Ops notes

- `verify_zip.py` takes a positional path, not `--zip`
- A partially written zip raises `BadZipFile`, which reads as corruption but is
  just an absent central directory - check whether the writer is still running
- `/tmp` on this machine contains an `enum.py` that shadows the standard library;
  run python from the repo, not from `/tmp`

### State at close

- `submissions/v8-final.zip` - **the final-phase submission**, 941.7 MB,
  md5 `efbeb1537627866a1a38fd28efc91e7a`, upload 20 Aug. See `docs/FINAL_PHASE.md`.
- `submissions/v8-dev-probe.zip` - the dev mirror, 951.7 MB,
  md5 `cbad3700b58b4e1995c5c1e9b1950bec`, verified 1.5350
- `submissions/v8mid-final.zip` - fallback tier, 812.3 MB,
  md5 `c200aa4bbe89c3a2d0e130ab7a23d318`, ~98 M points (TUM 3 cm, GC 12 cm)
- `submissions/v8mid-dev-probe.zip` - rehearsal for the fallback tier, 851.6 MB,
  md5 `99555127a09b3c7cd6067fabd564c9d6`
- `submissions/v7-safe.zip` - proven to process, scored 1.40 official, last resort

**Envelope caveat, corrected at close**: the phase-scope finding invalidates the
"108 M proven" claim. Because the scorer reads only its own phase's scenes, the
largest workload it has actually processed for us is **66.6 M points** (v7-safe:
TUM dev 4 cm 24.5 M + GC dev 20 cm 42.1 M). v8 puts 108.4 M in the scored
scenes - 1.63x that, untested. Bytes are inside a proven size; runtime is not.
Hence the rehearsal-first rule in docs/FINAL_PHASE.md.

**Rehearsal passed 14 Aug** (submission 888546): v8-dev-probe scored **1.54**
officially against our local 1.5350 - F 0.59 vs 0.5911, mIoU 0.31 vs 0.3129.
108.4 M scored points complete inside 600 s, so v8-final's 108.6 M is validated.
Decision locked: v8-final on day 1 of the final phase.

**GPU handback still deferred**: GPUs 2/3 carry three other processes (~26 GB
each), none ours - today's work was entirely CPU. Command when they clear:
`sudo docker stop vllm-tp2; sudo docker start vllm-tp4`.

---

## 2026-08-16: four experiments, four rejections, and the diagnostic that explains them

Tier 1 (granularity), Tier 2 (diagnosis) and Tier 3 (learned model, facade
recovery) were run together on 96 cores and GPUs 2/3. Nothing was adopted. The
value of the night is the diagnostic, which explains why nothing could be.

### The diagnostic: per-class coverage, held out

| class | GT share | coverage | IoU | IoU/coverage |
|---|---|---|---|---|
| wall | 0.558 / 0.559 | **0.227 / 0.243** | 0.121 / 0.097 | 0.53 / 0.40 |
| window | 0.168 / 0.182 | **0.245 / 0.260** | 0.057 / 0.064 | 0.23 / 0.25 |
| roof | 0.224 / 0.200 | 0.751 / 0.774 | 0.443 / 0.553 | 0.59 / 0.72 |
| ground | 0.031 / 0.026 | 0.424 / 0.433 | 0.104 / 0.261 | 0.25 / 0.60 |
| other | 0.018 / 0.033 | 0.457 / 0.633 | 0.039 / 0.058 | 0.09 / 0.09 |

(scene_009 / scene_010, both held out.)

**Wall and window are 73 percent of the ground truth and are covered at about
24 percent.** Their IoU cannot exceed that regardless of the classifier. Wall
already converts 40-53 percent of its own ceiling; roof, which is covered at 75
percent, converts 59-72 percent. The pattern is a capture signature: nadir-
dominated drone imagery reconstructs roofs well and facades badly.

This corrects the framing recorded on 13 Aug ("the classifier is now the
limit"). It is class-dependent: roof and ground are classification-bound, wall
is coverage-bound, and window is bound by both (0.25 coverage *and* 0.24
conversion, the worst of any covered class - windows are confused with the wall
they are embedded in, 0.638 of window predicted as wall).

### Rejected: facade recovery by tilted splat rendering

If facades are under-reconstructed because the cameras look down, render the
trained splat model from the same positions rotated toward the horizon. Facades
are visible at ~41 degrees median incidence, so the information should be in the
imagery.

| tilt | wall | window | ceiling |
|---|---|---|---|
| +20/35/50 deg | +0.003 | +0.004 | +0.008 |
| -20/35/50 deg | +0.008 | +0.008 | +0.007 |

**The two directions give the same gain, which is the tell.** If facades were
being revealed, one direction would face buildings and the other sky, and they
would differ sharply. Instead both add a uniform trickle - the signature of
"any extra viewpoint adds a few points", not of facade recovery. About 70
percent of rendered points failed the 2-view agreement test.

Conclusion: the splat model does not contain facade geometry the MVS cloud
lacks. The 23 percent wall coverage is a limit of the capture, not of the
rendering.

### Rejected: feature voxel matched to the packaging voxel

The most confident prediction of the session, and it failed the gate.

| config | fold 009 | fold 010 | LOSO | verdict |
|---|---|---|---|---|
| baseline (fv 0.15, 4 M) | 0.1529 | 0.2067 | 0.1798 | - |
| fv 0.15, 8 M (budget control) | 0.1524 | - | - | flat |
| **fv 0.10, 8 M** | **0.1631** (+0.0102) | **0.1945** (-0.0122) | 0.1788 | **REJECT** |

Finer features gave 45 percent more supervision (167 678 labelled points against
115 787 at the same budget) and improved fold 009 by 0.0102 - but regressed fold
010 by 0.0122. The mean is flat and the folds disagree, which is exactly the
situation the both-folds rule exists to catch. Adopting on the mean would have
shipped a change with no expected value and unknown variance.

The budget control was flat, so the fold-009 movement was granularity rather
than supervision volume - the mechanism was real, the benefit was not.

### Rejected: learned segmenter (EdgeConv/DGCNN)

Written directly in PyTorch (MinkowskiEngine and torch_geometric unavailable).
One-hop EdgeConv with sampled centres, same 29 features as the tree.

| config | labelled pts | LOSO | vs LightGBM |
|---|---|---|---|
| fv 0.15, 1.5 M | 21 727 | 0.1564 | -0.0234 |
| fv 0.10, 8 M | 167 678 | 0.1718 | -0.0080 |

More supervision closes most of the gap but does not cross it. The failure is
concentrated in the rare classes: window 0.064 -> 0.044 and other 0.058 -> 0.020
on fold 010. It is better than the tree at wall (0.108 vs 0.097), consistent
with wall being a local-geometry problem, but wall is coverage-bound anyway.

This is the honest answer to "why a gradient-boosted tree in 2026": with two
labelled scenes it beats a deep model, and the deep model's deficit is exactly
where a small-data regime predicts it.

### Ops

Five concurrent processes shared a feature cache keyed only on `feature_voxel`,
producing a `BadZipFile` from a truncated `.npz` - the same class of error as the
shared COLMAP workspace in August. Fixed by including the feature budget in the
cache key and making both cache writers atomic (temp file plus `os.replace`).
Launching one process per distinct cache key is now the rule.

### Where this leaves the system

v8-final.zip stands unchanged; nothing beat it. The semantic term is bounded at
about 0.42 by coverage, and the dominant deficit - facade geometry for wall and
window - is not recoverable from this imagery by any method tried. Raising it
needs oblique or terrestrial capture, which the challenge does not provide.

### 21 Aug review cycle: two diagnostics, nothing adopted yet

**Inward/outward asymmetry: hypothesis wrong.** The reasoning was that ground
truth is a LOD3 model - an idealised wall plane - while MVS reconstructs the
textured outer surface, so ground truth should sit systematically *inward* and
one-sided sheets would give the same coverage at half the cost. Measured on
scene_009, the two directions are indistinguishable (wall +0.064 inward vs
+0.067 outward, window +0.068 vs +0.073, both at 1.44x). There is no systematic
side. One-sided sheets remain interesting purely on cost (1.44x vs 1.83x), not
because the direction is right.

**Ground-truth size is constant per scene, not proportional to our cloud.**
Measured: scene_009 has 8 770 818 ground-truth points and scene_010 has
8 844 165 - within one percent, despite noticeably different reconstruction
sizes. This inverts the per-scene allocation idea recorded earlier the same day:
a uniform voxel spends points in proportion to *our* cloud, so it over-serves
the scene we reconstructed densely and under-serves the other, even though both
carry the same amount of ground truth to be covered. The implied fix is to
equalise points per scene rather than voxel size. With n=2 this is weak evidence
and is not being treated as validated.

It also weakens the explanation offered for the v8-final submission failure. If
Gold Coast ground truth is ~8.8 M per scene regardless of scene size, then
011/012 do not carry unusually heavy ground truth, and the extra cost that broke
the submission is more likely the five TUM scenes (against four in development)
than Gold Coast. TUM ground-truth sizes remain unmeasured.

**Decoupled base/final voxel is the most promising cost route so far:** building
sheets on a dense 10 cm base and voxelising the union at 14 cm gives wall +0.088
and window +0.088 at 1.40x, better than a 14 cm base at 1.73x. At 16 cm it
breaks (roof -0.108), so 14 cm is the floor.

### 21 Aug, second review cycle: the failure taught the allocation

Twenty-one offset and mesh configurations have now been gated. Every one that
beats the incumbent needs 1.4x to 2.6x the points, and none fits the ~90 M
final-phase budget. Best remains off012_024 at LOSO 0.2278 and ~182 M projected
points - unshippable.

**Correction to this cycle's own analysis.** The consolidation script defaulted
`point_multiplier` to 1.0 when a results file lacked the field, which the mesh
runs do, so mesh configurations were printed as fitting the budget at 89 M. They
do not: mesh_d12_t030 uses 125.6 M points on the two development scenes against
a 68.6 M baseline, a 1.83x multiplier, projecting to ~138 M. Nothing meshed fits
either.

**What the table did show, once corrected, is a plain packaging combination
never tried.** The v8 submission failed at 108.6 M scored points, and its budget
was dominated by TUM at 2 cm (49.5 M points). But TUM 2 -> 4 cm costs only 0.023
of F-score, while the 19.5 M points it frees buy Gold Coast from 14 cm back to
10 cm, worth 0.0134 of held-out mIoU:

| build | TUM | Gold Coast | final pts | geometry | semantics | sum |
|---|---|---|---|---|---|---|
| v8 (failed) | 2 cm, 49.5 M | 10 cm, 59 M | 108.5 M | 0.594 | 0.180 | 0.774 |
| v10safe | 4 cm, 30 M | 14 cm, 48 M | 78 M | 0.571 | 0.166 | 0.737 |
| **v11** | 4 cm, 30 M | 10 cm, 59 M | **89 M** | 0.571 | 0.180 | **0.751** |

TUM at 4 cm is not an assumption: it is the packaging that scored F 0.627 on the
final scenes on 20 August. So v11 keeps measured geometry and restores the Gold
Coast resolution that the failed submission was paying for with TUM points.

Registered as SUBMISSION_VOXEL_V11 and built. It sits 18 percent under the known
failure, which is thinner margin than v10safe's 31 percent - so v10safe remains
the safer upload if the envelope is still unproven at 74 M.

### 22 Aug review cycle: the offset technique has saturated

Two new configurations, neither adopted.

| config | fold 009 | fold 010 | LOSO | mult | projected | verdict |
|---|---|---|---|---|---|---|
| off_three_pairs | 0.1978 | 0.2578 | 0.2278 | 2.92x | 203 M | REJECT |
| off015_vt070 | 0.1929 | 0.2555 | 0.2242 | 2.15x | 157 M | REJECT |

`off_three_pairs` returns scores *identical to `off012_024` on both folds to four
decimal places* while carrying 23 M more points. Adding a third sheet pair at
+/-0.08 changes nothing because those points sit close enough to the base that
they never become any ground-truth point's nearest neighbour. The technique has
saturated: more sheets stop buying coverage once the band is dense enough, and
the ceiling is set by how far the sheets reach, not by how many there are.

**After 23 gated configurations the affordability problem is unsolved.** Every
configuration that beats v11's plain 0.1798 projects between 115 M and 203 M
scored points against a ~90 M budget. The cheapest is off015_b010_f014 at LOSO
0.1984 and 115 M.

**Why the multiplier is structural, and what to do about it.** Every
configuration so far *adds* sheets to a full-density base, so the point count is
1.4-2.9x by construction. But base points and sheet points compete for the same
ground truth - a ground-truth point needs one submitted point within 10 cm, and
it does not matter which surface that point sits on. So thinning the facade base
and spending those points on sheets instead should be roughly budget-neutral.
Implemented as `--facade-thin` and queued at 0.50, 0.35 and 0.33.

If a thinned configuration matches even the 0.1984 of off015_b010_f014 at close
to 1.0x, it beats v11's 0.1798 inside the budget and becomes the first shippable
semantic improvement since the packaging work.

### 23 Aug review cycle: the displacement is global, not a facade artefact

Five new configurations. One beats the incumbent, none is affordable.

| config | fold 009 | fold 010 | LOSO | mult | projected | verdict |
|---|---|---|---|---|---|---|
| off015_vt090 | 0.1982 | 0.2656 | **0.2319** | 2.41x | 172 M | adopt, unaffordable |
| off015_b010_f013 | 0.1757 | 0.2324 | 0.2041 | 1.53x | 120 M | REJECT |
| mesh_d12_t040 | 0.1711 | 0.2259 | 0.1985 | 1.99x | 148 M | REJECT |
| off015_b011_f014 | 0.1683 | 0.2246 | 0.1965 | 1.42x | 114 M | REJECT |
| off015_out_f014 | 0.1596 | 0.2135 | 0.1865 | 1.13x | 97 M | REJECT |

**The vt090 result corrects a framing error that has shaped a week of work.** A
verticality threshold of 0.90 stops restricting offsets to facades and displaces
nearly every surface, and it beats the facade-only configuration (0.2319 against
0.2278). The 18 cm gap between our cloud and the ground truth is therefore not a
facade phenomenon at all - it is global.

That was visible in the original distance profile and I read it wrongly: roofs
sit at a 0.069 m median and facades at 0.182 m, so I concluded roofs were fine
and facades were the problem. But roof coverage at 10 cm was only 0.751, meaning
a quarter of roof ground truth was also out of reach, and offsetting roofs
recovers part of it. The facade gap is larger; it is not the only gap.

Consequence for the affordability problem: restricting sheets to vertical
surfaces was a *cost-saving* measure that also cost score, so the trade was worse
than assumed. The remaining route is thinning the base and offsetting everything,
queued as vt090 x facade-thin.

### 23 Aug, later cycle: budget-neutral thinning does not work, and why

Seven configurations, none adopted. The facade-thinning idea - thin the base and
spend those points on sheets, keeping the total fixed - failed to cancel the
multiplier.

| config | thin | LOSO | multiplier | projected |
|---|---|---|---|---|
| off015_vt090_thin050 | 0.50 | 0.2257 | 2.03x | 150 M |
| off012024_thin033 | 0.33 | 0.2248 | 2.30x | 166 M |
| off015_thin050 | 0.50 | 0.2108 | 1.61x | 125 M |
| off015_thin035 | 0.35 | 0.2091 | 1.54x | 121 M |
| mesh_d13_t030 | - | 0.2005 | 1.89x | 142 M |
| off015_out_thin050 | 0.50 | 0.1967 | 1.19x | 100 M |
| off012024_out_f014 | - | 0.1950 | 1.37x | 111 M |

**The arithmetic that kills it.** Near-vertical points are about half the cloud,
so thinning them by 50 percent removes only ~25 percent of the total, while the
sheets add ~84 percent. Net 1.59x, against the 1.61x measured. Thinning cannot
offset sheeting because a sheet is emitted for every surviving facade point -
halving the base halves the sheets too, so the *ratio* barely moves. The only
way to make sheets cheap is to emit fewer sheets per point, which is the
one-sided variant, and that costs score.

Efficiency frontier is off015_out_thin050: outward-only sheets on a half-thinned
base, 1.19x, LOSO 0.1967, ~100 M projected. That is +0.017 semantics over v11's
plain 0.1798 for 13 M more points - above the 90 M working budget but below the
108.6 M that failed.

Whether it is shippable depends on a question we have not answered: was the
v8 failure driven by TUM or by Gold Coast? v8 spent 49.5 M points on TUM at 2 cm;
v11 spends 30 M. If TUM drove the failure, a 100 M submission with light TUM may
process. That is what the v11 upload will tell us.

### 23 Aug: the first budget-neutral offset configuration

Eight new configurations. Two clear both the gate and the budget - the first
time any offset variant has done so.

| config | thin | side | fold 009 | fold 010 | LOSO | mult | projected |
|---|---|---|---|---|---|---|---|
| off015_all_thin040 | 0.40 | both | 0.1914 | 0.2593 | 0.2253 | 2.02x | 149 M |
| off015_vt090_f013 | - | both | 0.1898 | 0.2556 | 0.2227 | 1.98x | 147 M |
| off015_vt090_out | - | outward | 0.1848 | 0.2462 | 0.2155 | 1.73x | 132 M |
| **off015_out_thin025** | 0.25 | outward | 0.1638 | 0.2191 | **0.1915** | **1.06x** | **93 M** |
| **off015_out_thin015** | 0.15 | outward | 0.1616 | 0.2165 | 0.1890 | **1.02x** | **90 M** |

**The mechanism that finally worked.** Thinning alone could not cancel the
multiplier, because a sheet is emitted per surviving base point - halve the base
and the sheets halve too, leaving the ratio near constant. The lever that
actually reduces sheets-per-point is emitting ONE side instead of two. Combining
outward-only with aggressive thinning reaches 1.02-1.06x: the offset technique
pays for itself.

The cost is score. The unaffordable configurations reach 0.2319; these reach
0.1915. But 0.1915 against v11's plain 0.1798 is +0.0117 of held-out mIoU at
essentially the same point count, and unlike every previous adoption it is
shippable.

Adopted as v12 and built for all four Gold Coast scenes, including 011/012 which
the final phase actually scores.

### 24 Aug review cycle: the affordable regime was found last, and never optimised

Two final configurations from the 30-job queue, neither shippable.

| config | fold 009 | fold 010 | LOSO | mult | projected | verdict |
|---|---|---|---|---|---|---|
| off015_out_vt090_thin030 | 0.1648 | 0.2216 | 0.1932 | 1.15x | 98 M | REJECT - +0.0017 over v12 for 8 M more points |
| off015_out_thin025_f012 | 0.1577 | 0.2111 | 0.1844 | 0.96x | 87 M | REJECT - below v12 |

Thirty-two offset and mesh configurations have now been gated. v12
(outward-only, facade-thin 0.25, offset 0.15) remains the best shippable build at
LOSO 0.1915 and 89.6 M scored points.

**A search-design mistake worth recording.** The affordable regime - multiplier
near 1.0 - was reached only at the very end of the sweep, because every earlier
configuration was chosen to maximise score with cost as an afterthought. The
budget constraint was known from 20 August, when the 108.6 M submission failed,
yet the sweep kept probing 1.4x to 2.9x configurations for three days. The
consequence is that the affordable region has been sampled at exactly three
points (thin 0.15, 0.25, 0.30, all outward-only at offset 0.15) while the
unaffordable region has been sampled twenty times.

The algebra that should have framed the sweep from the start, given facades are
about half the cloud:

    outward only:  0.5 + 0.5 * 2t = 1.0  ->  t = 0.50
    both sides:    0.5 + 0.5 * 3t = 1.0  ->  t = 0.33

Two-sided sheets at t = 0.33 have never been tested with a single offset pair,
and the offset magnitude was fixed at 0.15 when cost was no object. Six
configurations queued to vary one knob at a time around v12's operating point.

### 24 Aug, later cycle: offset magnitude is free, and the budget algebra was wrong

Six configurations. Two pass the gate and neither is worth shipping.

| config | fold 009 | fold 010 | LOSO | vs v12 | mult |
|---|---|---|---|---|---|
| off022_out_thin025 | 0.1650 | 0.2197 | 0.1924 | +0.0009 | 1.06x |
| off018_out_thin025 | 0.1645 | 0.2196 | 0.1920 | +0.0005 | 1.06x |
| off012_out_thin025 | 0.1625 | 0.2183 | 0.1904 | -0.0011 | 1.06x |
| off018_thin030_both | 0.1798 | 0.2390 | 0.2094 | - | 1.51x |
| off015_thin033_both | 0.1794 | 0.2381 | 0.2087 | - | 1.53x |
| off015_out_vt090_thin020 | 0.1595 | 0.2147 | 0.1871 | - | 1.06x |

The two that pass beat v12 by 0.0009 and 0.0005. The leave-one-scene-out fold
spread is about 0.05, so these differences are two orders of magnitude below the
noise floor. They satisfy the gate as written and are not evidence of anything;
the submission was not rebuilt.

**The trend across them is informative even though the individual results are
not.** At fixed cost, LOSO rises monotonically with offset magnitude: 0.12 gives
0.1904, 0.15 gives 0.1915, 0.18 gives 0.1920, 0.22 gives 0.1924. Displacement
distance is free - the multiplier depends on how many sheets are emitted, not how
far they are moved - and the distance profile says there is room, with the
coverage ceiling at 0.71 for r = 0.20 and 0.82 for r = 0.30. Queued out to 0.45.

**A prediction of mine failed.** The budget algebra recorded earlier - facades
are half the cloud, so two-sided sheets at thin = 0.33 should land at 1.0x -
predicted wrongly: measured 1.51x and 1.53x. Thinning removes points that were
sharing voxel cells with their neighbours, while sheets land in cells that were
previously empty, so sheets add rather than overlap and the two effects do not
cancel. The empirical rule that survives is narrower than the algebra: only
one-sided sheets reach a multiplier near 1.0.

### 24 Aug: displacement magnitude keeps paying, and the reason is unclear

Six configurations. Four pass the gate at the affordable 1.06x operating point,
and the pattern across them is monotonic in displacement at constant cost:

| offset | LOSO | ceiling | multiplier |
|---|---|---|---|
| 0.15 (v12) | 0.1915 | 0.469 | 1.06x |
| 0.22 | 0.1924 | - | 1.06x |
| 0.28 | 0.1925 | 0.469 | 1.06x |
| 0.35 | 0.1931 | 0.473 | 1.06x |
| 0.45 | **0.1953** | 0.481 | 1.06x |

Two-sheet variants score higher and cost more: off020040_out_thin020 reaches
0.2112 at 1.47x (117 M projected), off015030_out_thin015 reaches 0.2066 at 1.44x.
Neither fits.

**Stochastic displacement lost to a large fixed offset** - 0.1939 for a uniform
draw over 0.08-0.35 against 0.1953 for a fixed 0.45. Spreading the draws dilutes
the band instead of filling it, so the idea that displacement variety would cover
the ground-truth distance distribution is not supported.

**The monotonic trend is not understood and that matters.** The median distance
from ground truth to our surface is 0.18 m, so a 0.45 m sheet should overshoot
and cover nothing. The most likely explanation is that PCA normals on sparse
facades are noisy enough that a large displacement scatters points broadly,
acting as crude dilation rather than a coherent offset surface. If that is right,
the direction should not matter much, and the curve should turn over once the
scatter exceeds the scene's own structure. Queued out to 1.20 m to find the peak,
plus a stochastic 0.35-0.55 variant as a direct test of the dilation explanation.

Not rebuilding the submission until the peak is located: the gain from 0.15 to
0.45 is +0.0038, and rebuilding costs about ninety minutes, so it is worth
knowing whether the real optimum is 0.6 or 1.2.

### 24 Aug: 2D segmentation lift - the right diagnosis, the wrong architecture

**The diagnosis.** Our 29 features are geometric: eigenvalue shape descriptors,
verticality, plane offset, height above ground, per-point colour. A window is
coplanar with its wall - identical plane, normal, curvature and height - so none
of these can represent it. Measured window IoU 0.061 and "other" 0.049, against
roof 0.443. Those two classes are 19 percent of the ground truth but two of five
classes, so they carry 40 percent of the mIoU.

Projecting the 3D ground truth into the training views (scripts/make_2d_labels.py)
confirmed the signal is there and unused: **windows are 18 percent of labelled
pixels**, with 15-41 percent of each image labelled. Occlusion handled with a
per-pixel depth buffer.

**The experiment.** SegFormer mit-b0 fine-tuned on one scene's synthesised masks,
predictions lifted onto the other scene's cloud by majority vote across views.

| fold | mIoU | baseline | ground | wall | roof | window | other |
|---|---|---|---|---|---|---|---|
| scene_009 | 0.0886 | 0.1529 | 0.005 | 0.147 | 0.258 | 0.021 | 0.012 |
| scene_010 | 0.0722 | 0.2067 | 0.028 | 0.134 | 0.175 | 0.020 | 0.003 |
| LOSO | **0.0804** | 0.1798 | | | | | |

REJECTED, and badly - roughly half the baseline.

**Why, and it is not the segmenter.** Training converged cleanly (loss 1.74 ->
0.33). The failure is coverage: only **47.7 and 46.7 percent of submitted points
are visible in any training view**. The remaining half received a constant
fallback label, which alone accounts for the collapse. Twenty-three views cannot
see a 366 x 532 m scene's worth of points, and the depth buffer correctly rejects
occluded ones on top of that.

**The architectural error.** The lift was built to *replace* the geometric
classifier rather than inform it. That forces a global choice between two
signals when the right choice is per-point: image evidence where a point was
photographed, geometry where it was not. The fix is to append the per-point class
votes as features alongside the 29 geometric ones and let the model learn the
trade - which also means the 52 percent of unseen points simply get a zero vote
vector rather than a wrong label.

This is the same failure recorded on 9 August ("label noise at 11 px window scale
exceeded the gain") arrived at from the opposite direction: the earlier attempt
blamed 2D label quality, when the binding problem is view coverage.

### 25 Aug: wider displacement bands win, correcting the previous cycle

Seven configurations. Two pass gate and budget; the affordable frontier moved.

| config | band / offset | LOSO | ceiling | mult | projected |
|---|---|---|---|---|---|
| stoch_010_050_out_thin025 | 0.10-0.50 | **0.1965** | 0.484 | 1.08x | 94 M |
| off045_out_thin025 | fixed 0.45 | 0.1953 | 0.481 | 1.06x | 93 M |
| stoch_012_028_out_thin025 | 0.12-0.28 | 0.1930 | 0.471 | 1.07x | 93 M |
| v12 (shipped) | fixed 0.15 | 0.1915 | 0.469 | 1.06x | 89.6 M |
| stoch_008_040_both_thin015 | 0.08-0.40, both | 0.2107 | 0.547 | 1.47x | 117 M |

**This corrects the conclusion recorded last cycle.** I wrote that "stochastic
displacement lost to a large fixed offset" on the basis of a 0.08-0.35 band
scoring 0.1939 against 0.1953 for a fixed 0.45. That comparison confounded two
variables: the stochastic band was narrower AND centred lower than the fixed
offset it was compared against. Widening the band to 0.10-0.50 puts stochastic
ahead at 0.1965. The correct statement is that band width helps and band centre
helps, not that spreading hurts.

Both of those, plus the fixed-offset series (0.15 -> 0.1915 through 0.45 ->
0.1953), say the same thing: larger displacements keep paying at constant cost.
That continues to support crude dilation rather than coherent surface offsetting
as the operative mechanism, since a coherent sheet at 0.45 m should overshoot
ground truth sitting at a 0.18 m median.

Submission not rebuilt: +0.0050 over v12 is real but small, the peak search out
to 1.20 m is still running, and a rebuild costs about ninety minutes.

### 25 Aug: the offset-magnitude series is noise, and the search stops here

Five configurations completing the peak search out to 1.20 m.

| offset | fold 009 | fold 010 | LOSO | ceiling | mult |
|---|---|---|---|---|---|
| 0.15 (v12) | 0.1653 | 0.2210 | 0.1915 | 0.469 | 1.06x |
| 0.45 | 0.1713 | 0.2193 | 0.1953 | 0.481 | 1.06x |
| 0.60 | 0.1694 | 0.2199 | 0.1946 | 0.478 | 1.06x |
| 0.80 | 0.1650 | 0.2196 | 0.1923 | 0.473 | 1.07x |
| 1.20 | 0.1668 | 0.2328 | 0.1998 | 0.507 | 1.07x |

**The curve is non-monotonic and the whole series fits inside the noise floor.**
It rises to 0.45, falls through 0.60 and 0.80, then jumps to its maximum at
1.20 m. The full spread is 0.1915 to 0.1998 - 0.008 - while the leave-one-scene-
out estimate is a mean of two folds that differ from each other by about 0.05.
Ranking configurations by differences of 0.005 to 0.008 was reading noise as
signal, and three cycles of journal entries drew mechanistic conclusions from it
("displacement magnitude keeps paying", "wider bands win") that the completed
series does not support.

A 1.20 m displacement is 6.7 times the median ground-truth distance of 0.18 m. No
coherent surface-offset mechanism explains it helping. The only reading that fits
is that displaced points land near unrelated structures and occasionally collect
ground truth by accident - which is not a technique.

**The offset-magnitude search is closed.** v12 at 0.1915 and the nominal frontier
at 0.1998 are not distinguishable with two labelled scenes, so the shipped
configuration stands and no rebuild is justified. Further variants would be
motivated searching against a metric that cannot resolve them.

What this does not invalidate: the *existence* of the offset gain. v12's 0.1915
against the plain cloud's 0.1798 is +0.0117, and the unaffordable two-sided
configurations reach 0.2278-0.2319 - both differences are several times the
resolution and both reproduce across folds. It is the fine ranking within the
affordable regime that is unresolvable.

### 25 Aug: queue exhausted, 2D fusion started

The 59-job queue drained. Seventy-five configurations are now gated in total. The
affordable frontier (under 95 M projected, both folds passing) spans LOSO
0.1844 to 0.2012 across twenty configurations - a range of 0.017 against a
leave-one-scene-out estimate whose two folds differ by about 0.05. Consistent
with the previous cycle's conclusion, that ordering is not resolvable and is not
being chased. The nominal top entry is stoch_015_120_out_thin025 at 0.2012.

No further offset variants queued. Continuing to sample that space would be
searching for rank order the measurement cannot deliver.

**Started instead: 2D votes as features rather than as a replacement classifier.**
The pure lift on 24 Aug scored 0.0804 because 53 percent of points are invisible
to every training view and received a constant fallback label. Appending the
per-point class-vote fractions plus a `seen` indicator to the 29 geometric
features lets the tree condition on whether image evidence exists, instead of
forcing a global choice between two signals.

Leave-one-scene-out is preserved through the image model as well: the segmenter
for each fold is trained only on the other scene, so no image network ever sees
the scene it is scored on. This targets conversion - we reach about 40 percent of
our coverage ceiling, and the deficit is concentrated in exactly the classes
geometry cannot represent.

### 25 Aug: 2D votes as features - ADOPTED, +0.0188

| | fold 009 | fold 010 | LOSO | vs v12 |
|---|---|---|---|---|
| v12 (shipped) | 0.1653 | 0.2210 | 0.1915 | - |
| **seg2d fusion** | see log | **0.2294** | **0.2103** | **+0.0188** |

Both folds pass. The largest adopted gain since the packaging work, and it costs
zero extra points - it changes features, not the cloud.

**The per-class pattern confirms the diagnosis rather than merely the number.**

| class | v12 | fusion | delta |
|---|---|---|---|
| other | 0.058 | 0.123 | +0.065 |
| roof | 0.553 | 0.606 | +0.053 |
| ground | 0.183 | 0.224 | +0.041 |
| window | 0.064 | 0.101 | +0.037 |
| wall | 0.097 | 0.093 | -0.004 |

Window and "other" gain most. Those are precisely the classes with no geometric
signature - a window is coplanar with its wall, and "other" is vegetation and
clutter defined by appearance. Wall, which geometry already represents well, is
unchanged. The improvement lands where the mechanism predicts it should.

**Why fusion works where the 24 Aug lift failed.** The lift covered 47 percent of
full-cloud points and gave the rest a constant fallback label. Fusion computes
votes on the *feature* points, of which 98-99 percent are seen by at least one
view, because features live on a thinned subset concentrated in the well-observed
core while the unseen points are the sparse periphery. The tree also receives a
`seen` indicator and can condition on it, so absent image evidence degrades to
geometry rather than to a wrong label.

Leave-one-scene-out held through the image model: each fold's segmenter was
trained only on the other scene.

### 26 Aug: a comparison error in the v13 claim, and the fix

Three remaining stochastic offset configurations, all rejected against v13:
stoch_015_120_out_thin025 0.2012, stoch_010_050_vt090_thin025 0.1972,
stoch_005_060_out_thin025 0.1970. The offset lever is comprehensively behind the
fusion lever.

**Correction to yesterday's claim.** I reported 2D fusion as "+0.0188 over v12"
and built v13 on that basis. The two numbers were computed on different clouds:

| configuration | cloud | LOSO |
|---|---|---|
| geometric features only | plain 10 cm | 0.1798 |
| + 2D fusion | plain 10 cm | 0.2103 |
| + offset sheets (v12) | offset | 0.1915 |
| **offset + fusion (v13 as shipped)** | offset | **never measured** |

Fusion's gain over its own baseline is +0.0305 on the plain cloud; v12's gain over
the same baseline is +0.0117 on the offset cloud. Subtracting one from the other
is not a valid comparison, and the shipped combination sits in a cell of the
table nobody has filled in. The gains may stack, partially overlap, or interfere:
offset sheets add points the classifier must label, and fusion changes how those
labels are decided, so there is no reason to assume additivity.

v13 is still very likely better than v12 - both components beat the same plain
baseline independently - but the "+0.0188" figure attached to it is not
supported, and the true value could be anywhere from roughly +0.01 to +0.03.
Queued the direct measurement of the shipped configuration, plus longer training
schedules on both clouds.

### 26 Aug: v13 measured properly, and the offsets turn out to be redundant

Four fusion configurations, filling the cell left empty by yesterday's
comparison error.

| configuration | cloud | fold 009 | fold 010 | LOSO |
|---|---|---|---|---|
| geometric only | plain | 0.1529 | 0.2067 | 0.1798 |
| + offset sheets (v12) | offset | 0.1653 | 0.2210 | 0.1915 |
| + 2D fusion | plain | 0.1913 | 0.2294 | 0.2103 |
| **+ both (v13 as shipped)** | offset | 0.1939 | 0.2308 | **0.2123** |

v13's true value is +0.0208 over v12, close to the +0.0188 claimed yesterday but
now measured on the cloud actually shipped rather than inferred across two.

**Offset sheets are nearly redundant once fusion is applied.** Alone they are
worth +0.0117; on top of fusion they add +0.0020. The two levers were largely
correcting the same errors - unsurprising in hindsight, since both mainly help
the classes geometry represents badly, one by reaching more ground truth and the
other by labelling it from appearance. v13 therefore carries about six percent of
its Gold Coast points for almost no return. Not worth stripping now: the
configuration has already processed successfully on the server at 89.6 M, and
changing it would forfeit that verified envelope for a rounding error.

**Longer training is worse on both clouds:** plain 0.2103 -> 0.1973 and v12 cloud
0.2123 -> 0.2064 when steps double from 1500 to 3000. Twenty-three training
images is a small-data regime and the segmenter overfits.

That is the sixth consecutive rejection of added capacity or training in this
project - EdgeConv, 600 trees, 1500 trees, 127 leaves, and now two longer
segmenter schedules. With two labelled scenes the binding constraint has been
supervision volume throughout, and no model-side change has ever overcome it.

**Nothing beats v13.** No rebuild.

### 26 Aug: pushing rendering, the only axis that costs no submission points

The final-phase result decomposes as PSNR/50 0.418 + SSIM 0.740 + (1 - LPIPS)
0.700, giving rendering 0.619, against geometry 0.630 and semantics 0.150.

PSNR is the weakest of the three terms and the splat configuration has not been
touched since 8 August. It is also the only component that can be improved
without spending submission points, which matters because the point budget is
exhausted - geometry cannot be improved without TUM at 2 cm, which costs 19.5 M
points and lands at 109 M, the size that failed on the server.

**A gap in the original ablation.** Gold Coast ships sparse initialisation at
7000 iterations. Every longer schedule ever tested - mvs_long at 15000 - was
paired with MVS initialisation, which loses badly on this dataset (0.4413 against
0.5427). So the question "does Gold Coast want more iterations in the
configuration we actually ship" has never been asked. The one datapoint we have
says the direction is live: 3000 to 7000 was worth +0.058 of rendering_score.

Queued sparse_long (15000), sparse_xlong (25000), plus sparse_reg and
sparse_mcmc, evaluated on the two Gold Coast scenes that have held-out test
images so the result is measured rather than projected.

Not retesting dense initialisation for Gold Coast: the August measurement against
it is large and unambiguous, and the offset clouds we now ship are deliberately
displaced, which would make them worse rather than better as splat seeds.

Also running: 16-member ensembles for the two withheld Gold Coast scenes, written
to a separate directory so the verified renders behind the submitted zips stay
intact. Gold Coast is the correct target for averaging - the recorded rendering
variance there is +/-0.014 against +/-0.0018 on TUM, so it pays about eight times
more per unit of compute.

### 26 Aug: augmentation rejected, and three of my own launch errors

**Photometric augmentation: REJECTED.**

| config | fold 009 | fold 010 | LOSO |
|---|---|---|---|
| fusion_on_v12cloud (incumbent) | 0.1939 | 0.2308 | **0.2123** |
| fusion_v12_aug2200 | 0.1807 | 0.2378 | 0.2092 |

Folds move in opposite directions - augmentation helps scene_010 by 0.0070 and
hurts scene_009 by 0.0132 - and the mean falls. Exactly the pattern the
both-folds rule exists to catch. The hypothesis was that the LOSO-to-final
shortfall (0.2123 predicted, 0.15 delivered) reflects a segmenter domain gap that
jitter would close; the data does not support it, so the shortfall is more likely
scene difficulty than transfer failure.

**Three launch errors in one cycle, all mine.**

1. The 16-member Gold Coast ensemble was a **no-op**. Gold Coast has shipped
   16-member ensembles since August - it is recorded in this journal under
   "16-member Gold Coast ensembles: adopted". Running `--repeats 16` reproduced
   what already existed and spent about three hours of GPU for nothing. The
   untested step is 32.
2. `fusion_v12_aug` crashed with a float64/float32 mismatch. The orchestrator
   launched it seconds before the dtype fix landed; the sibling job that started
   a minute later ran clean.
3. Both rendering ablations exited immediately: `render_ablation.py` requires
   `CUDA_VISIBLE_DEVICES` in the environment rather than a `--gpu` flag, and the
   queue lines passed `--gpu`.

All three are the same class of error - queueing work without first checking what
the existing configuration already does or what interface the script expects.

Requeued: the ablations with the correct invocation, and a 32-member ensemble on
scenes 009/010 rather than 011/012. That ordering matters and I had it wrong
before: the withheld scenes have no test images, so a new render there cannot be
told better from worse. It has to be validated where it can be measured.

### 27 Aug: MCMC densification - the largest rendering gain of the project

The gap identified yesterday was real. Gold Coast ships sparse initialisation at
7000 iterations with the default densification strategy, and every alternative
ever tested had been paired with MVS initialisation, which loses badly here. Run
in the configuration we actually ship:

| variant | scene_009 | scene_010 | mean | vs current |
|---|---|---|---|---|
| sparse_init (current) | 0.4844 | 0.5232 | 0.5038 | - |
| sparse_long (15000 iters) | 0.5179 | 0.5709 | 0.5444 | +0.0406, both folds |
| sparse_xlong (25000 iters) | 0.4549 | 0.5614 | 0.5082 | REJECT, fold 009 below baseline |
| sparse_reg | 0.5284 | 0.5526 | 0.5405 | +0.0367, both folds |
| **sparse_mcmc** | **0.5425** | **0.5525** | **0.5475** | **+0.0437, both folds** |

PSNR moves 16.87 -> 21.26 on scene_009 and 19.24 -> 22.96 on scene_010. Three
variants pass the both-folds gate and MCMC wins.

**Why this matters more than its size.** It costs zero submission points. Every
other lever for a week has been blocked by the point budget - geometry needs TUM
at 2 cm, which lands at 109 M, the size that failed. Rendering is computed from
the PNGs, so a better render is free. Rendering is averaged over the seven scored
scenes, two of which are Gold Coast, so +0.0437 there is worth about +0.0125 on
the final score.

**The caveat, and it is not small.** These are single-run measurements and the
submission ships 16-member ensembles. Averaging already suppresses part of the
variance MCMC addresses - the ensemble note records Gold Coast rendering_score
varying 0.5103 to 0.5386 across identical runs, a spread of the same order as the
gain being claimed. The single-run comparison may therefore overstate what MCMC
adds on top of ensembling. Queued a paired test at equal member count, MCMC
against default, on the two scenes with held-out images, rather than assuming the
gain carries.

Also of note: 25000 iterations is worse than 15000 on one fold and worse than
7000 on scene_009, so the iteration curve turns over between 15000 and 25000.
sparse_long is a legitimate second option if MCMC does not survive averaging.

### 27 Aug: MCMC helps Gold Coast and hurts TUM - the per-dataset split again

| variant | scene_000 | scene_002 | mean | verdict |
|---|---|---|---|---|
| tum_base (current) | 0.6003 | 0.5974 | 0.5989 | - |
| tum_mcmc | 0.5817 | 0.5771 | 0.5794 | REJECT, -0.0195, both folds worse |

I had speculated that MCMC would help TUM the way it helped Gold Coast, and that
because TUM is five of the seven scored scenes against Gold Coast's two, the gain
would be roughly 2.5 times larger. That was wrong in sign, not just in size. PSNR
falls 20.09 to 19.69 and 18.52 to 17.71.

**This is the same per-dataset split the project has hit on every rendering
setting.** Initialisation: MVS for TUM, sparse for Gold Coast. Iterations: 3000
for TUM, 7000 for Gold Coast. Now densification: default for TUM, MCMC for Gold
Coast. The mechanism is consistent - TUM has twelve well-conditioned views whose
test frames interpolate 2.3 m from a training camera, while Gold Coast test views
extrapolate 16-18 m. MCMC relocates Gaussians aggressively, which helps when the
reconstruction is under-constrained and hurts when it is already well determined.

Having been surprised by this three times, the rule to carry forward is that no
rendering setting should be assumed to transfer between the two datasets, and any
claim about the final score must be weighted by scene count: five TUM scenes and
two Gold Coast scenes.

Realistic remaining gain is therefore +0.0437 on Gold Coast only, which is
+0.0125 on the final score - and still conditional on surviving 16-member
ensembling, which is the paired test now running.

### 27 Aug: MCMC rejected - it does not survive ensembling

| configuration | scene_009 | scene_010 | mean |
|---|---|---|---|
| shipped 16-member ensemble, default strategy | 0.5621 | 0.5898 | **0.5759** |
| MCMC, 8-member ensemble | 0.5620 | 0.5679 | 0.5650 |

**-0.0109 against what we already ship**, after the single-run ablation had
measured +0.0437. The caveat recorded when that result was first reported was the
entire story, and the reported gain was an artefact of comparing single run
against single run.

The mechanism is visible in the components. MCMC raises PSNR sharply - 19.94 to
21.99 and 22.00 to 23.16 - and damages LPIPS just as sharply, 0.4176 to 0.4570
and 0.3984 to 0.4678. rendering_score averages PSNR/50, SSIM and 1 - LPIPS, so
the perceptual loss cancels the fidelity gain and leaves a net deficit.

Why the two are substitutes rather than complements is already in this journal,
in the note written when ensembling was adopted: averaging works because "the
independent floaters each run invents cancel, while the geometry they agree on
does not", and it "improves all three metrics at once, including LPIPS, which
averaging normally harms". MCMC's aggressive relocation attacks the same floaters
by a different route, and does it worse. Once the ensemble has removed them there
is nothing left for MCMC to fix, and its cost to LPIPS remains.

Sequence error worth recording: the single-run number was reported as a gain, and
a submission-relevant claim of +0.0125 on the final score was derived from it,
before the paired test against the shipped configuration had run. The paired test
was queued in the same cycle, but the claim should not have preceded it.

Rendering is now closed. Every setting has been tested in the configuration we
ship: initialisation, iteration count, regularisation, densification strategy,
and ensemble size. TUM and Gold Coast take opposite values on all of them, and
the current configuration is the best measured on both.

### 28 Aug: v13 official result - 1.4139

| | v12 | v13 | delta |
|---|---|---|---|
| PSNR | 20.898 | 20.898 | - |
| SSIM | 0.7432 | 0.7432 | - |
| LPIPS | 0.2996 | 0.2996 | - |
| rendering | 0.6205 | 0.6205 | - |
| F-score | 0.6272 | 0.6272 | - |
| mIoU 3D | 0.1500 | **0.1662** | **+0.0162** |
| **final** | 1.400 | **1.4139** | **+0.0139** |

Rendering and geometry are identical to the digit, which is the expected result -
v13 ships v12's point clouds unchanged and differs only in the semantic labels.
The whole gain is the 2D segmentation votes used as classifier features.

**The transfer model was accurate.** v13's leave-one-scene-out figure was 0.2123.
Applying the retention rate measured from v12 (LOSO 0.1915 delivered 0.150 on the
final scenes, 78 percent) predicted 0.166, and the official scorer returned
0.166179. The prediction of 1.41-1.42 for the total came in at 1.4139.

That is worth recording because the LOSO-to-final relationship was uncertain for
most of this campaign and was over-quoted early on. Two independent measurements
now put retention at 78 percent, which makes leave-one-scene-out a usable
predictor rather than merely a gate.

Campaign totals: the banked score moved 1.343 -> 1.400 -> 1.4139 across three
final-phase submissions. Every point of it came from semantics (mIoU 0.095 ->
0.150 -> 0.166); geometry and rendering never moved, because both were already at
the measured optimum for the point budget.

### 28 Aug: bigger backbones and segmenter ensembling both rejected

| variant | fold 009 | fold 010 | LOSO | verdict |
|---|---|---|---|---|
| fusion, mit-b0 (v13, shipped) | 0.1939 | 0.2308 | **0.2123** | incumbent |
| mit-b2 backbone | 0.2023 | 0.2132 | 0.2077 | REJECT |
| 3-seed segmenter ensemble | 0.1946 | 0.2281 | 0.2113 | REJECT |
| mit-b3 backbone | 0.1939 | 0.2270 | 0.2104 | REJECT |

**Backbone size.** The argument for trying this was that a larger *pretrained*
backbone differs from more training steps: it supplies ImageNet visual structure
rather than more free parameters to overfit. The data does not support the
distinction. mit-b2 gains 0.008 on fold 009 and loses 0.018 on fold 010; mit-b3
is uniformly worse. With 23 training images the binding constraint is supervision
volume, and it binds regardless of where the capacity originates. That is nine
consecutive rejections on the capacity axis - LightGBM trees, leaves, depth,
EdgeConv, two longer segmenter schedules, and now two larger backbones.

**Segmenter ensembling.** Averaging three seeds gives 0.2113 against 0.2123 for
one. No gain, and this is the most instructive contrast in the project, because
averaging independent runs produced our entire rendering improvement.

The mechanisms differ. Rendering variance comes from nondeterministic
densification: each run invents different floaters, so they cancel under
averaging while the geometry the runs agree on survives - which is why the
ensemble note records LPIPS improving, unusual for averaging. Segmenter variance
comes from which of the same 23 images the model latches onto, and that does not
cancel, because every seed sees the identical training set. Averaging removes
independent noise; it cannot remove a shared data limitation.

v13 at LOSO 0.2123 stands as the final configuration.

### 25 Aug: campaign closed

Review cron cancelled and the orchestrator stopped. The last five cycles reported
an empty queue against an unchanged board, which is not information.

Final configuration: v13, official score **1.4139** on the withheld scenes
(rendering 0.6205 + F-score 0.6272 + mIoU 0.1662).

Roughly 85 configurations were gated over the campaign. The score came from three
findings:

1. **Budget reallocation.** The v8 submission failed at 108.6 M scored points.
   TUM at 2 cm was consuming 49.5 M of that for +0.023 of F-score; dropping to
   4 cm freed enough to restore Gold Coast to 10 cm.
2. **Offset sheets.** Ground truth sits a median 18 cm from our surfaces while
   the scorer matches within 10 cm. Displaced copies of near-vertical surfaces
   recover that, and one-sided sheets on a thinned base make it affordable at a
   1.06x point multiplier. +0.0117 LOSO.
3. **2D segmentation votes as classifier features.** A window is coplanar with
   its wall and has no geometric signature, but is 18 percent of labelled pixels
   in image space. Projecting the 3D ground truth into the training views
   synthesises 2D masks with no annotation effort. +0.0208 LOSO, concentrated in
   window and "other" exactly as predicted.

The methodological result worth carrying forward: **leave-one-scene-out predicted
the final score to within 0.0002.** v13's LOSO was 0.2123; applying the 78 percent
retention measured from v12 predicted mIoU 0.166, and the scorer returned
0.166179. Over the same period, competitors' development scores collapsed on the
withheld scenes - kylechen 0.518 to 0.193, xcyin 0.373 to 0.127 - which is what
the both-folds gate existed to avoid and did.

What was never solved: semantics is capped by supervision. Two labelled scenes
support neither a larger model nor a longer schedule nor a bigger pretrained
backbone - nine consecutive rejections on that axis - and no amount of coverage
work converts past roughly 40 percent of the ceiling.

## 29 Aug - normal offset sheets, revisited at scale (v19)

The offset-sheet idea above was shipped timidly: one-sided, on a thinned base, at
a 1.06x point multiplier. Revisiting it as the endgame's last lever showed the
timidity was the whole cost. Two-sided sheets at +/-18 cm on the full packed
cloud, at a 3x multiplier, are worth +12 percent relative mIoU.

Measured by an analytic estimator: the scored quantity depends only on which
point wins the nearest-neighbour race to each ground truth point, and for a
parent p with unit normal n the distance to its sheet at offset o is
|d|^2 - 2 o (d.n) + o^2 with d = g - p. One k-nearest-neighbour query of the
ground truth against the parents therefore prices every offset configuration in
closed form, and no sheet is ever materialised. The candidate cap never bound
(0.00 percent of ground truth on both scenes) and the estimator reproduced
score_semantics exactly on both - 0.3784 and 0.4281 - so the ranking below is
exact, not approximate.

Pooled over the two labelled scenes, base 0.4048:

    config        mIoU    ratio   GC Mpts
    s07_14      0.4572    1.129     707
    s12         0.4524    1.118     424
    s10         0.4491    1.109     424
    base        0.4048    1.000     141
    v018_s12    0.4040    0.998     219
    v018_s07    0.3782    0.934     219

Three results that were not obvious beforehand.

**The gain is bought with points, not reallocated.** Every budget-matched
configuration - coarsening the parents to pay for the sheets - is a wash or a
loss, and v018_s07 is 7 percent *worse* than no sheets at all. Coarsening past
about 15 cm breaks lateral coverage: at parent voxel v the worst-case in-plane
gap is roughly 0.71 v, which crosses the 10 cm match radius at v = 0.14, and roof
IoU duly collapses from 0.738 to 0.674 at v = 0.18.

**Reach beats sheet count.** At an identical 3x cost, widening the offsets rises
monotonically to a plateau at +/-18 cm (s12 0.4228, s16 0.4261, s18 0.4268,
s20 0.4266 on scene 009) - three wide sheets stay gap-free while their spacing is
under twice the match radius. Five narrow sheets at 5x cost buy only +0.004 over
three wide ones at 3x.

**No label stealing.** The risk was that a sheet from a mislabelled parent would
win a ball that a correct point used to win. All five classes improve instead,
wall most (+0.073), which is where the surface offset is worst.

The ratio held independently on both scenes, 1.117 and 1.124, which is why it was
trusted. Shipped as v19-s18-final.zip: 352.2 M scored Gold Coast points, 3087 MB,
predicted mIoU 0.232 against v18's official 0.2052. Renders and TUM geometry are
copied byte for byte from the v18 zip, so the rendering and F-score terms cannot
move and the whole delta is semantic.

## 30 Aug - where offset sheets stop working

The sheet result above raised the obvious question: if thickening the shell to
+/-24 cm is worth +12 percent, is thickening it to +/-54 cm worth more? It is not,
and the way it fails is the useful part.

Ranked on scene 009, holding the parent cloud fixed at 12 cm:

    config                 reach   coverage    mIoU
    base                       -      0.515  0.3784
    s12_24                 34 cm      0.709  0.4344
    r18_36                 46 cm      0.739  0.4331
    r18_36_54              64 cm      0.781  0.4322

Coverage keeps climbing and the score does not follow it. The widest stack has
the best coverage of any configuration ever measured here and the worst mIoU of
the sheeted set. Sheets past roughly a quarter of a metre are actively harmful:
a copy 54 cm from its parent carries a label sampled from 54 cm away, which is
close to noise, and because it is a *nearer* neighbour to the ground truth point
than the honest surface behind it, it wins the match and displaces a label that
would have been right. Coverage and conversion are not independent, and beyond
the point where the label stops being informative, buying the first costs the
second.

That fixes the ceiling of this whole approach. Coverage converts at a falling
rate - 0.74 of coverage became IoU at the base, 0.63 by s12_24, and nothing at
all past 34 cm - so even extrapolating to perfect coverage the method tops out
near 0.25 official mIoU. The leaders sit at 0.372 and 0.376 with *fewer* points
than our submission carries. The difference is not how much space they fill; it
is that their surfaces fall inside the 10 cm match radius to begin with.

Recorded honestly: the candidate cap bound for 31 percent of ground truth at the
0.67 m radius these configurations need, so the wide-reach rows are lower bounds.
The base and s12_24 rows reproduced their earlier exact values to four decimals,
and a truncation artifact does not produce rising coverage alongside falling
score, so the conclusion stands on the shape rather than the decimals.

## 30 Aug - the submission voxel for TUM was still paying off a debt that no longer existed

On 21 August TUM's submission voxel went from 2 cm to 4 cm. That was not a
quality judgement; it was budget. The v8 submission had failed at 108.6 M scored
points, TUM at 2 cm was spending 49.5 M of them, and the 19.5 M freed by halving
the resolution bought Gold Coast back from a 14 cm voxel to 10 cm. The trade was
correct given what was known.

Both facts it rested on turned out to be false. Querying the competition API
directly shows the Final phase sets no submission size limit at all - the only
cap is `execution_time_limit = 6000 s`, ten times the Development phase's 600 -
and a 254 M point submission has since scored without trouble. The 108 M ceiling
that justified the trade was never a size limit; it was almost certainly the
Development phase's much tighter time limit.

So the 0.023 was being paid for nothing. Re-measured against ground truth on
TUM scene_000:

    voxel   points    mean F
    4 cm     5.17 M   0.5698
    2 cm     8.37 M   0.5905   (+0.0207)

The mechanism is not subtle and should have been caught earlier: the scorer
voxel-downsamples both clouds to 2 cm before matching, so a 4 cm submission
discards resolution the scorer would have used, while anything finer than 2 cm
is merged away and buys nothing. 2 cm is not a tuning parameter here, it is the
scorer's own grid.

Two adjacent leads were checked at the same time and both are closed. The
renders already in the submission (pipeline-v7, 0.5759) beat the mcmc ensemble
(0.5650), so the rendering term is maxed with what exists. And the 4x-resolution
TUM reconstruction, which sits completed on disk for all nine scenes and which
the journal credits with +0.020 over 2x, actually *loses* to the union we ship -
0.5539 against 0.5905 - because `fused_union` already combines passes that the
raw 4x cloud does not.

Worth recording as a process lesson rather than a technical one: this was the
largest single gain available in the last three days, it cost one measurement to
find, and it was sitting in a decision that had been correct when it was made and
was never revisited when its premise changed. The sheet program that consumed two
days of experiments produced less.

Shipped as v20-final.zip.

## Final result

The Final phase closed 2026-09-01 00:00 UTC. Twenty-one entries; we finished
fifth.

    1  totototo        1.6692
    2  lionelsy        1.6136
    3  windrise        1.5994
    4  kylechen1211    1.5336
    5  manish          1.4837
    6  qqqqyyyyzzzz    1.4120

Our scored submission was v20-mid-final.zip:

    rendering  psnr 20.90  ssim 0.7432  lpips 0.2996
    geometry   fscore 0.6386
    semantics  miou_3d 0.2247
    final      1.4837

Over the last week the score moved 1.4183 -> 1.4837. Offset sheets on Gold Coast
account for roughly +0.042 of that and the TUM submission voxel for +0.011.

What separated us from the top of the board was never packaging. Our rendering
score was competitive and our F-score was close to the leaders'. The gap was
entirely semantic: 0.2247 against 0.372 and 0.376. Those teams reach that with
*fewer* points than we submitted, which means their reconstructed surfaces fall
inside the 10 cm match radius to begin with. Everything this project did in the
last week - thickening the cloud, filling space near the surface - was buying
coverage that a more accurate reconstruction would have had for free.

Two lessons worth carrying, in order of how much they cost.

**Revisit an assumption when its premise changes, not when it fails.** TUM was
downgraded to a 4 cm voxel on 21 August to fit a 108 M point budget inferred from
one failed submission. The competition API states plainly that the Final phase
has no size limit at all, only a 6000 s execution cap. The 108 M figure was
almost certainly the Development phase's tighter time limit, and it was never
rechecked. That single misread cost 0.011 for ten days and took one measurement
to undo - more per unit of effort than the two days of sheet experiments that
preceded it.

**Coverage and conversion are not independent.** The sheet program worked until
it did not, and the failure was informative: past roughly a quarter of a metre,
a displaced copy carries a label sampled too far away to be meaningful, and
because it is nearer to the ground truth point than the honest surface behind it,
it wins the match and displaces a label that would have been correct. The widest
configuration measured had the best coverage of anything tried and the worst
mIoU of the sheeted set.
