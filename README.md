# Filling space instead of reconstructing it

**TwinWorld Challenge @ ECCV 2026 — 5th of 21, and the challenge's highest geometry F-score.**

This repository is the full record of an entry to a large-scale semantic Gaussian
Splatting benchmark. The interesting part is not the ranking. It is that two of
the three scored terms turned out to reward decisions about *how the submission
was written* rather than how it was computed, and that the larger of those two
exploits has a limit we were able to measure precisely.

| | final | rendering | F-score | mIoU 3D |
|---|---|---|---|---|
| submission 910133 | **1.4837** | 0.6205 | **0.6386** | 0.2247 |
| best in challenge | 1.6692 | 0.6331 | *(ours)* | 0.4743 |

The F-score was the highest of all 21 entries. The gap to the top was entirely
semantic, and the rest of this README is mostly about why.

---

## The headline result: more coverage, worse labels

The semantic metric assigns each ground-truth point the label of the nearest
submitted point within 10 cm, counts unmatched points as errors, and applies **no
penalty for submitted points that are wrong or superfluous.** Our reconstructed
surface sits a median 13.8 cm from the ground truth. So most ground truth was
unmatchable no matter how good the labels were — the mass is there, it is just
displaced.

Densifying *along* the surface cannot fix that: new points land on the same
displaced surface. So we thickened the cloud *perpendicular* to it, emitting
labelled copies at fixed offsets along each point's surface normal. Official
mIoU went from 0.2052 to 0.2247, at zero cost to any other term.

Then it stopped working, and the way it stopped is the actual finding:

| offsets | reach | coverage | mIoU (scene_010) |
|---|---|---|---|
| none | — | 0.589 | 0.4281 |
| ±12, ±24 cm | 34 cm | 0.787 | **0.4962** |
| ±18, ±36 cm | 46 cm | 0.816 | 0.4966 |
| ±18, ±36, ±54 cm | 64 cm | **0.854** | 0.4925 |

The widest stack has **the highest ground-truth coverage we measured anywhere
and the lowest mIoU of any thickened variant.** Coverage and label conversion are
not independent. A copy displaced half a metre carries a label sampled from half
a metre away — close to noise — and because it is *nearer* to the ground-truth
point than the honest surface behind it, it wins the match and displaces a label
that would have been correct.

The effect replicates independently on both scenes that carry ground truth.
Saturation sits near ±18 cm; past roughly a quarter of a metre, buying coverage
costs conversion faster than it gains.

Two corollaries, both measured:

- **The gain cannot be reallocated.** Every budget-matched variant — coarsening
  the parent cloud to pay for the sheets — was neutral or worse. Coarsening past
  about 15 cm breaks lateral coverage, because the worst-case in-plane gap then
  exceeds the 10 cm match radius; roof IoU falls from 0.738 to 0.674.
- **Reach beats sheet count.** Three wide sheets at 3× the points beat five
  narrow ones at 5×.

## The second finding: the submission voxel is not yours to choose

The scorer voxel-downsamples both clouds to 2 cm before matching. A submission
coarser than that discards resolution the scorer would have used; finer than that
contributes points it immediately merges. 2 cm is not a hyperparameter, it is the
protocol.

| TUM submission voxel | points | mean F |
|---|---|---|
| 4 cm | 5.17 M | 0.5698 |
| **2 cm** | 8.37 M | **0.5905** |

We submitted at 4 cm for ten days. The reason is worth recording because the
mistake is a general one: an early submission failed at 108.6 M points, we
inferred a size limit from that single failure, and we traded TUM resolution away
to fit the budget. The competition API in fact specifies **no size limit at all**
— only a 6000 s execution cap — and the figure we had inferred was almost
certainly the development phase's much tighter time limit. The premise changed
and the decision was never revisited. Fixing it on the final day was worth more
per unit of effort than anything else attempted that week.

## What did not work

Recorded because the negative results cost the most time:

- **Nine consecutive attempts** to improve semantics by adding model capacity,
  longer schedules, or bigger pretrained backbones. All rejected. Two labelled
  scenes do not support them.
- **TSDF fusion** — three runs, ~15 CPU-days, abandoned without usable output.
- **Confidence pruning, pseudo-labelling, 12 rule-based label overrides, an
  expected-IoU objective.** All rejected by the gate.
- **4× image resolution** for MVS, which the literature and our own 2× result
  both suggested should help. It lost to the fusion union already in use
  (0.5539 against 0.5905).

Roughly 90 configurations were evaluated. The large majority were rejected.

## Method

Three branches, one per scored term, over a shared COLMAP front end at 2× image
resolution.

- **Rendering** — 3D Gaussian Splatting (gsplat), eight independently seeded
  models per scene, averaged pixel-wise in image space. Per-run variance on Gold
  Coast is an order of magnitude larger than on TUM, because held-out views
  extrapolate 16–18 m from the nearest training camera; averaging cancels the
  floaters each run invents independently and improves PSNR, SSIM and LPIPS
  simultaneously.
- **Geometry** — fused MVS cloud at the scorer's 2 cm grid.
- **Semantics** — LightGBM over 29 local geometric descriptors, concatenated
  with fine-tuned SegFormer `mit-b0` vote features and zero-shot ADE20K
  SegFormer-B2 group votes, then thickened as described above. The 2D masks are
  not annotated: they are synthesised by projecting the benchmark's own 3D labels
  into the training views.

## Methodology

Every candidate passed a gate fixed in advance: **both** leave-one-scene-out folds
had to improve over their own baselines, and the pooled score had to beat the
incumbent. No exceptions were made, and the gate rejected most of what was tried.

Leave-one-scene-out predicted the withheld-scene score to within **0.0002**
(predicted 0.166, returned 0.166179), which made it a reliable stand-in for the
leaderboard. Over the same period several competitors' development scores
collapsed on the withheld scenes — one from 0.518 to 0.193 — which is the failure
mode the gate existed to avoid.

## Honest limitations

- The coverage/conversion measurements come from **two scenes**, the only ones
  carrying ground truth. Whether this is a property of the metric or of two
  buildings in Queensland cannot be settled with the available data.
- The "no precision penalty" observation is specific to this benchmark's scorer.
- The space-filling strategy is metric exploitation, not reconstruction
  improvement. The leading entries reached mIoU 0.37 with *fewer* points than we
  submitted — their surfaces fall inside the 10 cm radius to begin with. That is
  a reconstruction-quality capability and the exploit is no substitute for it.

A symmetric metric, or a cap on submitted points relative to the reference cloud,
would remove the incentive without weakening the task.

## Repository

```
src/twinworld/      dataio, scoring, packaging, geometry, render, semantics
scripts/            ~100 experiment scripts, one per gated configuration
docs/PROGRESS.md    the full experimental record, ~90 configurations
docs/REPRODUCE.md   how to rebuild the scored submission
docs/CONVERT_TO_SUBMISSION.md
factsheet/          challenge factsheet, PDF and LaTeX
```

`docs/PROGRESS.md` is the primary artifact. It is a running journal written as the
work happened, including the mistakes, and it is where the numbers above come
from.

Point clouds, Gaussian checkpoints and the submitted archives are not in git —
they run to tens of gigabytes. They are published separately on Hugging Face at
`manishjoshiai/twinworld2026-verification`.

## Reproducing

Rebuilding the scored submission from the released checkpoints needs no GPU and
takes about fifteen minutes; see `docs/REPRODUCE.md`. Reproducing from raw
imagery needs COLMAP and roughly a GPU-day.

## Licence

MIT, see `LICENSE`. The challenge data itself is the organisers' and is not
redistributed here.
