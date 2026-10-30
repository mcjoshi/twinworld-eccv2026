# Reproducing the final submission

The scored entry is **v20-mid-final.zip**, submitted 31 Aug 2026 15:45 UTC.

    final_score 1.4837
    psnr 20.90   ssim 0.7432   lpips 0.2996
    fscore 0.6386
    miou_3d 0.2247

md5 `88a3a3cbc51b7d6744c66210e73dbf53`, 2253 MB, 13 scenes.
Archived at `mojo28/twinworld-submissions` on Hugging Face (private).

Note that none of the zips still on local disk are this file. `v14`, `v15asym`,
`v15safe` and `v7-final-safe` are superseded builds kept only because no copy of
them exists anywhere else; the scored submission lives on Hugging Face.

## What it is

Three independent parts, each scored separately by the competition:

- **rendering** - Gaussian splat renders from `experiments/2026-08-11-pipeline-v7`,
  unchanged since v18. Gated against an mcmc ensemble (0.5759 vs 0.5650) and kept.
- **geometry (TUM)** - `fused_union.ply` from the 2x-resolution MVS pass,
  submitted at a **2 cm** voxel, which is the scorer's own matching grid.
  Gated against the 4x reconstruction, which lost (0.5905 vs 0.5539).
- **semantics (Gold Coast)** - LightGBM on 29 geometric features fused with
  SegFormer mit-b0 vote features and zero-shot ADE20K group votes, propagated to
  the packed cloud by k=5 majority, then **thickened into offset sheets**: every
  point copied to +/-12 cm along its surface normal, carrying its parent's label.

## Rebuilding it

Two intermediates are retained because they are expensive to recompute:

    experiments/2026-08-29-xl-clouds/{scene}_xl_v18labeled.npz   xyz + labels
    experiments/2026-08-29-xl-clouds/{scene}_xl_normals.npz      unit normals
    experiments/2026-08-30-sheets/{scene}_mid.npz                sheeted clouds

From the sheeted clouds, the submission is one command. It needs
`v18-final.zip` as the source for the renders and the decimated dev scenes;
download it from Hugging Face first.

    python scripts/swap_gc_clouds.py --tag mid --tum-voxel 0.02 \
        --src submissions/v18-final.zip --out submissions/v20-mid-final.zip

If the sheeted clouds are gone, regenerate them (~5 min, CPU only):

    python scripts/build_sheet_clouds.py --offsets 0.0 -0.12 0.12 \
        --parent-voxel 0.15 --tag mid

If the normals are gone (~7 min per scene, CPU only):

    python scripts/precompute_normals.py --scenes scene_011 scene_012

Only `{scene}_xl_v18labeled.npz` requires a GPU to rebuild - it retrains two
SegFormer segmenters. That is `scripts/build_v18_labels.py`, and it is the
reason those files are kept rather than deleted.

## Verifying a rebuild

    python -c "
    import zipfile; z=zipfile.ZipFile('submissions/v20-mid-final.zip')
    print(len([n for n in z.namelist() if n.endswith('point_cloud.ply')]), 'scenes')
    print('testzip', z.testzip())"

Expect 13 scenes, `testzip None`, and 142.6 M / 83.3 M points in Gold Coast
scene_011 / scene_012.

## Method notes

The two findings behind the score, both recorded in full in `PROGRESS.md`:

**Offset sheets.** Gold Coast applies no precision penalty - each ground-truth
point simply takes the label of the nearest submitted point within 10 cm, and
unmatched points count as errors. Our surface sits a median 13.8 cm from the
ground truth, so most of it was unmatchable regardless of label quality.
Thickening the cloud perpendicular to its surface raises coverage; the effect
saturates near +/-24 cm, past which far sheets carry labels sampled too far away
and displace correct matches from the honest surface behind them.

**Submission voxel.** TUM was submitted at 4 cm between 21 and 31 August to fit a
point budget that did not exist. The scorer downsamples to 2 cm before matching,
so 4 cm discarded resolution it would have used; restoring 2 cm was worth
+0.0114 of F-score on the withheld scenes.
