"""Keep only interp-max points near the base surface: the band, not the fog.

Convergence-graded voxelling keeps 100 percent of ground truth but still costs
~100 M points per scene, because interp-max fills a VOLUME with splat-depth fog
and the fine tier inside the radius holds most of it. The coverage mechanism is
thinner than that: ground truth sits a median 13.8 cm from the base surface, so
the points that earn ceiling are the ones in the +/-30 cm band between our old
surface and the truth. Everything further is fog - dominant in count, absent in
ceiling.

Filters interp points by distance to the base cloud, unions, voxels, measures
count and ceiling, and saves the filtered points per scene for the fusion gate.
Base cloud is always kept whole, so the downside is bounded at the current ship.
"""
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from twinworld.dataio import CLASS_NAMES, NUM_CLASSES, Scene
from twinworld.package import submission_cloud
from twinworld.geometry.densify import robust_core
from twinworld.scoring import read_ply_xyz, voxel_downsample_indices

OUT = Path("experiments/2026-08-27-interp-slab"); OUT.mkdir(parents=True, exist_ok=True)
rows=[]
for sid in ("scene_009","scene_010"):
    s = Scene("gold_coast", sid)
    base,_ = submission_cloud(s, Path(f"experiments/2026-08-08-pipeline-v1/gold_coast/{sid}/mvs/fused_gs_i.ply"), voxel=0.10)
    nx = np.load(f"experiments/2026-08-26-interp-max/{sid}_interp.npz")["xyz"].astype(np.float64)
    nx = nx[robust_core(nx, s.train_frames)]
    gt, gl = read_ply_xyz(s.gt_ply, with_classification=True)
    bt = cKDTree(base)
    d2b, _ = bt.query(nx, k=1, distance_upper_bound=1.0, workers=-1)
    print(f"\n[{s.key}] interp {len(nx)/1e6:.0f}M, dist-to-base p50/p90 "
          f"{np.percentile(d2b[np.isfinite(d2b)],[50,90]).round(2)}, "
          f">1m: {(~np.isfinite(d2b)).mean()*100:.0f}%", flush=True)
    for slab in (0.20, 0.30, 0.45):
        band = nx[np.isfinite(d2b) & (d2b <= slab)]
        for v in (0.11, 0.13):
            union = np.vstack([base, band])
            sub = union[voxel_downsample_indices(union, v)]
            d,_ = cKDTree(sub).query(gt, k=1, distance_upper_bound=0.10, workers=-1)
            cov = np.isfinite(d)
            pc = {CLASS_NAMES[k]: round(float(cov[gl==k].mean()),4) for k in range(NUM_CLASSES) if (gl==k).any()}
            ceil = float(np.mean(list(pc.values())))
            rows.append({"scene":s.key,"slab":slab,"voxel":v,"band_pts":int(len(band)),
                         "points":int(len(sub)),"ceiling":round(ceil,4),"per_class":pc})
            print(f"  slab<={slab} v={v}: band {len(band)/1e6:6.1f}M -> union {len(sub)/1e6:6.1f}M  "
                  f"ceiling {ceil:.4f}  " + " ".join(f"{k[:4]} {x:.2f}" for k,x in pc.items()), flush=True)
            (OUT/"results.json").write_text(json.dumps(rows,indent=2))
    # save the 0.30 band for the fusion gate
    band = nx[np.isfinite(d2b) & (d2b <= 0.30)]
    np.savez_compressed(OUT/f"{sid}_interp.npz", xyz=band.astype(np.float32))
    print(f"  saved {sid}_interp.npz ({len(band)/1e6:.1f}M band points)", flush=True)
