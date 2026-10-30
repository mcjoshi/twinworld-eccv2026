"""Archive and upload each scene's Gaussian outputs as its ensemble completes.

The challenge's final verification asks for the reconstruction in its native
representation - the Gaussians themselves, not the point cloud derived from
them. One archive per scene rather than one for everything: a 32 GB upload that
fails at 90 percent has to start again, and per-scene archives also let the
organisers fetch only the scenes they want to check.

A scene is complete when render_ensemble has written its ensemble.json, which
happens only after all members have been averaged. Scenes already present in the
repository are skipped, so this is safe to re-run as the remaining GPUs finish.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import tarfile
import time
from pathlib import Path

REPO = "mojo28/twinworld2026-verification"
SRC = Path("experiments/2026-09-04-verification")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--staging", required=True, help="where to write the tars")
    ap.add_argument("--once", action="store_true", help="one pass, do not wait")
    a = ap.parse_args()

    from huggingface_hub import HfApi

    api = HfApi(token=os.environ["HF_TOKEN"])
    staging = Path(a.staging)
    staging.mkdir(parents=True, exist_ok=True)

    while True:
        done = sorted(SRC.glob("*/*/gs/ensemble.json"))
        present = {f for f in api.list_repo_files(REPO)}
        pending = []
        for e in done:
            scene = e.parent.parent            # .../<dataset>/<scene_id>
            name = f"{scene.parent.name}_{scene.name}.tar"
            # A scene ships only with the full ensemble. Members can be lost to
            # transient failures - four processes racing to JIT-compile the
            # gsplat extension at startup cost us member 0 twice - and a
            # seven-member archive would misrepresent what was submitted.
            n = len(list(scene.glob("gs/member*/splats.pt")))
            if n < 8:
                print(f"[upload] {name}: only {n}/8 members, skipping", flush=True)
                continue
            if f"reconstruction/{name}" not in present:
                pending.append((scene, name))

        if not pending:
            print(f"[upload] nothing pending ({len(done)}/13 scenes complete)",
                  flush=True)
            if a.once or len(done) >= 13:
                break
            time.sleep(120)
            continue

        for scene, name in pending:
            t0 = time.time()
            tar_path = staging / name
            if not tar_path.exists():
                with tarfile.open(tar_path, "w") as tf:
                    tf.add(scene, arcname=f"{scene.parent.name}/{scene.name}")
                    tf.add("docs/CONVERT_TO_SUBMISSION.md",
                           arcname="CONVERT_TO_SUBMISSION.md")
            mb = tar_path.stat().st_size / 1e6
            print(f"[upload] {name}: {mb:.0f} MB packed ({time.time()-t0:.0f}s), "
                  f"uploading", flush=True)
            api.upload_file(path_or_fileobj=str(tar_path),
                            path_in_repo=f"reconstruction/{name}",
                            repo_id=REPO, repo_type="model")
            tar_path.unlink()          # staging is not a second copy
            print(f"[upload] {name}: DONE ({time.time()-t0:.0f}s)", flush=True)

        if a.once:
            break


if __name__ == "__main__":
    main()
