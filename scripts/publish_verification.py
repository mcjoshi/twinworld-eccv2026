"""Publish the whole final-verification package to one repository.

The mojo28 account hit its private-storage limit partway through, so everything
moves to a fresh account. This uploads the small files first - they are what the
organisers read - then the bulky Gaussian archives, so a partial run still
leaves a coherent, useful repository rather than a pile of tarballs with no
factsheet.

Scenes are archived one at a time and each tar is deleted after upload, so
staging never holds more than one archive. Anything already present in the
repository is skipped, making the script safe to re-run as the last scene
finishes training.
"""

from __future__ import annotations

import argparse
import os
import tarfile
import time
from pathlib import Path

SRC = Path("experiments/2026-09-04-verification")

SMALL = [
    ("factsheet/TwinWorld2026_factsheet.pdf", "TwinWorld2026_factsheet.pdf"),
    ("factsheet/factsheet-latex.zip", "factsheet-latex.zip"),
    ("README.md", "README.md"),
    ("docs/CONVERT_TO_SUBMISSION.md", "CONVERT_TO_SUBMISSION.md"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--staging", required=True)
    ap.add_argument("--code-zip", required=True)
    ap.add_argument("--checkpoints", required=True)
    ap.add_argument("--private", action="store_true", default=True)
    a = ap.parse_args()

    from huggingface_hub import HfApi

    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(a.repo, repo_type="model", private=a.private, exist_ok=True)
    staging = Path(a.staging)
    staging.mkdir(parents=True, exist_ok=True)

    def present() -> set[str]:
        return set(api.list_repo_files(a.repo))

    def push(local: str | Path, remote: str) -> None:
        if remote in present():
            print(f"[skip] {remote}", flush=True)
            return
        t0 = time.time()
        api.upload_file(path_or_fileobj=str(local), path_in_repo=remote,
                        repo_id=a.repo, repo_type="model")
        print(f"[ok]   {remote} ({time.time()-t0:.0f}s)", flush=True)

    # 1. the documents the organisers actually read
    for local, remote in SMALL:
        if Path(local).exists():
            push(local, remote)
    push(a.code_zip, "twinworld-code.zip")

    # 2. the semantic branch's frozen output - rebuilds the submission on CPU
    push(a.checkpoints, "twinworld-semantic-checkpoints.tar")

    # 3. the Gaussians, one scene at a time
    while True:
        scenes = sorted(SRC.glob("*/*/gs/ensemble.json"))
        remaining = []
        for e in scenes:
            scene = e.parent.parent
            name = f"{scene.parent.name}_{scene.name}.tar"
            n = len(list(scene.glob("gs/member*/splats.pt")))
            if n < 8:
                print(f"[wait] {name}: {n}/8 members", flush=True)
                continue
            if f"reconstruction/{name}" not in present():
                remaining.append((scene, name))

        for scene, name in remaining:
            tar_path = staging / name
            t0 = time.time()
            with tarfile.open(tar_path, "w") as tf:
                tf.add(scene, arcname=f"{scene.parent.name}/{scene.name}")
                tf.add("docs/CONVERT_TO_SUBMISSION.md",
                       arcname="CONVERT_TO_SUBMISSION.md")
            print(f"[tar]  {name}: {tar_path.stat().st_size/1e6:.0f} MB "
                  f"({time.time()-t0:.0f}s)", flush=True)
            push(tar_path, f"reconstruction/{name}")
            tar_path.unlink()

        n_done = sum(1 for f in present() if f.startswith("reconstruction/"))
        print(f"[state] {n_done}/13 scenes uploaded", flush=True)
        if n_done >= 13:
            break
        time.sleep(120)

    print("[done] verification package complete", flush=True)


if __name__ == "__main__":
    main()
