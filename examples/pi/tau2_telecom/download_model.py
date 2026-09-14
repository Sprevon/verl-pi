#!/usr/bin/env python3
"""Download a public model snapshot and record the exact upstream revision."""

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--revision", default="main")
    args = parser.parse_args()
    revision = HfApi().model_info(args.repo_id, revision=args.revision).sha
    snapshot_download(
        args.repo_id,
        revision=revision,
        local_dir=args.output_dir,
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"],
    )
    destination = Path(args.output_dir)
    if not list(destination.glob("*.safetensors")):
        raise RuntimeError("Downloaded snapshot is missing model weights")
    manifest = {"repo_id": args.repo_id, "revision": revision, "local_dir": str(destination.resolve())}
    (destination / "pi_download_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
