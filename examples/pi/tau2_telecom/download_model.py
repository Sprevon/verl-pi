#!/usr/bin/env python3
"""Download a public model snapshot and record the exact upstream revision."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source", choices=["huggingface", "modelscope"], default="huggingface")
    parser.add_argument("--revision")
    args = parser.parse_args()
    patterns = ["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"]
    upstream_files = {}
    if args.source == "huggingface":
        from huggingface_hub import HfApi, snapshot_download

        revision = HfApi().model_info(args.repo_id, revision=args.revision or "main").sha
        snapshot_download(args.repo_id, revision=revision, local_dir=args.output_dir, allow_patterns=patterns)
        revision_detail = {"commit": revision}
    else:
        from modelscope.hub.api import HubApi
        from modelscope.hub.snapshot_download import snapshot_download

        api = HubApi()
        revision = args.revision or "master"
        revision_detail = api.get_valid_revision_detail(args.repo_id, revision=revision)
        upstream_files = {
            item["Path"]: {"sha256": item["Sha256"], "revision": item["Revision"]}
            for item in api.get_model_files(args.repo_id, revision=revision, recursive=True)
            if item["Type"] == "blob"
        }
        snapshot_download(args.repo_id, revision=revision, local_dir=args.output_dir, allow_patterns=patterns)
        if revision_detail != api.get_valid_revision_detail(args.repo_id, revision=revision):
            raise RuntimeError("ModelScope revision changed during download; retry with a stable revision")
    destination = Path(args.output_dir)
    if not list(destination.glob("*.safetensors")):
        raise RuntimeError("Downloaded snapshot is missing model weights")
    files = {}
    for path in sorted(destination.iterdir()):
        if path.is_file() and path.suffix in {".json", ".safetensors", ".txt", ".model", ".jinja"}:
            if path.name == "pi_download_manifest.json":
                continue
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            expected = upstream_files.get(path.name)
            if expected and expected["sha256"] != digest:
                raise RuntimeError(f"Downloaded file does not match the upstream SHA-256: {path.name}")
            files[path.name] = {"bytes": path.stat().st_size, "sha256": digest}
            if expected:
                files[path.name]["upstream_revision"] = expected["revision"]
    manifest = {
        "repo_id": args.repo_id,
        "source": args.source,
        "revision": revision,
        "revision_detail": revision_detail,
        "local_dir": str(destination.resolve()),
        "files": files,
    }
    (destination / "pi_download_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
