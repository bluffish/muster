#!/usr/bin/env python3
"""Migrate one published run's replays to the Hugging Face dataset.

Extracts each replay's payload from the published HTML (the same split
muster-retemplate-replay uses), gzips it to a staging directory, uploads
the run folder plus its manifest to bluffish/muster-replays, verifies the
remote file count, and (with --delete-local) removes the local HTML
copies, leaving manifest.json in place for the viewer app.

Usage: migrate_run_hf.py <run-name> [--delete-local]
"""

from __future__ import annotations

import gzip
import shutil
import sys
from pathlib import Path

PAYLOAD_START = "const replay="
PAYLOAD_END = ", cfg=replay.config, teams=replay.team;"
REPO = "bluffish/muster-replays"
PUBLISHED = Path("/srv/muster/runs")
STAGING = Path("/srv/muster-hf-staging")


def extract_payload(html: str) -> str:
    _, found, remainder = html.partition(PAYLOAD_START)
    payload, ended, _ = remainder.partition(PAYLOAD_END)
    if not found or not ended or not payload.startswith("{"):
        raise ValueError("no replay payload found")
    return payload


def main() -> None:
    run = sys.argv[1]
    delete_local = "--delete-local" in sys.argv
    source = PUBLISHED / run / "replays"
    replays = sorted(source.glob("update-*.html"))
    if not replays:
        raise SystemExit(f"no replays under {source}")
    staging = STAGING / run
    staging.mkdir(parents=True, exist_ok=True)

    staged = 0
    for path in replays:
        target = staging / (path.stem + ".json.gz")
        if target.exists():
            staged += 1
            continue
        payload = extract_payload(path.read_text())
        with gzip.open(target, "wt", compresslevel=6) as handle:
            handle.write(payload)
        staged += 1
        if staged % 500 == 0:
            print(f"staged {staged}/{len(replays)}", flush=True)
    manifest = source / "manifest.json"
    if manifest.exists():
        shutil.copy(manifest, staging / "manifest.json")
    print(f"staged {staged} payloads for {run}", flush=True)

    from huggingface_hub import HfApi

    api = HfApi()
    api.upload_folder(
        folder_path=str(staging),
        path_in_repo=run,
        repo_id=REPO,
        repo_type="dataset",
        commit_message=f"replays: {run} ({staged} updates)",
    )
    remote = [
        name
        for name in api.list_repo_files(REPO, repo_type="dataset")
        if name.startswith(run + "/") and name.endswith(".json.gz")
    ]
    print(f"remote has {len(remote)} payloads", flush=True)
    if len(remote) < staged:
        raise SystemExit("remote is missing files; NOT deleting local copies")

    if delete_local:
        for path in replays:
            path.unlink()
        shutil.rmtree(staging)
        print(f"deleted {len(replays)} local replay files (manifest kept)", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
