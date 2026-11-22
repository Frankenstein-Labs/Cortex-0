#!/usr/bin/env python3
"""Publish an exported checkpoint to the Hugging Face Hub.

Requires ``HF_TOKEN`` with write access to the target organisation.  Without a
token the script exits with status 2 and explains what is missing; it never
invents a successful upload.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", default="export/Cortex-0")
    parser.add_argument("--repo-id", default="Frankenstein-Labs/Cortex-0")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--commit-message", default="Publish CORTEX-0 checkpoint")
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.exists():
        print(json.dumps({"status": "blocked", "reason": f"folder not found: {folder}"}))
        return 2
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print(
            json.dumps(
                {
                    "status": "blocked",
                    "reason": "HF_TOKEN is not set",
                    "action": "Create a write token at https://huggingface.co/settings/tokens, export HF_TOKEN, then rerun this script.",
                    "repo_id": args.repo_id,
                    "folder": str(folder),
                },
                ensure_ascii=False,
            )
        )
        return 2

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    url = api.create_repo(
        repo_id=args.repo_id, repo_type="model", private=args.private, exist_ok=True
    )
    api.upload_folder(
        folder_path=str(folder),
        repo_id=args.repo_id,
        repo_type="model",
        commit_message=args.commit_message,
    )
    print(json.dumps({"status": "published", "repo_id": args.repo_id, "url": str(url)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
