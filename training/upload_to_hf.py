"""
Upload a fine-tuned FabSearch checkpoint to a HuggingFace Model repo.

Usage:
    # one-time auth
    huggingface-cli login         # or: set HF_TOKEN=<token>

    # default: uploads ~/Downloads/fab_search_vit_b32.pt to knayirp/fab-search-vit-b32
    python -m training.upload_to_hf

    # custom file / repo / remote name
    python -m training.upload_to_hf \
        --file "C:/Users/admin/Downloads/fab_search_vit_b32.pt" \
        --repo knayirp/fab-search-vit-b32 \
        --name fab_search_vit_b32_v1.pt \
        --message "Round 2 LoRA finetune"
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi, create_repo, login

DEFAULT_FILE = Path.home() / "Downloads" / "fab_search_vit_b32.pt"
DEFAULT_REPO = "knayirp/fab-search-vit-b32"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--file", default=str(DEFAULT_FILE), help="Local checkpoint path")
    p.add_argument("--repo", default=DEFAULT_REPO, help="HF repo id (user/name)")
    p.add_argument("--name", default=None, help="Filename in repo (default: same as local basename)")
    p.add_argument("--message", default="Upload fine-tuned ViT-B-32 weights")
    p.add_argument("--private", action="store_true", help="Create repo as private if it doesn't exist")
    args = p.parse_args()

    src = Path(args.file).expanduser().resolve()
    if not src.exists():
        raise SystemExit(f"File not found: {src}")
    remote_name = args.name or src.name

    token = os.environ.get("HF_TOKEN")
    if token:
        login(token=token, add_to_git_credential=False)

    create_repo(args.repo, repo_type="model", exist_ok=True, private=args.private)

    api = HfApi()
    url = api.upload_file(
        path_or_fileobj=str(src),
        path_in_repo=remote_name,
        repo_id=args.repo,
        repo_type="model",
        commit_message=args.message,
    )
    size_mb = src.stat().st_size / (1024 * 1024)
    print(f"Uploaded {src.name} ({size_mb:.1f} MB) -> {args.repo}/{remote_name}")
    print(f"Commit: {url}")


if __name__ == "__main__":
    main()
