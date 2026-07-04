"""
Upload model checkpoints, code, and model card to Hugging Face Hub.

Usage:
    python scripts/hf_model_upload.py
    python scripts/hf_model_upload.py --repo aleksandr-dzhumurat/geospatial-trajectory-transformer
    python scripts/hf_model_upload.py --dry-run
"""

import argparse
import glob
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
REPO_ID = "aleksandr-dzhumurat/geospatial-trajectory-transformer"


def collect_files() -> list[tuple[str, str]]:
    """Return list of (local_path, path_in_repo) pairs to upload."""
    files = [
        (str(REPO_ROOT / "data/checkpoints/ckpt_final.pt"), "ckpt_final.pt"),
        (str(REPO_ROOT / "scripts/config.json"),            "config.json"),
        (str(REPO_ROOT / "scripts/backbone.py"),            "backbone.py"),
        (str(REPO_ROOT / "scripts/experts.py"),             "experts.py"),
        (str(REPO_ROOT / "docs/hf_model_card.md"),          "README.md"),
    ]

    # Latest expert checkpoint per city (zero-padded step → lexicographic sort is safe)
    for expert in ("porto", "beijing"):
        matches = sorted(
            glob.glob(str(REPO_ROOT / f"data/checkpoints/expert_{expert}_step_*.pt"))
        )
        if matches:
            local = matches[-1]
            files.append((local, Path(local).name))
        else:
            print(f"[warn] No expert checkpoint found for '{expert}', skipping.")

    return files


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload model to Hugging Face Hub")
    parser.add_argument("--repo", default=REPO_ID, help="HF repo id (owner/name)")
    parser.add_argument("--dry-run", action="store_true", help="Print files without uploading")
    args = parser.parse_args()

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    files = collect_files()

    # Validate all local paths exist before starting any upload
    missing = [local for local, _ in files if not Path(local).exists()]
    if missing:
        print("Missing local files:")
        for p in missing:
            print(f"  {p}")
        print("\nRun `make download-ckpts` to fetch checkpoints from S3.")
        sys.exit(1)

    if args.dry_run:
        print(f"[dry-run] Would upload {len(files)} files to {args.repo}:")
        for local, remote in files:
            size_mb = Path(local).stat().st_size / 1024 / 1024
            print(f"  {local}  →  {remote}  ({size_mb:.1f} MB)")
        return

    api = HfApi()
    for local, remote in files:
        size_mb = Path(local).stat().st_size / 1024 / 1024
        print(f"Uploading {local} → {remote} ({size_mb:.1f} MB) ...")
        api.upload_file(
            path_or_fileobj=local,
            path_in_repo=remote,
            repo_id=args.repo,
            repo_type="model",
        )
        print(f"  done.")

    print(f"\nAll files uploaded to https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
