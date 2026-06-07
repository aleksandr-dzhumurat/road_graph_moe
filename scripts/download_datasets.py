"""
Download Phase 1 datasets: Porto Taxi Trips and T-Drive (Beijing).

Usage:
    python scripts/download_datasets.py                  # download both
    python scripts/download_datasets.py --dataset porto
    python scripts/download_datasets.py --dataset tdrive
    python scripts/download_datasets.py --out-dir /custom/path
"""

import argparse
import os
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "data" / "raw"

PORTO_DIR = "porto"
TDRIVE_DIR = "tdrive"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _download_file(url: str, dest: Path, desc: str) -> Path:
    """Stream-download url → dest, showing a tqdm progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(
        desc=desc, total=total, unit="B", unit_scale=True, unit_divisor=1024
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))
    return dest


def _extract_zip(zip_path: Path, out_dir: Path) -> None:
    print(f"Extracting {zip_path.name} → {out_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)


# ---------------------------------------------------------------------------
# Porto Taxi Trips (UCI ML Repo ID 339)
# ---------------------------------------------------------------------------

def download_porto(out_dir: Path) -> None:
    """
    Download via the ucimlrepo Python package and save as CSV.
    Dataset: Taxi Service Trajectory — ECML PKDD 2015 (ID 339).
    Ref: https://archive.ics.uci.edu/dataset/339
    """
    dest = out_dir / PORTO_DIR
    dest.mkdir(parents=True, exist_ok=True)

    marker = dest / ".done"
    if marker.exists():
        print(f"[porto] Already downloaded at {dest}, skipping.")
        return

    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError:
        print("ERROR: ucimlrepo not installed. Run: pip install ucimlrepo")
        sys.exit(1)

    print("[porto] Fetching from UCI ML Repository (ID=339) …")
    dataset = fetch_ucirepo(id=339)

    features = dataset.data.features
    targets = dataset.data.targets

    features_path = dest / "porto_features.csv"
    targets_path = dest / "porto_targets.csv"

    features.to_csv(features_path, index=False)
    print(f"[porto] Features saved → {features_path}  ({len(features):,} rows)")

    if targets is not None:
        targets.to_csv(targets_path, index=False)
        print(f"[porto] Targets saved  → {targets_path}  ({len(targets):,} rows)")

    marker.touch()
    print("[porto] Done.")


# ---------------------------------------------------------------------------
# T-Drive (Beijing, Microsoft Research)
# Kaggle mirror: https://www.kaggle.com/datasets/arashnic/tdriver
# ---------------------------------------------------------------------------

TDRIVE_KAGGLE_DATASET = "arashnic/tdriver"


def download_tdrive(out_dir: Path) -> None:
    """
    Download T-Drive via the Kaggle API.
    Requires ~/.kaggle/kaggle.json with your API credentials.
    Get them at: https://www.kaggle.com/settings → API → Create New Token.

    Ref: https://www.microsoft.com/en-us/research/publication/t-drive-trajectory-data-sample/
    """
    dest = out_dir / TDRIVE_DIR
    dest.mkdir(parents=True, exist_ok=True)

    marker = dest / ".done"
    if marker.exists():
        print(f"[tdrive] Already downloaded at {dest}, skipping.")
        return

    try:
        import kaggle.api as kaggle_api
    except ImportError:
        print("ERROR: kaggle package not installed. Run: pip install kaggle")
        sys.exit(1)

    try:
        kaggle_api.authenticate()
    except OSError as exc:
        print(
            f"ERROR: Kaggle credentials not found ({exc}).\n"
            "  1. Go to https://www.kaggle.com/settings → API → Create New Token\n"
            "  2. Place kaggle.json at ~/.kaggle/kaggle.json\n"
            "  3. chmod 600 ~/.kaggle/kaggle.json"
        )
        sys.exit(1)

    print(f"[tdrive] Downloading {TDRIVE_KAGGLE_DATASET} via Kaggle API …")
    kaggle_api.dataset_download_files(
        TDRIVE_KAGGLE_DATASET,
        path=str(dest),
        unzip=True,
        quiet=False,
    )

    marker.touch()
    print(f"[tdrive] Done → {dest}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DATASETS = {
    "porto": download_porto,
    "tdrive": download_tdrive,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Phase 1 GPS trajectory datasets.")
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS) + ["all"],
        default="all",
        help="Which dataset to download (default: all)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Root output directory (default: {DEFAULT_OUT})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    targets = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]

    print(f"Output directory: {args.out_dir}")
    for name in targets:
        print(f"\n{'='*50}")
        print(f" Dataset: {name.upper()}")
        print(f"{'='*50}")
        DATASETS[name](args.out_dir)

    print("\nAll requested datasets downloaded.")


if __name__ == "__main__":
    main()
