"""
Prepare train/val/test splits for surgical workflow data.

Splits pre-extracted clip or window datasets into train, val, and test
partitions, stratified by site + specialty to prevent data leakage and
ensure each split reflects the full distribution of recording conditions.

Works with both phase-clip .npz files and idle-window .npz files.

Strategy:
    1.  Read metadata from each .npz file (video stem, site, specialty).
    2.  Group files by video stem (all clips from one video go to the
        same split to avoid temporal leakage).
    3.  Stratified split by site+specialty using the configured ratios.
    4.  Write split manifests as JSON for the training scripts.

Usage:
    python prepare_splits.py --config ../config/config.yaml \
        --data-dir ../data/clips/phase --output-dir ../data/splits/phase
    python prepare_splits.py --config ../config/config.yaml \
        --data-dir ../data/clips/idle --output-dir ../data/splits/idle
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# Shared library import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from shared.config import load_config

DEFAULT_CONFIG = str(Path(__file__).resolve().parent.parent / "config" / "config.yaml")


def discover_npz_files(data_dir: Path) -> list[Path]:
    """Find all .npz data files in a directory (non-recursive)."""
    return sorted(p for p in data_dir.iterdir()
                  if p.suffix == ".npz" and p.is_file())


def extract_video_stem(npz_path: Path) -> str:
    """Infer the source video stem from an .npz filename.

    Handles patterns like:
        case001_clip0001234.npz  ->  case001
        case001_idle_windows.npz ->  case001
    """
    name = npz_path.stem
    # Try common suffixes
    for suffix in ("_idle_windows",):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    # Phase clip pattern: {stem}_clip{digits}
    if "_clip" in name:
        return name[: name.rfind("_clip")]
    return name


def load_site_specialty_map(data_dir: Path) -> dict:
    """Attempt to load a site/specialty mapping file.

    Looks for a metadata JSON produced by the extraction scripts or
    a user-provided mapping at data_dir/video_metadata.json:
        { "case001": {"site": "hospital_a", "specialty": "general"}, ... }

    Falls back to reading 'site' field from individual .npz files.
    """
    meta_path = data_dir / "video_metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            return json.load(f)
    return {}


def build_video_groups(npz_files: list[Path], metadata: dict,
                       data_dir: Path) -> dict:
    """Group .npz files by video stem and attach stratification keys.

    Returns:
        { video_stem: { "files": [paths], "site": str, "specialty": str } }
    """
    groups = defaultdict(lambda: {"files": [], "site": "unknown",
                                  "specialty": "unknown"})
    for p in npz_files:
        stem = extract_video_stem(p)
        groups[stem]["files"].append(str(p.relative_to(data_dir)))

        # Try metadata lookup first
        if stem in metadata:
            groups[stem]["site"] = metadata[stem].get("site", "unknown")
            groups[stem]["specialty"] = metadata[stem].get("specialty", "unknown")
        else:
            # Fall back: read site from .npz if present
            try:
                with np.load(str(p), allow_pickle=True) as data:
                    if "site" in data:
                        groups[stem]["site"] = str(data["site"])
            except Exception:
                pass

    return dict(groups)


def stratified_split(groups: dict, ratios: dict, seed: int = 42) -> dict:
    """Split video stems into train/val/test stratified by site+specialty.

    Args:
        groups: output of build_video_groups.
        ratios: {"train": 0.7, "val": 0.15, "test": 0.15}
        seed: random seed for reproducibility.

    Returns:
        {"train": [stem, ...], "val": [stem, ...], "test": [stem, ...]}
    """
    rng = random.Random(seed)

    # Group stems by stratum
    strata = defaultdict(list)
    for stem, info in groups.items():
        key = f"{info['site']}_{info['specialty']}"
        strata[key].append(stem)

    splits = {"train": [], "val": [], "test": []}
    for stratum_key in sorted(strata):
        stems = sorted(strata[stratum_key])
        rng.shuffle(stems)
        n = len(stems)
        n_train = max(1, round(n * ratios["train"]))
        n_val = max(0, round(n * ratios["val"]))
        # Ensure at least 1 in train; remainder goes to test
        n_test = n - n_train - n_val
        if n_test < 0:
            n_val = n - n_train
            n_test = 0

        splits["train"].extend(stems[:n_train])
        splits["val"].extend(stems[n_train:n_train + n_val])
        splits["test"].extend(stems[n_train + n_val:])

    return splits


def write_split_manifests(splits: dict, groups: dict,
                          output_dir: Path) -> None:
    """Write per-split JSON manifests listing all .npz files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {}

    for split_name, stems in splits.items():
        files = []
        for stem in sorted(stems):
            files.extend(groups[stem]["files"])
        manifest = {
            "split": split_name,
            "n_videos": len(stems),
            "n_files": len(files),
            "video_stems": sorted(stems),
            "files": sorted(files),
        }
        path = output_dir / f"{split_name}.json"
        with open(path, "w") as f:
            json.dump(manifest, f, indent=2)
        summary[split_name] = {"videos": len(stems), "files": len(files)}
        print(f"  {split_name}: {len(stems)} videos, {len(files)} files -> {path}")

    # Combined summary
    with open(output_dir / "split_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare train/val/test splits stratified by site+specialty."
    )
    parser.add_argument(
        "--config", type=str, default=DEFAULT_CONFIG,
        help="Path to config YAML (default: ../config/config.yaml)"
    )
    parser.add_argument(
        "--data-dir", type=str, required=True,
        help="Directory containing .npz clip/window files"
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Output directory for split manifest JSON files"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    ratios = config["dataset"]["split_ratio"]

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    if not data_dir.is_dir():
        print(f"Error: data directory not found: {data_dir}")
        sys.exit(1)

    # Discover files
    npz_files = discover_npz_files(data_dir)
    if not npz_files:
        print(f"No .npz files found in {data_dir}")
        sys.exit(1)
    print(f"Found {len(npz_files)} .npz files in {data_dir}")

    # Build video groups
    metadata = load_site_specialty_map(data_dir)
    groups = build_video_groups(npz_files, metadata, data_dir)
    print(f"Grouped into {len(groups)} videos")

    # Split
    splits = stratified_split(groups, ratios, seed=args.seed)
    print(f"\nSplit ratios: {ratios}")
    write_split_manifests(splits, groups, output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
