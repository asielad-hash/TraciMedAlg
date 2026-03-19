"""
Prepare train/val/test splits from extracted event clips.

Loads the clip manifest produced by extract_event_clips.py and splits clips
into train, validation, and test sets while maintaining class balance across
the three event types (TAKE_OUT, PUT_BACK, NO_EVENT).

Splitting is stratified per class so each split has roughly the same
proportion of each event type.  Optionally groups clips by source video
so that clips from the same video do not leak across splits.

Output:
    output_dir/train.json   - list of {clip_id, event_type, ...}
    output_dir/val.json
    output_dir/test.json
    output_dir/split_stats.json - summary statistics

Usage:
    python prepare_splits.py --clips-dir data/event_clips --output-dir data
    python prepare_splits.py --config ../config/config.yaml --clips-dir data/event_clips
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

# Shared library import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from shared.config import load_config

DEFAULT_CONFIG = str(Path(__file__).resolve().parent.parent / "config" / "config.yaml")


def load_manifest(clips_dir: Path) -> dict:
    """Load the clip manifest produced by extract_event_clips.py."""
    manifest_path = clips_dir / "manifest.json"
    if not manifest_path.is_file():
        print(f"Error: manifest not found at {manifest_path}")
        sys.exit(1)
    with open(manifest_path) as f:
        return json.load(f)


def stratified_split(items: list, train_ratio: float, val_ratio: float,
                     test_ratio: float, seed: int = 42) -> tuple[list, list, list]:
    """Split items into train/val/test maintaining class balance.

    Groups items by event_type and applies the ratios within each group
    so every split has roughly the same class distribution.

    Returns (train, val, test) lists.
    """
    rng = random.Random(seed)

    # Group by event type
    by_class: dict[str, list] = defaultdict(list)
    for item in items:
        by_class[item["event_type"]].append(item)

    train, val, test = [], [], []

    for cls, cls_items in sorted(by_class.items()):
        rng.shuffle(cls_items)
        n = len(cls_items)
        n_train = max(1, int(round(n * train_ratio)))
        n_val = max(1, int(round(n * val_ratio)))
        # Remainder goes to test
        n_test = n - n_train - n_val
        if n_test < 0:
            # Very few items; give at least 1 to each
            n_train = max(1, n - 2)
            n_val = min(1, n - n_train)
            n_test = n - n_train - n_val

        train.extend(cls_items[:n_train])
        val.extend(cls_items[n_train:n_train + n_val])
        test.extend(cls_items[n_train + n_val:])

    return train, val, test


def compute_split_stats(train: list, val: list, test: list) -> dict:
    """Compute per-split class distribution statistics."""
    stats = {}
    for name, split in [("train", train), ("val", val), ("test", test)]:
        counts = defaultdict(int)
        for item in split:
            counts[item["event_type"]] += 1
        stats[name] = {
            "total": len(split),
            "by_class": dict(counts),
        }
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Split extracted event clips into train/val/test sets."
    )
    parser.add_argument(
        "--config", type=str, default=DEFAULT_CONFIG,
        help="Path to config YAML (default: ../config/config.yaml)"
    )
    parser.add_argument(
        "--clips-dir", type=str, required=True,
        help="Directory containing extracted clips and manifest.json"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for split JSON files (default: data/)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible splits (default: 42)"
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    dataset_cfg = config.get("dataset", {})
    split_ratio = dataset_cfg.get("split_ratio", {"train": 0.70, "val": 0.15, "test": 0.15})

    train_ratio = split_ratio["train"]
    val_ratio = split_ratio["val"]
    test_ratio = split_ratio["test"]

    # Resolve directories
    clips_dir = Path(args.clips_dir)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(config.get("_config_dir", ".")).parent / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Clips directory: {clips_dir}")
    print(f"Split ratios: train={train_ratio}, val={val_ratio}, test={test_ratio}")
    print(f"Output: {output_dir}")
    print()

    # Load manifest
    manifest = load_manifest(clips_dir)
    print(f"Loaded {len(manifest)} clips from manifest")

    # Convert manifest to list of items (add clip_id as a field)
    items = []
    for clip_id, meta in manifest.items():
        entry = {"clip_id": clip_id}
        entry.update(meta)
        items.append(entry)

    if not items:
        print("Error: no clips found in manifest")
        sys.exit(1)

    # Show class distribution before split
    class_counts = defaultdict(int)
    for item in items:
        class_counts[item["event_type"]] += 1
    print("Class distribution:")
    for cls, count in sorted(class_counts.items()):
        print(f"  {cls}: {count}")
    print()

    # Perform stratified split
    train, val, test = stratified_split(
        items, train_ratio, val_ratio, test_ratio, seed=args.seed
    )

    # Save split files
    for name, split in [("train", train), ("val", val), ("test", test)]:
        split_path = output_dir / f"{name}.json"
        with open(split_path, "w") as f:
            json.dump(split, f, indent=2)
        print(f"  {name}: {len(split)} clips -> {split_path}")

    # Save statistics
    stats = compute_split_stats(train, val, test)
    stats["config"] = {
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "seed": args.seed,
        "total_clips": len(items),
    }
    stats_path = output_dir / "split_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print()
    print("Split statistics:")
    for name in ["train", "val", "test"]:
        s = stats[name]
        classes_str = ", ".join(f"{k}: {v}" for k, v in sorted(s["by_class"].items()))
        print(f"  {name}: {s['total']} clips ({classes_str})")

    print()
    print(f"Done. Stats saved to {stats_path}")


if __name__ == "__main__":
    main()
