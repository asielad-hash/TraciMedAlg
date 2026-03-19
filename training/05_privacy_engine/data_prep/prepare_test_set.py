"""
Organize test frames with ground truth for privacy engine evaluation.

Takes a directory of test images paired with ground truth JSON files and
validates the expected format: each frame (*.jpg / *.png) should have a
corresponding *_gt.json containing face and text/badge bounding boxes.

Ground truth JSON schema:
    {
        "faces": [[x1, y1, x2, y2], ...],
        "badges": [[x1, y1, x2, y2], ...]
    }

Generates summary statistics (total frames, total faces, total badges)
and copies validated pairs into the output directory.

Usage:
    python prepare_test_set.py --input-dir /raw/annotated --output-dir ../data/test_frames
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

# Shared library import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from shared.config import load_config

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

DEFAULT_CONFIG = str(Path(__file__).resolve().parent.parent / "config" / "config.yaml")


def find_image_gt_pairs(input_dir: Path) -> list[tuple[Path, Path]]:
    """Find all image files that have a matching *_gt.json ground truth file.

    For ``frame_001.jpg`` the expected ground truth is ``frame_001_gt.json``.

    Returns:
        List of (image_path, gt_json_path) tuples, sorted by name.
    """
    pairs = []
    for img_path in sorted(input_dir.iterdir()):
        if not img_path.is_file() or img_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        gt_path = img_path.with_name(img_path.stem + "_gt.json")
        if gt_path.is_file():
            pairs.append((img_path, gt_path))
    return pairs


def validate_gt_json(gt_path: Path) -> tuple[dict | None, list[str]]:
    """Validate a single ground truth JSON file.

    Returns:
        (parsed_data, list_of_errors)
    """
    errors: list[str] = []
    try:
        with open(gt_path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        return None, [f"Cannot read {gt_path.name}: {exc}"]

    if not isinstance(data, dict):
        return None, [f"{gt_path.name}: root must be a JSON object"]

    for key in ("faces", "badges"):
        if key not in data:
            # Missing key is acceptable — means zero annotations of that type
            data.setdefault(key, [])
            continue
        if not isinstance(data[key], list):
            errors.append(f"{gt_path.name}: '{key}' must be a list of bboxes")
            continue
        for i, bbox in enumerate(data[key]):
            if not (isinstance(bbox, list) and len(bbox) == 4
                    and all(isinstance(v, (int, float)) for v in bbox)):
                errors.append(
                    f"{gt_path.name}: {key}[{i}] must be [x1, y1, x2, y2] "
                    f"with numeric values, got {bbox!r}"
                )
    return data, errors


def main():
    parser = argparse.ArgumentParser(
        description="Validate and organize test frames + ground truth for "
                    "privacy engine evaluation."
    )
    parser.add_argument(
        "--input-dir", type=str, required=True,
        help="Directory with annotated frames and *_gt.json files"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Where to copy validated test set (default: ../data/test_frames)"
    )
    parser.add_argument(
        "--config", type=str, default=DEFAULT_CONFIG,
        help="Path to config YAML (default: ../config/config.yaml)"
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Exit with error if any validation issue is found"
    )
    args = parser.parse_args()

    # Resolve paths ----------------------------------------------------------
    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        print(f"Error: input directory not found: {input_dir}")
        sys.exit(1)

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(__file__).resolve().parent.parent / "data" / "test_frames"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover pairs ---------------------------------------------------------
    pairs = find_image_gt_pairs(input_dir)
    if not pairs:
        print(f"No image + *_gt.json pairs found in {input_dir}")
        print("Expected layout: frame_001.jpg + frame_001_gt.json")
        sys.exit(1)

    print(f"Found {len(pairs)} image/ground-truth pair(s) in {input_dir}")
    print()

    # Validate and copy ------------------------------------------------------
    total_faces = 0
    total_badges = 0
    valid_count = 0
    all_errors: list[str] = []

    for img_path, gt_path in pairs:
        data, errors = validate_gt_json(gt_path)
        if errors:
            all_errors.extend(errors)
            for e in errors:
                print(f"  WARN: {e}")
            if args.strict:
                continue

        if data is None:
            continue

        n_faces = len(data.get("faces", []))
        n_badges = len(data.get("badges", []))
        total_faces += n_faces
        total_badges += n_badges
        valid_count += 1

        # Copy image and ground truth to output dir
        shutil.copy2(img_path, output_dir / img_path.name)
        shutil.copy2(gt_path, output_dir / gt_path.name)

        print(f"  {img_path.name:<40} faces={n_faces:<4} badges={n_badges}")

    # Summary ----------------------------------------------------------------
    summary = {
        "total_frames": valid_count,
        "total_faces": total_faces,
        "total_badges": total_badges,
        "avg_faces_per_frame": round(total_faces / max(valid_count, 1), 2),
        "avg_badges_per_frame": round(total_badges / max(valid_count, 1), 2),
        "validation_errors": len(all_errors),
    }

    summary_path = output_dir / "test_set_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print("=" * 60)
    print(f"  Valid frames copied : {valid_count}/{len(pairs)}")
    print(f"  Total faces         : {total_faces}")
    print(f"  Total badges        : {total_badges}")
    print(f"  Avg faces/frame     : {summary['avg_faces_per_frame']}")
    print(f"  Avg badges/frame    : {summary['avg_badges_per_frame']}")
    print(f"  Validation errors   : {len(all_errors)}")
    print(f"  Summary written to  : {summary_path}")
    print("=" * 60)

    if args.strict and all_errors:
        print(f"\nStrict mode: {len(all_errors)} error(s) detected, exiting.")
        sys.exit(1)


if __name__ == "__main__":
    main()
