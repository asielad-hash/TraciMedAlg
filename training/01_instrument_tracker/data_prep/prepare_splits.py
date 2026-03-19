"""
Stratified train/val/test split for the instrument tracker dataset.

Parses the video naming convention {site}_{specialty}_{case_id} to
group images by video, then stratifies splits by site + specialty
according to config ratios. Optionally generates site-holdout
cross-validation folds.

Outputs: train.json, val.json, test.json, split_metadata.json

Usage:
    python prepare_splits.py --config ../config/config.yaml --coco-dir ./data/coco --output-dir ./data/splits
    python prepare_splits.py --config ../config/config.yaml --coco-dir ./data/coco --output-dir ./data/splits --seed 42
"""

import argparse
import collections
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from shared.config import load_config


def parse_video_stem(stem: str) -> dict:
    """Parse video naming convention {site}_{specialty}_{case_id}.

    Falls back gracefully if the convention is not followed.
    """
    parts = stem.split("_")
    if len(parts) >= 3:
        return {
            "site": parts[0],
            "specialty": parts[1],
            "case_id": "_".join(parts[2:]),
        }
    elif len(parts) == 2:
        return {"site": parts[0], "specialty": parts[1], "case_id": "unknown"}
    else:
        return {"site": "unknown", "specialty": "unknown", "case_id": stem}


def group_images_by_video(coco: dict) -> dict[str, list[dict]]:
    """Group COCO images by their source video stem."""
    groups = collections.defaultdict(list)
    for img in coco["images"]:
        source = img.get("source_video", "")
        if source:
            stem = Path(source).stem
        else:
            # Derive from file_name: {stem}_frame{idx}.jpg
            fname = img["file_name"]
            stem = fname.rsplit("_frame", 1)[0] if "_frame" in fname else fname
        groups[stem].append(img)
    return dict(groups)


def group_annotations_by_image(coco: dict) -> dict[int, list[dict]]:
    """Map image_id -> list of annotations."""
    mapping = collections.defaultdict(list)
    for ann in coco["annotations"]:
        mapping[ann["image_id"]].append(ann)
    return dict(mapping)


def stratified_split(video_groups: dict[str, list], split_ratio: dict,
                     stratify_by: list[str], seed: int) -> dict[str, list[str]]:
    """Split video stems into train/val/test, stratified by metadata fields.

    Returns: {"train": [stems], "val": [stems], "test": [stems]}
    """
    rng = random.Random(seed)

    # Group videos by stratification key
    strata = collections.defaultdict(list)
    for stem in video_groups:
        parsed = parse_video_stem(stem)
        key = tuple(parsed.get(k, "unknown") for k in stratify_by)
        strata[key].append(stem)

    splits = {"train": [], "val": [], "test": []}
    train_r = split_ratio.get("train", 0.7)
    val_r = split_ratio.get("val", 0.15)
    # test gets the remainder

    for key, stems in sorted(strata.items()):
        rng.shuffle(stems)
        n = len(stems)
        n_train = max(1, round(n * train_r)) if n >= 3 else n
        n_val = max(1, round(n * val_r)) if n >= 3 else 0
        n_test = n - n_train - n_val

        # Ensure at least one in each split when possible
        if n >= 3 and n_test <= 0:
            n_test = 1
            n_train = n - n_val - n_test

        splits["train"].extend(stems[:n_train])
        splits["val"].extend(stems[n_train:n_train + n_val])
        splits["test"].extend(stems[n_train + n_val:])

    return splits


def site_holdout_cv(video_groups: dict[str, list]) -> list[dict]:
    """Generate leave-one-site-out cross-validation folds.

    Returns list of {"fold": i, "holdout_site": site,
                     "train": [stems], "test": [stems]}
    """
    site_to_stems = collections.defaultdict(list)
    for stem in video_groups:
        parsed = parse_video_stem(stem)
        site_to_stems[parsed["site"]].append(stem)

    folds = []
    for i, (site, test_stems) in enumerate(sorted(site_to_stems.items())):
        train_stems = [s for s in video_groups if s not in test_stems]
        folds.append({
            "fold": i,
            "holdout_site": site,
            "train": sorted(train_stems),
            "test": sorted(test_stems),
        })
    return folds


def build_split_coco(coco: dict, stems: list[str],
                     video_groups: dict, ann_by_image: dict) -> dict:
    """Build a COCO subset containing only images from the given video stems."""
    img_ids = set()
    for stem in stems:
        for img in video_groups.get(stem, []):
            img_ids.add(img["id"])

    images = [img for img in coco["images"] if img["id"] in img_ids]
    annotations = []
    for img_id in img_ids:
        annotations.extend(ann_by_image.get(img_id, []))

    return {
        "images": images,
        "annotations": annotations,
        "categories": coco["categories"],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Stratified train/val/test split for instrument tracker."
    )
    parser.add_argument(
        "--config", type=str,
        default=str(Path(__file__).resolve().parent.parent / "config" / "config.yaml"),
        help="Path to config YAML (default: ../config/config.yaml).",
    )
    parser.add_argument(
        "--coco-dir", type=str, required=True,
        help="Directory containing the COCO annotations.json file.",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Output directory for split JSON files.",
    )
    parser.add_argument(
        "--coco-name", type=str, default="annotations.json",
        help="Name of the COCO JSON file (default: annotations.json).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_cfg = config.get("dataset", {})
    split_ratio = dataset_cfg.get("split_ratio", {"train": 0.7, "val": 0.15, "test": 0.15})
    stratify_by = dataset_cfg.get("stratify_by", ["site", "specialty"])
    do_site_holdout = dataset_cfg.get("site_holdout", False)

    coco_path = Path(args.coco_dir) / args.coco_name
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not coco_path.exists():
        print(f"Error: COCO file not found: {coco_path}")
        sys.exit(1)

    print(f"Loading {coco_path} ...")
    with open(coco_path) as f:
        coco = json.load(f)

    print(f"  {len(coco['images'])} images, {len(coco['annotations'])} annotations, "
          f"{len(coco['categories'])} categories")

    video_groups = group_images_by_video(coco)
    ann_by_image = group_annotations_by_image(coco)
    print(f"  {len(video_groups)} unique videos")

    # Print stratification info
    strata_info = collections.defaultdict(list)
    for stem in video_groups:
        parsed = parse_video_stem(stem)
        key = tuple(parsed.get(k, "unknown") for k in stratify_by)
        strata_info[key].append(stem)
    print(f"\nStratification groups ({', '.join(stratify_by)}):")
    for key, stems in sorted(strata_info.items()):
        print(f"  {key}: {len(stems)} videos")

    # Main stratified split
    print(f"\nSplit ratios: train={split_ratio['train']}, "
          f"val={split_ratio['val']}, test={split_ratio['test']}")
    splits = stratified_split(video_groups, split_ratio, stratify_by, args.seed)

    metadata = {
        "seed": args.seed,
        "stratify_by": stratify_by,
        "split_ratio": split_ratio,
        "splits": {},
    }

    for split_name in ["train", "val", "test"]:
        stems = splits[split_name]
        split_coco = build_split_coco(coco, stems, video_groups, ann_by_image)
        out_path = output_dir / f"{split_name}.json"
        with open(out_path, "w") as f:
            json.dump(split_coco, f, indent=2)

        n_imgs = len(split_coco["images"])
        n_anns = len(split_coco["annotations"])
        print(f"  {split_name}: {len(stems)} videos, {n_imgs} images, {n_anns} annotations -> {out_path}")

        metadata["splits"][split_name] = {
            "videos": sorted(stems),
            "num_videos": len(stems),
            "num_images": n_imgs,
            "num_annotations": n_anns,
        }

    # Site-holdout cross-validation
    if do_site_holdout:
        print(f"\nSite-holdout cross-validation:")
        folds = site_holdout_cv(video_groups)
        metadata["site_holdout_folds"] = []

        cv_dir = output_dir / "site_holdout_cv"
        cv_dir.mkdir(parents=True, exist_ok=True)

        for fold in folds:
            fold_dir = cv_dir / f"fold_{fold['fold']}_{fold['holdout_site']}"
            fold_dir.mkdir(parents=True, exist_ok=True)

            train_coco = build_split_coco(coco, fold["train"], video_groups, ann_by_image)
            test_coco = build_split_coco(coco, fold["test"], video_groups, ann_by_image)

            with open(fold_dir / "train.json", "w") as f:
                json.dump(train_coco, f, indent=2)
            with open(fold_dir / "test.json", "w") as f:
                json.dump(test_coco, f, indent=2)

            print(f"  Fold {fold['fold']} (holdout={fold['holdout_site']}): "
                  f"train={len(fold['train'])} videos, test={len(fold['test'])} videos "
                  f"-> {fold_dir}")

            metadata["site_holdout_folds"].append({
                "fold": fold["fold"],
                "holdout_site": fold["holdout_site"],
                "train_videos": fold["train"],
                "test_videos": fold["test"],
            })

    # Write split metadata
    meta_path = output_dir / "split_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nSplit metadata: {meta_path}")
    print("Done.")


if __name__ == "__main__":
    main()
