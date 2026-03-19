"""
Instrument Tracker — Full Training Data Pipeline
===================================================
One command to go from SurgicalInstruments catalog → ready-to-train
YOLOv8 dataset with ground truth labels and balanced splits.

What this script does:
  1. COLLECT  — Pull instrument taxonomy + exemplar images from catalog app
  2. ORGANIZE — Build class directory structure from exemplar images
  3. LABEL    — Generate YOLO detection labels (ground truth from catalog images)
  4. SPLIT    — Stratified shuffle ensuring ALL classes appear in train AND test
  5. EXPORT   — Write data.yaml for YOLOv8 training

The result: a ready-to-train YOLO dataset where every instrument class
is represented in both train and test sets.

Usage:
    # Full pipeline (catalog must be running on localhost:3000):
    python run_pipeline.py --api-url http://localhost:3000

    # From cloud catalog:
    python run_pipeline.py --api-url https://your-app.onrender.com

    # Skip catalog sync (use existing data/):
    python run_pipeline.py --skip-sync

    # Custom split ratio:
    python run_pipeline.py --api-url http://localhost:3000 --train-ratio 0.8 --val-ratio 0.1

Output:
    data/yolo_dataset/
      images/train/    — training images
      images/val/      — validation images
      images/test/     — test images
      labels/train/    — YOLO label files
      labels/val/
      labels/test/
      data.yaml        — YOLOv8 config (point train_detector.py here)

Then train with:
    python train/train_detector.py --data-yaml data/yolo_dataset/data.yaml
"""

import argparse
import base64
import json
import os
import random
import shutil
import sys
import urllib.request
import urllib.error
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

# Shared library
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.config import load_config

# -----------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------

PIPELINE_DIR = Path(__file__).resolve().parent
DATA_DIR = PIPELINE_DIR / "data"
CONFIG_PATH = PIPELINE_DIR / "config" / "config.yaml"


# -----------------------------------------------------------------------
# Step 1: COLLECT — pull from SurgicalInstruments API
# -----------------------------------------------------------------------

def api_get(base_url: str, endpoint: str):
    url = f"{base_url.rstrip('/')}{endpoint}"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.URLError as e:
        print(f"  ERROR: Cannot reach {url}")
        print(f"  Is the SurgicalInstruments app running?")
        raise SystemExit(1) from e


def _normalize_instrument_name(name: str, family: str) -> str:
    """Normalize instrument name for deduplication across kits.

    PROBLEM: The same instrument appears with different names across kits:
      "Richardson Retractor"
      "Retractor, Richardson, Loop Handle, 38mm x 38mm"

    RULES:
      - Same instrument, different name → MERGE (one training class)
      - Same instrument, different COLOR → MERGE (visual variant, same class)
      - Same instrument, different SIZE → KEEP SEPARATE (different class)
      - Same instrument, curved vs straight → KEEP SEPARATE (different class)

    HOW:
      1. Lowercase, remove punctuation
      2. KEEP size/dimension info (14cm, 38mm, large, small, etc.)
      3. KEEP shape descriptors (curved, straight, angled, etc.)
      4. REMOVE color words (blue, gold, black, titanium-colored, etc.)
      5. REMOVE filler words (the, a, with, for, etc.)
      6. Sort remaining tokens alphabetically for order-independent match
      7. Prefix with family
    """
    import re

    n = name.lower().strip()

    # Remove color words (these are just visual variants, same class)
    colors = {"blue", "gold", "golden", "black", "silver", "green", "red",
              "white", "yellow", "purple", "titanium", "ebony", "ivory",
              "colored", "coated", "insulated"}
    n = re.sub(r'[,\-\(\)\.\#\/]', ' ', n)
    tokens = n.split()
    tokens = [t for t in tokens if t not in colors]

    # Remove common filler words
    fillers = {"the", "a", "an", "with", "for", "and", "or", "type",
               "style", "model", "series", "set", "ea", "each", "pc", "pcs"}
    tokens = [t for t in tokens if t and t not in fillers]

    # Remove family name from tokens (already captured separately)
    family_tokens = set(family.lower().split())
    tokens = [t for t in tokens if t not in family_tokens]

    # Sort tokens for order-independent matching
    canonical = " ".join(sorted(set(tokens)))

    # Prefix with family for uniqueness
    family_key = family.lower().replace(" ", "_")
    return f"{family_key}__{canonical}" if canonical else family_key



    """Pull all instruments + images from the catalog API."""
    print("=" * 60)
    print("STEP 1: COLLECT — Pulling from SurgicalInstruments catalog")
    print("=" * 60)

    instruments = api_get(api_url, "/api/instruments")
    kits = api_get(api_url, "/api/kit-catalogs")
    stats = api_get(api_url, "/api/stats")

    print(f"  Found: {len(instruments)} instruments, "
          f"{stats.get('images', '?')} images, "
          f"{stats.get('families', '?')} families")

    # ---------------------------------------------------------------
    # CHALLENGE 1: Same instrument, different names across kits
    # ---------------------------------------------------------------
    # The catalog may have "Richardson Retractor" in one kit and
    # "Retractor, Richardson, Loop Handle, 38mm x 38mm" in another.
    # We normalize names to group them into a single training class.
    #
    # CHALLENGE 2: Same instrument type, different visual variants
    # (colors, sizes, coatings). These are KEPT as one class because
    # the detector should recognize all variants of the same tool.
    # The visual diversity actually improves training robustness.
    # ---------------------------------------------------------------

    # Step A: Normalize instrument names for deduplication
    raw_instruments = []
    for inst in instruments:
        name = inst.get("name", "Unknown")
        family = inst.get("family", "Unknown")
        image_count = inst.get("imageCount", inst.get("image_count", 0))
        video_count = inst.get("videoCount", inst.get("video_count", 0))
        inst_id = inst.get("id")

        if image_count == 0 and video_count == 0:
            continue

        canonical = _normalize_instrument_name(name, family)
        raw_instruments.append({
            "name": name,
            "family": family,
            "canonical": canonical,
            "inst_id": inst_id,
            "image_count": image_count,
            "video_count": video_count,
            "kit_names": inst.get("kitNames", []),
        })

    # Step B: Group duplicates into single classes
    # All instruments with the same canonical name → one class
    canonical_groups = defaultdict(list)
    for inst in raw_instruments:
        canonical_groups[inst["canonical"]].append(inst)

    classes = {}
    class_id = 0
    dedup_log = []

    for canonical, group in sorted(canonical_groups.items()):
        # Merge all variants into one class
        all_names = list(set(g["name"] for g in group))
        all_families = list(set(g["family"] for g in group))
        all_kits = list(set(k for g in group for k in g["kit_names"]))
        total_images = sum(g["image_count"] for g in group)
        total_videos = sum(g["video_count"] for g in group)

        # Use the shortest name as display name
        display_name = min(all_names, key=len)
        display_family = all_families[0]

        class_key = f"{display_family}__{canonical}"
        classes[class_key] = {
            "class_id": class_id,
            "name": display_name,
            "family": display_family,
            "canonical": canonical,
            "all_names": all_names,
            "all_families": all_families,
            "inst_ids": [g["inst_id"] for g in group],
            "inst_id": group[0]["inst_id"],  # primary
            "image_count": total_images,
            "video_count": total_videos,
            "kit_names": all_kits,
        }

        if len(all_names) > 1:
            dedup_log.append({
                "canonical": canonical,
                "merged_names": all_names,
                "families": all_families,
            })

        class_id += 1

    if dedup_log:
        print(f"\n  DEDUPLICATION: Merged {len(dedup_log)} groups of duplicate names:")
        for entry in dedup_log[:10]:
            print(f"    '{entry['canonical']}' ← {entry['merged_names']}")
        if len(dedup_log) > 10:
            print(f"    ... and {len(dedup_log) - 10} more")
        dedup_path = output_dir / "dedup_log.json"
        with open(dedup_path, "w") as f:
            json.dump(dedup_log, f, indent=2)
        print(f"    Full log: {dedup_path}")

    print(f"\n  Classes (after dedup): {len(classes)}")
    total_media = sum(c["image_count"] + c["video_count"] for c in classes.values())
    print(f"  Total media to process: {total_media} (images + videos)")

    # Save class map
    class_map_path = output_dir / "class_map.json"
    class_map_export = {
        k: {"class_id": v["class_id"], "name": v["name"], "family": v["family"],
             "canonical": v["canonical"], "all_names": v.get("all_names", [v["name"]]),
             "image_count": v["image_count"], "video_count": v["video_count"],
             "kit_names": v["kit_names"]}
        for k, v in classes.items()
    }
    with open(class_map_path, "w") as f:
        json.dump(class_map_export, f, indent=2)

    # Save kit data
    kit_map = {}
    for kit in kits:
        kit_data = kit.get("data", "{}")
        if isinstance(kit_data, str):
            try:
                kit_data = json.loads(kit_data)
            except json.JSONDecodeError:
                kit_data = {}
        kit_map[kit.get("name", "Unknown")] = kit_data
    with open(output_dir / "kit_definitions.json", "w") as f:
        json.dump(kit_map, f, indent=2)

    return classes


# -----------------------------------------------------------------------
# Step 2: ORGANIZE — download images into class folders
# -----------------------------------------------------------------------

def _download_file(api_url: str, filename: str, output_path: str) -> bool:
    """Download a file from the SurgicalInstruments API."""
    url = f"{api_url.rstrip('/')}/api/images/{filename}/data"
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            data = resp.read()
        # Handle base64 JSON response
        try:
            json_data = json.loads(data)
            if "image_data" in json_data:
                b64 = json_data["image_data"]
                if b64.startswith("data:"):
                    b64 = b64.split(",", 1)[1]
                data = base64.b64decode(b64)
        except (json.JSONDecodeError, ValueError):
            pass
        with open(output_path, "wb") as f:
            f.write(data)
        return True
    except Exception as e:
        print(f"    WARNING: Failed {filename}: {e}")
        return False


def _extract_video_frames(video_path: str, output_dir: Path, prefix: str,
                           max_frames: int = 10, jitter: bool = True) -> list:
    """Extract diverse frames from a video file.

    Extracts frames at evenly-spaced intervals with random jitter
    to avoid near-duplicate frames and maximize visual diversity.

    Args:
        max_frames: how many frames to extract per video
        jitter: add random offset to frame positions (±10% of interval)
    """
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 2:
        cap.release()
        return []

    # Calculate evenly-spaced frame positions
    interval = total / (max_frames + 1)
    positions = []
    for i in range(1, max_frames + 1):
        pos = int(i * interval)
        if jitter:
            jitter_range = max(1, int(interval * 0.1))
            pos += random.randint(-jitter_range, jitter_range)
        pos = max(0, min(total - 1, pos))
        positions.append(pos)

    # Remove duplicates and sort
    positions = sorted(set(positions))

    frames_saved = []
    for idx, pos in enumerate(positions):
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame = cap.read()
        if not ret:
            continue
        fname = f"{prefix}_vidframe_{idx:03d}.jpg"
        out_path = output_dir / fname
        cv2.imwrite(str(out_path), frame)
        frames_saved.append({"path": str(out_path), "filename": fname, "frame_idx": pos})

    cap.release()
    return frames_saved


def organize_images(api_url: str, classes: dict, output_dir: Path,
                     video_frames_per_clip: int = 10) -> dict:
    """Download all images AND extract frames from videos, organized by class."""
    print("\n" + "=" * 60)
    print("STEP 2: ORGANIZE — Downloading images + extracting video frames")
    print("=" * 60)

    images_dir = output_dir / "all_images"
    videos_dir = output_dir / "all_videos"
    images_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)

    image_records = []  # [{path, class_id, class_name, family}]

    for class_key, cls_info in classes.items():
        class_id = cls_info["class_id"]
        name_safe = cls_info["name"][:60].replace("/", "-").replace(" ", "_").replace(",", "")
        family_safe = cls_info["family"].replace("/", "-").replace(" ", "_")
        prefix = f"{family_safe}__{name_safe}"

        # Get full instrument details (includes both images AND videos)
        try:
            detail = api_get(api_url, f"/api/instruments/{cls_info['inst_id']}")
        except SystemExit:
            continue

        all_media = detail.get("images", [])

        still_count = 0
        video_count = 0

        for i, media in enumerate(all_media):
            filename = media.get("filename", "")
            mime = media.get("mime_type", "")

            if "video" in mime:
                # --- VIDEO: download then extract frames ---
                vid_path = videos_dir / f"{prefix}_{i:03d}.mp4"
                if not vid_path.exists():
                    _download_file(api_url, filename, str(vid_path))

                if vid_path.exists():
                    extracted = _extract_video_frames(
                        str(vid_path), images_dir, f"{prefix}_v{i}",
                        max_frames=video_frames_per_clip, jitter=True,
                    )
                    for frame_rec in extracted:
                        frame_rec["class_id"] = class_id
                        frame_rec["class_name"] = cls_info["name"]
                        frame_rec["family"] = cls_info["family"]
                        frame_rec["source"] = "video"
                        image_records.append(frame_rec)
                    video_count += len(extracted)
            else:
                # --- STILL IMAGE: download directly ---
                img_name = f"{prefix}_{i:03d}.jpg"
                img_path = images_dir / img_name

                if img_path.exists():
                    image_records.append({
                        "path": str(img_path), "filename": img_name,
                        "class_id": class_id, "class_name": cls_info["name"],
                        "family": cls_info["family"], "source": "image",
                    })
                    still_count += 1
                    continue

                if _download_file(api_url, filename, str(img_path)):
                    image_records.append({
                        "path": str(img_path), "filename": img_name,
                        "class_id": class_id, "class_name": cls_info["name"],
                        "family": cls_info["family"], "source": "image",
                    })
                    still_count += 1

        total = still_count + video_count
        parts = []
        if still_count:
            parts.append(f"{still_count} images")
        if video_count:
            parts.append(f"{video_count} video frames")
        print(f"  [{class_id:3d}] {cls_info['family']}/{cls_info['name'][:40]}: {' + '.join(parts) or '0'}")

    print(f"\n  Total images: {len(image_records)}")
    return image_records


# -----------------------------------------------------------------------
# Step 3: LABEL — generate YOLO ground truth
# -----------------------------------------------------------------------

def generate_labels(image_records: list, output_dir: Path, classes: dict):
    """Generate YOLO detection labels.

    For catalog images (single instrument on white/neutral background),
    the ground truth is the full image bounding box — the instrument
    fills most of the frame.

    For OR video frames (multiple instruments), use SAM3 zero-shot
    annotations instead (see generate_zero_shot.py).
    """
    print("\n" + "=" * 60)
    print("STEP 3: LABEL — Generating YOLO ground truth")
    print("=" * 60)

    labels_dir = output_dir / "all_labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    labeled = 0
    for rec in image_records:
        img_path = Path(rec["path"])
        label_name = img_path.stem + ".txt"
        label_path = labels_dir / label_name

        # Read image to get dimensions
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        # For catalog images: detect the instrument region
        # Convert to grayscale, threshold to find non-white region
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Adaptive threshold to find the instrument (dark object on light bg)
        # Or light object on any background
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Use the largest contour as the instrument
            largest = max(contours, key=cv2.contourArea)
            x, y, bw, bh = cv2.boundingRect(largest)

            # Skip tiny detections (noise)
            if bw * bh < (w * h * 0.01):
                # Fallback: use center 80% of image
                x, y = int(w * 0.1), int(h * 0.1)
                bw, bh = int(w * 0.8), int(h * 0.8)
        else:
            # No contour found — use center 80%
            x, y = int(w * 0.1), int(h * 0.1)
            bw, bh = int(w * 0.8), int(h * 0.8)

        # Convert to YOLO format: class_id cx cy w h (all normalized)
        cx = (x + bw / 2) / w
        cy = (y + bh / 2) / h
        nw = bw / w
        nh = bh / h

        # Clamp to [0, 1]
        cx = max(0, min(1, cx))
        cy = max(0, min(1, cy))
        nw = max(0.01, min(1, nw))
        nh = max(0.01, min(1, nh))

        with open(label_path, "w") as f:
            f.write(f"{rec['class_id']} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

        rec["label_path"] = str(label_path)
        labeled += 1

    print(f"  Generated {labeled} label files")
    return image_records


# -----------------------------------------------------------------------
# Step 4: SPLIT — stratified shuffle, all classes in train AND test
# -----------------------------------------------------------------------

def stratified_split(image_records: list, train_ratio: float = 0.7,
                      val_ratio: float = 0.15, seed: int = 42) -> dict:
    """Split ensuring EVERY class appears in train, val, and test.

    Strategy:
      1. Group images by class
      2. For each class: shuffle, then allocate to train/val/test
      3. Classes with < 3 images: put in train only (not enough to split)
      4. Verify: every class in train set, every class in test set
    """
    print("\n" + "=" * 60)
    print("STEP 4: SPLIT — Balanced stratified shuffle")
    print("=" * 60)

    random.seed(seed)
    test_ratio = 1.0 - train_ratio - val_ratio

    # Group by class
    by_class = defaultdict(list)
    for rec in image_records:
        by_class[rec["class_id"]].append(rec)

    splits = {"train": [], "val": [], "test": []}
    class_distribution = {}

    for class_id, records in sorted(by_class.items()):
        random.shuffle(records)
        n = len(records)
        class_name = records[0]["class_name"]

        if n < 3:
            # Not enough to split — put all in train
            splits["train"].extend(records)
            class_distribution[class_name] = {"train": n, "val": 0, "test": 0, "total": n}
            print(f"  [{class_id:3d}] {class_name[:40]}: {n} images (train only — too few to split)")
            continue

        # Ensure at least 1 in each split
        n_val = max(1, int(n * val_ratio))
        n_test = max(1, int(n * test_ratio))
        n_train = n - n_val - n_test

        if n_train < 1:
            n_train = 1
            remaining = n - 1
            n_val = remaining // 2
            n_test = remaining - n_val

        splits["train"].extend(records[:n_train])
        splits["val"].extend(records[n_train:n_train + n_val])
        splits["test"].extend(records[n_train + n_val:])

        class_distribution[class_name] = {
            "train": n_train, "val": n_val, "test": n_test, "total": n,
        }

    # Shuffle within each split
    for split_name in splits:
        random.shuffle(splits[split_name])

    # Verify
    train_classes = set(r["class_id"] for r in splits["train"])
    val_classes = set(r["class_id"] for r in splits["val"])
    test_classes = set(r["class_id"] for r in splits["test"])
    all_classes = set(r["class_id"] for r in image_records)

    print(f"\n  Split results:")
    print(f"    Train: {len(splits['train'])} images, {len(train_classes)} classes")
    print(f"    Val:   {len(splits['val'])} images, {len(val_classes)} classes")
    print(f"    Test:  {len(splits['test'])} images, {len(test_classes)} classes")

    missing_in_test = all_classes - test_classes
    if missing_in_test:
        print(f"\n  WARNING: {len(missing_in_test)} classes not in test set (too few images)")
    else:
        print(f"\n  ALL {len(all_classes)} classes present in both train and test sets")

    return splits, class_distribution


# -----------------------------------------------------------------------
# Step 5: EXPORT — write YOLO dataset
# -----------------------------------------------------------------------

def export_yolo_dataset(splits: dict, classes: dict, output_dir: Path):
    """Copy images + labels into YOLO directory structure and write data.yaml."""
    print("\n" + "=" * 60)
    print("STEP 5: EXPORT — Writing YOLO dataset")
    print("=" * 60)

    yolo_dir = output_dir / "yolo_dataset"

    for split_name, records in splits.items():
        img_dir = yolo_dir / "images" / split_name
        lbl_dir = yolo_dir / "labels" / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for rec in records:
            src_img = Path(rec["path"])
            src_lbl = Path(rec.get("label_path", ""))

            dst_img = img_dir / src_img.name
            dst_lbl = lbl_dir / (src_img.stem + ".txt")

            if src_img.exists() and not dst_img.exists():
                shutil.copy2(src_img, dst_img)
            if src_lbl.exists() and not dst_lbl.exists():
                shutil.copy2(src_lbl, dst_lbl)

    # Build class name list ordered by class_id
    class_names = {}
    for cls_info in classes.values():
        cid = cls_info["class_id"]
        class_names[cid] = f"{cls_info['family']}/{cls_info['name']}"

    max_id = max(class_names.keys()) if class_names else 0
    names_list = [class_names.get(i, f"class_{i}") for i in range(max_id + 1)]

    # Write data.yaml
    data_yaml = {
        "path": str(yolo_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(names_list),
        "names": names_list,
    }

    import yaml
    yaml_path = yolo_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)

    print(f"  Dataset:   {yolo_dir}")
    print(f"  data.yaml: {yaml_path}")
    print(f"  Classes:   {len(names_list)}")
    print(f"  Train:     {len(splits['train'])} images")
    print(f"  Val:       {len(splits['val'])} images")
    print(f"  Test:      {len(splits['test'])} images")

    return yaml_path


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Full pipeline: catalog → training data with ground truth",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This is the ONE COMMAND that builds your entire training dataset.

Examples:
  python run_pipeline.py --api-url http://localhost:3000
  python run_pipeline.py --api-url http://localhost:3000 --train-ratio 0.8
  python run_pipeline.py --skip-sync   # use existing data/
        """,
    )
    parser.add_argument("--api-url", default="http://localhost:3000",
                        help="SurgicalInstruments app URL")
    parser.add_argument("--output-dir", default=str(DATA_DIR),
                        help="Output directory (default: data/)")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--video-frames", type=int, default=10,
                        help="Frames to extract per video clip (with jitter)")
    parser.add_argument("--skip-sync", action="store_true",
                        help="Skip catalog sync, use existing data/all_images/")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("INSTRUMENT TRACKER — FULL TRAINING DATA PIPELINE")
    print("=" * 60)
    print(f"  Source:    {args.api_url}")
    print(f"  Output:    {output_dir}")
    print(f"  Split:     {args.train_ratio:.0%} train / {args.val_ratio:.0%} val / "
          f"{1 - args.train_ratio - args.val_ratio:.0%} test")
    print()

    # Step 1: Collect
    if args.skip_sync:
        print("Skipping catalog sync (--skip-sync)")
        class_map_path = output_dir / "class_map.json"
        if not class_map_path.exists():
            print("ERROR: No class_map.json found. Run without --skip-sync first.")
            return
        with open(class_map_path) as f:
            class_map = json.load(f)
        # Reconstruct classes dict with empty images (will use existing files)
        classes = {k: {**v, "images": []} for k, v in class_map.items()}
    else:
        classes = collect_from_catalog(args.api_url, output_dir)

    # Step 2: Organize (download images)
    if not args.skip_sync:
        image_records = organize_images(args.api_url, classes, output_dir,
                                            video_frames_per_clip=args.video_frames)
    else:
        # Rebuild from existing files
        images_dir = output_dir / "all_images"
        class_map_path = output_dir / "class_map.json"
        with open(class_map_path) as f:
            class_map = json.load(f)

        image_records = []
        for img_path in sorted(images_dir.glob("*.jpg")):
            # Parse filename: family__name_NNN.jpg
            stem = img_path.stem
            for ck, ci in class_map.items():
                family_safe = ci["family"].replace("/", "-").replace(" ", "_")
                name_safe = ci["name"][:60].replace("/", "-").replace(" ", "_").replace(",", "")
                prefix = f"{family_safe}__{name_safe}"
                if stem.startswith(prefix):
                    image_records.append({
                        "path": str(img_path),
                        "filename": img_path.name,
                        "class_id": ci["class_id"],
                        "class_name": ci["name"],
                        "family": ci["family"],
                    })
                    break

        print(f"  Found {len(image_records)} existing images")

    if not image_records:
        print("\nERROR: No images found. Check the SurgicalInstruments app has images uploaded.")
        return

    # Step 3: Label
    image_records = generate_labels(image_records, output_dir, classes)

    # Step 4: Split
    splits, distribution = stratified_split(
        image_records, args.train_ratio, args.val_ratio, args.seed
    )

    # Save distribution report
    dist_path = output_dir / "class_distribution.json"
    with open(dist_path, "w") as f:
        json.dump(distribution, f, indent=2)

    # Step 5: Export
    yaml_path = export_yolo_dataset(splits, classes, output_dir)

    # Final summary
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\n  Dataset ready at: {output_dir / 'yolo_dataset'}")
    print(f"  data.yaml:        {yaml_path}")
    print(f"  Class distribution: {dist_path}")
    print()
    print("  Next step — train the detector:")
    print(f"    python train/train_detector.py --data-yaml {yaml_path}")
    print()
    print("  Or train with custom settings:")
    print(f"    python train/train_detector.py --data-yaml {yaml_path} "
          f"--epochs 50 --imgsz 1024 --batch-size 4")


if __name__ == "__main__":
    main()
