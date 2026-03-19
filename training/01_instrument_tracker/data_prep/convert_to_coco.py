"""
Convert tracked metadata or keyframe annotations to COCO format.

Reads _tracked_meta.json and/or _keyframes.json files from an annotation
directory, resolves video info, and writes a unified COCO JSON dataset
with images, annotations, and categories.

Usage:
    python convert_to_coco.py --anno-dir ./data/tracked --video-dir /path/to/videos --output-dir ./data/coco
    python convert_to_coco.py --anno-dir ./data/annotations --video-dir /videos --output-dir ./data/coco
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from shared.annotation_io import load_keyframes, keyframes_to_coco, load_tracked_meta
from shared.video_io import get_video_info

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def find_video(stem: str, video_dir: Path) -> Path | None:
    """Find the video file matching a stem in video_dir."""
    for ext in VIDEO_EXTENSIONS:
        candidate = video_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def tracked_meta_to_partial(meta: dict) -> dict:
    """Convert one _tracked_meta.json to a partial COCO-like dict for merging."""
    stem = meta["video_stem"]
    images = []
    annotations = []
    cat_names = set()

    for fdata in meta.get("frames", []):
        fidx = fdata["frame"]
        fname = f"{stem}_frame{fidx:06d}.jpg"
        images.append({
            "file_name": fname,
            "width": meta["width"],
            "height": meta["height"],
            "frame_idx": fidx,
            "source_video": meta.get("video", ""),
        })
        for j in range(len(fdata.get("obj_ids", []))):
            label = fdata["labels"][j] if j < len(fdata.get("labels", [])) else f"object_{fdata['obj_ids'][j]}"
            bbox = fdata["bboxes"][j] if j < len(fdata.get("bboxes", [])) else [0, 0, 0, 0]
            area = fdata["areas"][j] if j < len(fdata.get("areas", [])) else 0
            cat_names.add(label)
            annotations.append({
                "file_name": fname,
                "category_name": label,
                "bbox": bbox,
                "area": area,
                "obj_id": fdata["obj_ids"][j],
                "iscrowd": 0,
            })

    return {"images": images, "annotations": annotations, "category_names": cat_names}


def keyframes_to_partial(kf_path: Path, video_dir: Path) -> dict | None:
    """Convert one _keyframes.json to a partial COCO-like dict for merging."""
    stem = kf_path.stem.replace("_keyframes", "")
    video_path = find_video(stem, video_dir)
    if video_path is None:
        print(f"  Warning: no matching video for {kf_path.name}, skipping")
        return None

    video_info = get_video_info(str(video_path))
    keyframes = load_keyframes(str(video_path))
    if not keyframes:
        print(f"  No keyframes found, skipping")
        return None

    coco_partial = keyframes_to_coco(keyframes, video_info)
    cat_id_to_name = {c["id"]: c["name"] for c in coco_partial.get("categories", [])}
    img_id_to_fname = {img["id"]: img for img in coco_partial["images"]}

    images = []
    for img in coco_partial["images"]:
        images.append({
            "file_name": img["file_name"],
            "width": img["width"],
            "height": img["height"],
            "frame_idx": img.get("frame_idx", 0),
            "source_video": video_info["name"],
        })

    annotations = []
    for ann in coco_partial["annotations"]:
        img = img_id_to_fname.get(ann["image_id"])
        if img is None:
            continue
        annotations.append({
            "file_name": img["file_name"],
            "category_name": cat_id_to_name.get(ann["category_id"], "unknown"),
            "bbox": ann.get("bbox", [0, 0, 0, 0]),
            "area": ann.get("area", 0),
            "obj_id": ann.get("obj_id", ann["id"]),
            "iscrowd": 0,
        })

    cat_names = set(cat_id_to_name.values())
    return {"images": images, "annotations": annotations, "category_names": cat_names}


def merge_partials(partials: list[dict]) -> dict:
    """Merge partial dicts into a single COCO dataset with global IDs."""
    all_cat_names = sorted({n for p in partials for n in p["category_names"]})
    cat_to_id = {name: i + 1 for i, name in enumerate(all_cat_names)}

    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": cid, "name": name} for name, cid in cat_to_id.items()],
    }

    img_id = 0
    ann_id = 0
    fname_to_img_id = {}

    for partial in partials:
        for img in partial["images"]:
            fname = img["file_name"]
            if fname not in fname_to_img_id:
                fname_to_img_id[fname] = img_id
                coco["images"].append({
                    "id": img_id, "file_name": fname,
                    "width": img["width"], "height": img["height"],
                    "frame_idx": img.get("frame_idx", 0),
                    "source_video": img.get("source_video", ""),
                })
                img_id += 1

        for ann in partial["annotations"]:
            coco["annotations"].append({
                "id": ann_id,
                "image_id": fname_to_img_id[ann["file_name"]],
                "category_id": cat_to_id[ann["category_name"]],
                "bbox": ann["bbox"],
                "area": ann["area"],
                "obj_id": ann.get("obj_id", ann_id),
                "iscrowd": ann.get("iscrowd", 0),
            })
            ann_id += 1

    return coco


def main():
    parser = argparse.ArgumentParser(
        description="Convert tracked metadata or keyframe annotations to COCO format."
    )
    parser.add_argument(
        "--anno-dir", type=str, required=True,
        help="Directory with _tracked_meta.json and/or _keyframes.json files.",
    )
    parser.add_argument(
        "--video-dir", type=str, required=True,
        help="Directory with source videos (needed for keyframe resolution).",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Output directory for COCO JSON.",
    )
    parser.add_argument(
        "--output-name", type=str, default="annotations.json",
        help="Filename for the output COCO JSON (default: annotations.json).",
    )
    args = parser.parse_args()

    anno_dir = Path(args.anno_dir)
    video_dir = Path(args.video_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not anno_dir.is_dir():
        print(f"Error: annotation directory not found: {anno_dir}")
        sys.exit(1)

    tracked_metas = sorted(anno_dir.glob("*_tracked_meta.json"))
    keyframe_files = sorted(anno_dir.glob("*_keyframes.json"))

    if not tracked_metas and not keyframe_files:
        print(f"No annotation files found in {anno_dir}")
        sys.exit(1)

    print(f"Found {len(tracked_metas)} tracked-meta + {len(keyframe_files)} keyframe files")
    print(f"Video dir: {video_dir}")
    print()

    partials = []

    for mp in tracked_metas:
        print(f"  tracked: {mp.name}")
        meta = load_tracked_meta(str(mp))
        partial = tracked_meta_to_partial(meta)
        print(f"    -> {len(partial['images'])} images, {len(partial['annotations'])} annotations")
        partials.append(partial)

    for kp in keyframe_files:
        print(f"  keyframes: {kp.name}")
        partial = keyframes_to_partial(kp, video_dir)
        if partial:
            print(f"    -> {len(partial['images'])} images, {len(partial['annotations'])} annotations")
            partials.append(partial)

    if not partials:
        print("No valid annotations to merge.")
        sys.exit(1)

    print(f"\nMerging {len(partials)} sources ...")
    coco = merge_partials(partials)

    out_path = output_dir / args.output_name
    with open(out_path, "w") as f:
        json.dump(coco, f, indent=2)

    print(f"\nCOCO dataset: {out_path}")
    print(f"  {len(coco['images'])} images, {len(coco['annotations'])} annotations, "
          f"{len(coco['categories'])} categories")
    for cat in coco["categories"]:
        n = sum(1 for a in coco["annotations"] if a["category_id"] == cat["id"])
        print(f"    [{cat['id']:2d}] {cat['name']}: {n}")


if __name__ == "__main__":
    main()
