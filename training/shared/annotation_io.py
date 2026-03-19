"""
Annotation I/O — load/save keyframes, COCO conversion, tracked metadata.
Extracted from trackInstruments/python/track.py
"""

import json
from pathlib import Path


def load_keyframes(video_path: str) -> dict:
    """Load keyframe annotations for a video.

    Searches for _keyframes.json first, then legacy _annotations.json.
    Returns: {frame_idx: [annotations]} where each annotation has
    obj_id, points, labels, color, text, count.
    """
    video_path = Path(video_path)
    stem = video_path.stem
    folder = video_path.parent

    # Try _keyframes.json (new multi-keyframe format)
    for f in sorted(folder.glob(f"*_keyframes.json"), reverse=True):
        with open(f) as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            # Convert string keys to int
            return {int(k): v for k, v in data.items()}

    # Try legacy _annotations.json (single-frame format)
    for f in sorted(folder.glob(f"*_annotations.json"), reverse=True):
        with open(f) as fh:
            data = json.load(fh)
        if isinstance(data, list):
            return {0: data}  # legacy: all annotations on frame 0

    return {}


def save_keyframes(data: dict, output_path: str):
    """Save keyframe annotations to JSON."""
    # Ensure keys are strings for JSON
    out = {str(k): v for k, v in data.items()}
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved keyframes to {output_path}")


def load_tracked_meta(meta_path: str) -> dict:
    """Load _tracked_meta.json produced by track.py.

    Returns dict with keys: fps, instruments, frames.
    frames is a list of {frame, obj_ids, bboxes, pip_paths}.
    """
    with open(meta_path) as f:
        return json.load(f)


def keyframes_to_coco(keyframes: dict, video_info: dict,
                       image_dir: str = None) -> dict:
    """Convert keyframe annotations to COCO JSON format.

    Args:
        keyframes: {frame_idx: [annotations]} from load_keyframes()
        video_info: {width, height, fps, stem} from video_io.get_video_info()
        image_dir: directory where extracted frames are stored

    Returns: COCO-format dict {images, annotations, categories}
    """
    coco = {"images": [], "annotations": [], "categories": []}

    # Build category list from unique text labels
    categories = {}
    cat_id = 1

    img_id = 0
    ann_id = 0

    for frame_idx, anns in sorted(keyframes.items()):
        fname = f"{video_info['stem']}_frame{frame_idx:06d}.jpg"
        coco["images"].append({
            "id": img_id,
            "file_name": fname,
            "width": video_info["width"],
            "height": video_info["height"],
            "frame_idx": frame_idx,
        })

        for ann in anns:
            text = ann.get("text", "unknown")
            if text not in categories:
                categories[text] = cat_id
                coco["categories"].append({"id": cat_id, "name": text})
                cat_id += 1

            # If we have points, create a simple annotation
            points = ann.get("points", [])
            coco["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": categories[text],
                "obj_id": ann.get("obj_id", ann_id),
                "points": points,
                "labels": ann.get("labels", [1] * len(points)),
                "count": ann.get("count", 1),
            })
            ann_id += 1
        img_id += 1

    return coco


def coco_to_keyframes(coco_path: str) -> dict:
    """Convert COCO JSON back to keyframe format."""
    with open(coco_path) as f:
        coco = json.load(f)

    # Build category map
    cat_map = {c["id"]: c["name"] for c in coco["categories"]}

    # Group annotations by image
    img_anns = {}
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        if img_id not in img_anns:
            img_anns[img_id] = []
        img_anns[img_id].append(ann)

    keyframes = {}
    for img in coco["images"]:
        frame_idx = img.get("frame_idx", img["id"])
        anns = img_anns.get(img["id"], [])
        kf_anns = []
        for ann in anns:
            kf_anns.append({
                "obj_id": ann.get("obj_id", ann["id"]),
                "text": cat_map.get(ann["category_id"], "unknown"),
                "points": ann.get("points", []),
                "labels": ann.get("labels", []),
                "count": ann.get("count", 1),
                "color": None,
            })
        if kf_anns:
            keyframes[frame_idx] = kf_anns

    return keyframes
