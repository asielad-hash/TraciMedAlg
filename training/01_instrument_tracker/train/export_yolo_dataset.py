"""
Export YOLO-seg dataset from tracked metadata.

Reads _tracked_meta.json files produced by the instrument tracker and
generates a YOLO segmentation dataset with polygon label files suitable
for Ultralytics YOLOv8-seg training.

Usage:
    python export_yolo_dataset.py \
        --meta-dir ./data/tracked \
        --images-dir ./data/frames \
        --output-dir ./data/yolo_dataset \
        --val-ratio 0.2
"""

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Shared library
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from shared.annotation_io import load_tracked_meta
from shared.mask_utils import mask_to_bbox


def _rle_to_mask(rle: dict, height: int, width: int) -> np.ndarray:
    """Decode a run-length encoded mask (COCO-style counts/size)."""
    counts = rle.get("counts", [])
    if isinstance(counts, str):
        # COCO compressed RLE — use pycocotools if available, else skip
        try:
            from pycocotools import mask as mask_util
            m = mask_util.decode(rle)
            return m.astype(np.uint8)
        except ImportError:
            return np.zeros((height, width), dtype=np.uint8)
    # Uncompressed RLE: list of ints [start, length, start, length, ...]
    mask = np.zeros(height * width, dtype=np.uint8)
    pos = 0
    for i, val in enumerate(counts):
        if i % 2 == 1:
            mask[pos:pos + val] = 1
        pos += val
    return mask.reshape((height, width), order="F")


def _mask_to_polygon(mask: np.ndarray, normalize: bool = True) -> list[list[float]]:
    """Convert a binary mask to YOLO polygon format.

    Returns a list of polygons, each polygon is a flat list of
    normalised x y x y ... coordinates.
    """
    h, w = mask.shape[:2]
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    polygons = []
    for cnt in contours:
        if len(cnt) < 3:
            continue
        pts = cnt.squeeze()
        if pts.ndim != 2:
            continue
        flat = []
        for x, y in pts:
            if normalize:
                flat.extend([float(x) / w, float(y) / h])
            else:
                flat.extend([float(x), float(y)])
        polygons.append(flat)
    return polygons


def build_class_map(meta_files: list[Path]) -> dict[str, int]:
    """Scan all meta files and build a {instrument_text: class_id} map."""
    names = set()
    for mf in meta_files:
        meta = load_tracked_meta(str(mf))
        for info in meta.get("instruments", {}).values():
            text = info.get("text", "unknown").strip()
            names.add(text)
    return {name: idx for idx, name in enumerate(sorted(names))}


def export_frame(
    meta: dict,
    frame_data: dict,
    images_dir: Path,
    out_images: Path,
    out_labels: Path,
    class_map: dict[str, int],
    video_stem: str,
) -> bool:
    """Export one frame's image and YOLO label file.

    Returns True if at least one valid annotation was written.
    """
    fidx = frame_data["frame"]
    instruments = meta.get("instruments", {})
    obj_ids = frame_data.get("obj_ids", [])

    if not obj_ids:
        return False

    # Locate frame image
    frame_name = f"{video_stem}_frame{fidx:06d}.jpg"
    src_img = images_dir / frame_name
    if not src_img.exists():
        # Try png fallback
        frame_name_png = f"{video_stem}_frame{fidx:06d}.png"
        src_img = images_dir / frame_name_png
        if not src_img.exists():
            return False

    # Read image dimensions
    img = cv2.imread(str(src_img))
    if img is None:
        return False
    h, w = img.shape[:2]

    label_lines = []

    for oid in obj_ids:
        info = instruments.get(str(oid), {})
        text = info.get("text", "unknown").strip()
        cls_id = class_map.get(text)
        if cls_id is None:
            continue

        # Try to load mask from pip_path
        mask = None
        pip_paths = frame_data.get("pip_paths", {})
        pip = pip_paths.get(str(oid)) if isinstance(pip_paths, dict) else None
        if pip and Path(pip).exists():
            m = cv2.imread(pip, cv2.IMREAD_GRAYSCALE)
            if m is not None:
                mask = (m > 127).astype(np.uint8)

        # Try RLE mask stored in frame_data
        if mask is None:
            rle_masks = frame_data.get("masks", {})
            rle = rle_masks.get(str(oid))
            if rle is not None:
                mask = _rle_to_mask(rle, h, w)

        if mask is None or mask.sum() == 0:
            continue

        polys = _mask_to_polygon(mask, normalize=True)
        for poly in polys:
            if len(poly) < 6:
                continue
            coords_str = " ".join(f"{v:.6f}" for v in poly)
            label_lines.append(f"{cls_id} {coords_str}")

    if not label_lines:
        return False

    # Copy image
    dst_img = out_images / frame_name
    shutil.copy2(str(src_img), str(dst_img))

    # Write label
    label_path = out_labels / (Path(frame_name).stem + ".txt")
    label_path.write_text("\n".join(label_lines) + "\n")
    return True


def write_data_yaml(output_dir: Path, class_map: dict[str, int]):
    """Write Ultralytics data.yaml."""
    yaml_path = output_dir / "data.yaml"
    names_block = "\n".join(f"  {idx}: '{name}'" for name, idx in
                            sorted(class_map.items(), key=lambda x: x[1]))
    content = (
        f"path: {output_dir.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"\n"
        f"nc: {len(class_map)}\n"
        f"names:\n"
        f"{names_block}\n"
    )
    yaml_path.write_text(content)
    print(f"Wrote {yaml_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Export YOLO-seg dataset from _tracked_meta.json files."
    )
    parser.add_argument(
        "--meta-dir", required=True,
        help="Directory containing *_tracked_meta.json files.",
    )
    parser.add_argument(
        "--images-dir", required=True,
        help="Directory containing extracted video frames (JPG/PNG).",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Output directory for YOLO dataset.",
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.2,
        help="Fraction of videos to use for validation (default: 0.2).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for train/val split.",
    )
    args = parser.parse_args()

    meta_dir = Path(args.meta_dir)
    images_dir = Path(args.images_dir)
    output_dir = Path(args.output_dir)

    meta_files = sorted(meta_dir.glob("*_tracked_meta.json"))
    if not meta_files:
        print(f"ERROR: No *_tracked_meta.json files found in {meta_dir}")
        sys.exit(1)

    print(f"Found {len(meta_files)} tracked metadata file(s)")

    # Build class map
    class_map = build_class_map(meta_files)
    print(f"Classes ({len(class_map)}): {list(class_map.keys())}")

    # Train/val split at the video level
    random.seed(args.seed)
    shuffled = list(meta_files)
    random.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * args.val_ratio))
    val_files = set(str(f) for f in shuffled[:n_val])

    # Create directory structure
    for split in ("train", "val"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    total_frames = 0
    for mf in meta_files:
        meta = load_tracked_meta(str(mf))
        video_stem = mf.stem.replace("_tracked_meta", "")
        split = "val" if str(mf) in val_files else "train"
        out_images = output_dir / "images" / split
        out_labels = output_dir / "labels" / split

        frames = meta.get("frames", [])
        exported = 0
        for fdata in frames:
            ok = export_frame(
                meta, fdata, images_dir, out_images, out_labels,
                class_map, video_stem,
            )
            if ok:
                exported += 1

        total_frames += exported
        print(f"  {mf.name}: {exported}/{len(frames)} frames -> {split}")

    write_data_yaml(output_dir, class_map)
    print(f"\nDone. Exported {total_frames} annotated frames total.")


if __name__ == "__main__":
    main()
