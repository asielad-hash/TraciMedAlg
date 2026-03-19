"""
Extract event clips from annotated tray-monitoring videos.

Reads per-video event annotations ({video_stem}_events.json) and cuts each
video into 32-frame clips centered on annotated TAKE_OUT / PUT_BACK events.
Also samples NO_EVENT clips from non-event intervals to provide negative
examples.  Each clip is saved as a numbered directory of JPEG frames.

Expected annotation format (see data/DATA_README.txt):
  {
    "video": "...",
    "tray_roi": [x, y, w, h],
    "events": [
      {"type": "TAKE_OUT", "timestamp_sec": 45.2, "duration_sec": 2.1, ...},
      ...
    ]
  }

Output structure:
  output_dir/
    clip_000000/          # numbered clip directories
      frame_00.jpg
      frame_01.jpg
      ...
    clip_000001/
      ...
    manifest.json         # clip_id -> {event_type, source_video, center_frame, ...}

Usage:
    python extract_event_clips.py --video-dir data/videos --anno-dir data/annotations --output-dir data/event_clips
    python extract_event_clips.py --config ../config/config.yaml --video-dir data/videos --anno-dir data/annotations
"""

import argparse
import json
import random
import sys
from pathlib import Path

# Shared library import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from shared.config import load_config
from shared.video_io import get_video_info, get_frame_bgr

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

DEFAULT_CONFIG = str(Path(__file__).resolve().parent.parent / "config" / "config.yaml")


def load_event_annotations(anno_path: Path) -> dict:
    """Load a single event annotation JSON file."""
    with open(anno_path) as f:
        return json.load(f)


def find_annotation_files(anno_dir: Path) -> list[Path]:
    """Collect all *_events.json annotation files, sorted by name."""
    return sorted(anno_dir.glob("*_events.json"))


def find_video_for_annotation(anno: dict, video_dir: Path) -> Path | None:
    """Resolve the video file referenced by an annotation."""
    video_name = anno.get("video", "")
    candidate = video_dir / video_name
    if candidate.is_file():
        return candidate
    # Try matching by stem
    stem = Path(video_name).stem
    for ext in VIDEO_EXTENSIONS:
        candidate = video_dir / f"{stem}{ext}"
        if candidate.is_file():
            return candidate
    return None


def event_center_frame(event: dict, fps: float) -> int:
    """Compute the center frame index of an annotated event."""
    start_sec = event["timestamp_sec"]
    duration = event.get("duration_sec", 1.0)
    center_sec = start_sec + duration / 2.0
    return int(round(center_sec * fps))


def extract_clip_frames(video_path: str, center_frame: int,
                        clip_length: int, total_frames: int) -> tuple[int, int]:
    """Compute start/end frame for a clip centered on center_frame.

    Returns (start_frame, end_frame) clamped to video bounds.
    """
    half = clip_length // 2
    start = center_frame - half
    end = start + clip_length

    # Clamp to video bounds
    if start < 0:
        start = 0
        end = clip_length
    if end > total_frames:
        end = total_frames
        start = max(0, end - clip_length)

    return start, end


def save_clip(video_path: str, start_frame: int, end_frame: int,
              clip_dir: Path) -> int:
    """Extract frames [start_frame, end_frame) and save as JPEGs.

    Returns number of frames actually saved.
    """
    import cv2

    clip_dir.mkdir(parents=True, exist_ok=True)
    saved = 0

    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    for i in range(end_frame - start_frame):
        ret, frame = cap.read()
        if not ret:
            break
        out_path = clip_dir / f"frame_{i:02d}.jpg"
        cv2.imwrite(str(out_path), frame)
        saved += 1

    cap.release()
    return saved


def sample_no_event_intervals(events: list, total_frames: int, fps: float,
                              clip_length: int, margin_sec: float = 3.0) -> list[int]:
    """Sample center-frame positions for NO_EVENT clips.

    Picks positions that are at least margin_sec away from any annotated event.
    Returns approximately as many NO_EVENT centers as there are event clips.
    """
    margin_frames = int(margin_sec * fps)

    # Build set of "occupied" frame ranges
    occupied = set()
    for ev in events:
        center = event_center_frame(ev, fps)
        half = clip_length // 2
        lo = max(0, center - half - margin_frames)
        hi = min(total_frames, center + half + margin_frames)
        for f in range(lo, hi):
            occupied.add(f)

    # Candidate positions: every clip_length frames that are not occupied
    candidates = []
    half = clip_length // 2
    for c in range(half, total_frames - half, clip_length):
        if c not in occupied:
            candidates.append(c)

    # Sample roughly the same count as event clips
    n_desired = max(len(events), 1)
    if len(candidates) <= n_desired:
        return candidates
    return sorted(random.sample(candidates, n_desired))


def main():
    parser = argparse.ArgumentParser(
        description="Extract event clips from annotated tray-monitoring videos."
    )
    parser.add_argument(
        "--config", type=str, default=DEFAULT_CONFIG,
        help="Path to config YAML (default: ../config/config.yaml)"
    )
    parser.add_argument(
        "--video-dir", type=str, required=True,
        help="Directory containing source tray videos"
    )
    parser.add_argument(
        "--anno-dir", type=str, required=True,
        help="Directory containing *_events.json annotation files"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for clip directories (default: data/event_clips)"
    )
    parser.add_argument(
        "--clip-length", type=int, default=None,
        help="Number of frames per clip (default: from config, typically 32)"
    )
    parser.add_argument(
        "--no-event-margin", type=float, default=3.0,
        help="Minimum seconds between NO_EVENT clips and real events (default: 3.0)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for NO_EVENT sampling (default: 42)"
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    training_cfg = config.get("training", {})
    clip_length = args.clip_length or training_cfg.get("clip_length_frames", 32)

    random.seed(args.seed)

    # Resolve directories
    video_dir = Path(args.video_dir)
    anno_dir = Path(args.anno_dir)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(config.get("_config_dir", ".")).parent / "data" / "event_clips"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not video_dir.is_dir():
        print(f"Error: video directory not found: {video_dir}")
        sys.exit(1)
    if not anno_dir.is_dir():
        print(f"Error: annotation directory not found: {anno_dir}")
        sys.exit(1)

    # Find annotation files
    anno_files = find_annotation_files(anno_dir)
    if not anno_files:
        print(f"No *_events.json files found in {anno_dir}")
        sys.exit(1)

    print(f"Found {len(anno_files)} annotation file(s)")
    print(f"Clip length: {clip_length} frames")
    print(f"Output: {output_dir}")
    print()

    manifest = {}
    clip_counter = 0
    stats = {"TAKE_OUT": 0, "PUT_BACK": 0, "NO_EVENT": 0}

    for anno_path in anno_files:
        anno = load_event_annotations(anno_path)
        video_path = find_video_for_annotation(anno, video_dir)
        if video_path is None:
            print(f"  Warning: video not found for {anno_path.name}, skipping")
            continue

        info = get_video_info(str(video_path))
        fps = info["fps"]
        total_frames = info["total_frames"]
        events = anno.get("events", [])

        print(f"Processing {video_path.name}: {len(events)} events, "
              f"{total_frames} frames, {fps:.1f} fps")

        # --- Extract event clips (TAKE_OUT / PUT_BACK) ---
        for ev in events:
            event_type = ev["type"]
            center = event_center_frame(ev, fps)
            start, end = extract_clip_frames(
                str(video_path), center, clip_length, total_frames
            )

            clip_id = f"clip_{clip_counter:06d}"
            clip_dir = output_dir / clip_id
            n_saved = save_clip(str(video_path), start, end, clip_dir)

            manifest[clip_id] = {
                "event_type": event_type,
                "source_video": video_path.name,
                "center_frame": center,
                "start_frame": start,
                "end_frame": end,
                "frames_saved": n_saved,
                "instrument": ev.get("instrument_if_known"),
                "confidence": ev.get("confidence", "certain"),
            }
            stats[event_type] = stats.get(event_type, 0) + 1
            clip_counter += 1

        # --- Sample and extract NO_EVENT clips ---
        no_event_centers = sample_no_event_intervals(
            events, total_frames, fps, clip_length,
            margin_sec=args.no_event_margin,
        )
        for center in no_event_centers:
            start, end = extract_clip_frames(
                str(video_path), center, clip_length, total_frames
            )

            clip_id = f"clip_{clip_counter:06d}"
            clip_dir = output_dir / clip_id
            n_saved = save_clip(str(video_path), start, end, clip_dir)

            manifest[clip_id] = {
                "event_type": "NO_EVENT",
                "source_video": video_path.name,
                "center_frame": center,
                "start_frame": start,
                "end_frame": end,
                "frames_saved": n_saved,
                "instrument": None,
                "confidence": "certain",
            }
            stats["NO_EVENT"] += 1
            clip_counter += 1

    # Write manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print()
    print(f"Done. Extracted {clip_counter} clips total.")
    for event_type, count in sorted(stats.items()):
        print(f"  {event_type}: {count}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
