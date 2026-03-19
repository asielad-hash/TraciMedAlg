"""
Extract temporal clips for surgical phase recognition training.

Reads per-video phase boundary annotations from JSON files and extracts
overlapping 64-frame clips (stride 16) from each video.  Each clip is
labelled with per-frame phase IDs and a majority-vote phase label.

Expected annotation format (data/phase_labels/{stem}_phases.json):
    {
      "video": "case001.mp4",
      "fps": 30.0,
      "phases": [
        {"phase": "setup_preparation", "start_sec": 0.0, "end_sec": 120.5},
        {"phase": "incision_active_surgery", "start_sec": 120.5, "end_sec": 3600.0},
        ...
      ]
    }

Output per clip (saved as .npz):
    frames   — (64, H, W, 3) uint8 BGR array
    phase_ids — (64,) int32 per-frame phase index
    majority_phase — int32 scalar

Usage:
    python extract_phase_clips.py --config ../config/config.yaml \
        --video-dir ../data/videos --phase-dir ../data/phase_labels \
        --output-dir ../data/clips/phase
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

# Shared library import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from shared.config import load_config
from shared.video_io import get_video_info

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
DEFAULT_CONFIG = str(Path(__file__).resolve().parent.parent / "config" / "config.yaml")


def load_phase_annotations(phase_path: Path) -> dict:
    """Load phase boundary annotations from JSON."""
    with open(phase_path) as f:
        return json.load(f)


def build_frame_phase_map(annotations: dict, fps: float, total_frames: int,
                          phase_to_id: dict) -> np.ndarray:
    """Create a per-frame array of phase IDs from boundary annotations.

    Args:
        annotations: parsed JSON with 'phases' list.
        fps: video frame rate.
        total_frames: total frame count in the video.
        phase_to_id: mapping from phase name to integer ID.

    Returns:
        np.ndarray of shape (total_frames,) with dtype int32.
    """
    phase_ids = np.full(total_frames, -1, dtype=np.int32)
    for seg in annotations["phases"]:
        start_frame = int(seg["start_sec"] * fps)
        end_frame = min(int(seg["end_sec"] * fps), total_frames)
        pid = phase_to_id.get(seg["phase"], -1)
        phase_ids[start_frame:end_frame] = pid
    return phase_ids


def extract_clip_frames(video_path: str, start_idx: int,
                        clip_length: int) -> np.ndarray | None:
    """Read a contiguous block of frames from a video.

    Returns:
        np.ndarray of shape (clip_length, H, W, 3) or None on failure.
    """
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
    frames = []
    for _ in range(clip_length):
        ret, frame = cap.read()
        if not ret:
            cap.release()
            return None
        frames.append(frame)
    cap.release()
    return np.stack(frames, axis=0)


def process_video(video_path: Path, phase_path: Path, output_dir: Path,
                  phase_to_id: dict, clip_length: int, stride: int) -> int:
    """Extract all phase clips from one video.

    Returns:
        Number of clips saved.
    """
    annotations = load_phase_annotations(phase_path)
    info = get_video_info(str(video_path))
    fps = info["fps"]
    total_frames = info["total_frames"]

    frame_phases = build_frame_phase_map(annotations, fps, total_frames, phase_to_id)
    stem = video_path.stem
    saved = 0

    for start in range(0, total_frames - clip_length + 1, stride):
        end = start + clip_length
        clip_phases = frame_phases[start:end]

        # Skip clips with unlabelled frames
        if (clip_phases < 0).any():
            continue

        # Majority vote
        majority_phase = Counter(clip_phases.tolist()).most_common(1)[0][0]

        # Extract pixel data
        frames = extract_clip_frames(str(video_path), start, clip_length)
        if frames is None:
            continue

        clip_name = f"{stem}_clip{start:07d}.npz"
        np.savez_compressed(
            str(output_dir / clip_name),
            frames=frames,
            phase_ids=clip_phases,
            majority_phase=np.int32(majority_phase),
            start_frame=np.int32(start),
            fps=np.float32(fps),
            video_stem=stem,
        )
        saved += 1

    return saved


def main():
    parser = argparse.ArgumentParser(
        description="Extract temporal clips for surgical phase recognition."
    )
    parser.add_argument(
        "--config", type=str, default=DEFAULT_CONFIG,
        help="Path to config YAML (default: ../config/config.yaml)"
    )
    parser.add_argument(
        "--video-dir", type=str, required=True,
        help="Directory containing source video files"
    )
    parser.add_argument(
        "--phase-dir", type=str, required=True,
        help="Directory containing {stem}_phases.json annotation files"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for clip .npz files (default: data/clips/phase)"
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    phase_cfg = config["phase_recognition"]
    phases = phase_cfg["phases"]
    phase_to_id = {name: i for i, name in enumerate(phases)}
    clip_length = phase_cfg["training"]["clip_length_frames"]
    stride = phase_cfg["training"]["clip_stride_frames"]

    # Resolve directories
    video_dir = Path(args.video_dir)
    phase_dir = Path(args.phase_dir)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(config["_config_dir"]).parent / "data" / "clips" / "phase"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not video_dir.is_dir():
        print(f"Error: video directory not found: {video_dir}")
        sys.exit(1)
    if not phase_dir.is_dir():
        print(f"Error: phase annotation directory not found: {phase_dir}")
        sys.exit(1)

    # Discover videos that have matching annotations
    videos = sorted(
        p for p in video_dir.iterdir()
        if p.suffix.lower() in VIDEO_EXTENSIONS and p.is_file()
    )
    print(f"Found {len(videos)} video(s) in {video_dir}")
    print(f"Clip length: {clip_length} frames, stride: {stride}")
    print(f"Phases ({len(phases)}): {phases}")
    print(f"Output: {output_dir}\n")

    # Save phase mapping for downstream use
    meta = {"phases": phases, "phase_to_id": phase_to_id,
            "clip_length": clip_length, "stride": stride}
    with open(output_dir / "phase_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    total_clips = 0
    for i, vpath in enumerate(videos, 1):
        anno_path = phase_dir / f"{vpath.stem}_phases.json"
        if not anno_path.exists():
            print(f"  [{i}/{len(videos)}] {vpath.name} — skipped (no annotation)")
            continue

        n = process_video(vpath, anno_path, output_dir, phase_to_id,
                          clip_length, stride)
        total_clips += n
        print(f"  [{i}/{len(videos)}] {vpath.name} — {n} clips")

    print(f"\nDone. Saved {total_clips} clips to {output_dir}")


if __name__ == "__main__":
    main()
