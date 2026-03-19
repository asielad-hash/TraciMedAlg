"""
Extract video frames at configurable stride.

Reads all videos from a source directory and extracts JPEG frames
into data/images/ using the shared video_io library. Frame extraction
stride is configurable (e.g., stride=30 extracts one frame per second
for a 30 fps video).

Usage:
    python extract_frames.py --config ../config/config.yaml --video-dir /path/to/videos
    python extract_frames.py --config ../config/config.yaml --video-dir /path/to/videos --stride 15 --output-dir ./out
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from shared.config import load_config
from shared.video_io import extract_frames_to_dir

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def find_videos(video_dir: Path) -> list[Path]:
    """Return sorted list of video files in a directory."""
    videos = [
        p for p in video_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    ]
    return sorted(videos)


def main():
    parser = argparse.ArgumentParser(
        description="Extract video frames at configurable stride."
    )
    parser.add_argument(
        "--config", type=str,
        default=str(Path(__file__).resolve().parent.parent / "config" / "config.yaml"),
        help="Path to config YAML (default: ../config/config.yaml)",
    )
    parser.add_argument(
        "--video-dir", type=str, required=True,
        help="Directory containing source video files.",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for extracted frames (default: config dataset.root / images).",
    )
    parser.add_argument(
        "--stride", type=int, default=30,
        help="Extract every Nth frame (default: 30).",
    )
    parser.add_argument(
        "--max-frames", type=int, default=None,
        help="Maximum frames to extract per video (default: unlimited).",
    )
    parser.add_argument(
        "--resize", type=int, nargs=2, default=None, metavar=("W", "H"),
        help="Resize frames to WxH (default: original resolution).",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    # Resolve output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        config_dir = Path(config["_config_dir"])
        data_root = config_dir / config.get("dataset", {}).get("root", "./data")
        output_dir = data_root / "images"
    output_dir.mkdir(parents=True, exist_ok=True)

    video_dir = Path(args.video_dir)
    if not video_dir.is_dir():
        print(f"Error: video directory not found: {video_dir}")
        sys.exit(1)

    videos = find_videos(video_dir)
    if not videos:
        print(f"No video files found in {video_dir}")
        sys.exit(1)

    print(f"Found {len(videos)} video(s) in {video_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Stride: {args.stride}")

    resize = tuple(args.resize) if args.resize else None
    all_frame_info = []

    for i, vpath in enumerate(videos, 1):
        print(f"\n[{i}/{len(videos)}] Processing {vpath.name} ...")
        frames = extract_frames_to_dir(
            video_path=str(vpath),
            output_dir=str(output_dir),
            stride=args.stride,
            max_frames=args.max_frames,
            resize=resize,
        )
        all_frame_info.extend(frames)

    # Write manifest
    manifest_path = output_dir / "frame_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump({
            "stride": args.stride,
            "total_extracted": len(all_frame_info),
            "videos_processed": len(videos),
            "frames": all_frame_info,
        }, f, indent=2)
    print(f"\nDone. Extracted {len(all_frame_info)} total frames.")
    print(f"Manifest saved to {manifest_path}")


if __name__ == "__main__":
    main()
