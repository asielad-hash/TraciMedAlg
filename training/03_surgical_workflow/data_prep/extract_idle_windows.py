"""
Extract sliding-window features for idle-period detection.

Reads per-video idle annotations and computes motion energy features
using frame differencing (lightweight proxy for RAFT optical flow).
Each window is labelled idle (1) or active (0) based on >=50% temporal
overlap with annotated idle segments.

Expected annotation format (data/idle_labels/{stem}_idle.json):
    {
      "video": "case001.mp4",
      "site": "hospital_a",
      "idle_periods": [
        {"start_sec": 300.0, "end_sec": 360.0, "reason": "waiting_for_path"},
        ...
      ]
    }

Output per video (saved as .npz):
    motion_energy   — (N_windows,) float32  mean absolute frame diff
    motion_std      — (N_windows,) float32  std of frame diffs in window
    labels          — (N_windows,) int32     1=idle, 0=active
    window_starts   — (N_windows,) float32  start time in seconds
    site            — str

Also computes per-site ambient noise profiles (median motion at rest)
and saves them to ambient_noise.json.

Usage:
    python extract_idle_windows.py --config ../config/config.yaml \
        --video-dir ../data/videos --idle-dir ../data/idle_labels \
        --output-dir ../data/clips/idle
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

# Shared library import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from shared.config import load_config
from shared.video_io import get_video_info

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
DEFAULT_CONFIG = str(Path(__file__).resolve().parent.parent / "config" / "config.yaml")


def load_idle_annotations(idle_path: Path) -> dict:
    """Load idle period annotations from JSON."""
    with open(idle_path) as f:
        return json.load(f)


def compute_frame_diffs(video_path: str, fps: float,
                        frame_interval: int) -> np.ndarray:
    """Compute absolute frame differences at the given interval.

    Uses grayscale conversion and mean absolute difference as a
    lightweight motion energy signal (proxy for optical flow magnitude).

    Args:
        video_path: path to video file.
        fps: video frame rate.
        frame_interval: sample every Nth frame.

    Returns:
        1-D float32 array of per-sample motion energy values.
    """
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    diffs = []
    prev_gray = None
    idx = 0

    while idx < total:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if prev_gray is not None:
            diff = np.abs(gray - prev_gray).mean()
            diffs.append(diff)
        else:
            diffs.append(0.0)
        prev_gray = gray
        idx += frame_interval

    cap.release()
    return np.array(diffs, dtype=np.float32)


def sliding_windows(diffs: np.ndarray, fps: float, frame_interval: int,
                    window_size_sec: float):
    """Compute sliding-window motion statistics.

    Args:
        diffs: per-sample motion energy from compute_frame_diffs.
        fps: video frame rate.
        frame_interval: sampling interval used for diffs.
        window_size_sec: window duration in seconds.

    Returns:
        motion_energy  — (N,) mean motion per window
        motion_std     — (N,) std motion per window
        window_starts  — (N,) start time in seconds
    """
    samples_per_sec = fps / frame_interval
    win_samples = max(int(window_size_sec * samples_per_sec), 1)
    stride = max(win_samples // 2, 1)  # 50% overlap between windows

    energies, stds, starts = [], [], []
    for s in range(0, len(diffs) - win_samples + 1, stride):
        window = diffs[s:s + win_samples]
        energies.append(window.mean())
        stds.append(window.std())
        start_sec = s / samples_per_sec
        starts.append(start_sec)

    return (np.array(energies, dtype=np.float32),
            np.array(stds, dtype=np.float32),
            np.array(starts, dtype=np.float32))


def label_windows(window_starts: np.ndarray, window_size_sec: float,
                  idle_periods: list, overlap_threshold: float = 0.5
                  ) -> np.ndarray:
    """Label each window as idle (1) or active (0).

    A window is labelled idle if >= overlap_threshold of its duration
    overlaps with any annotated idle period.

    Args:
        window_starts: start time of each window in seconds.
        window_size_sec: duration of each window.
        idle_periods: list of dicts with 'start_sec' and 'end_sec'.
        overlap_threshold: fraction of window that must overlap.

    Returns:
        int32 array of labels (1=idle, 0=active).
    """
    labels = np.zeros(len(window_starts), dtype=np.int32)
    for i, ws in enumerate(window_starts):
        we = ws + window_size_sec
        total_overlap = 0.0
        for seg in idle_periods:
            overlap_start = max(ws, seg["start_sec"])
            overlap_end = min(we, seg["end_sec"])
            if overlap_end > overlap_start:
                total_overlap += overlap_end - overlap_start
        if total_overlap / window_size_sec >= overlap_threshold:
            labels[i] = 1
    return labels


def process_video(video_path: Path, idle_path: Path, output_dir: Path,
                  window_size_sec: float, frame_interval: int
                  ) -> dict | None:
    """Extract idle-detection features from one video.

    Returns:
        Dict with site, idle motion stats for ambient noise profiling,
        or None on failure.
    """
    annotations = load_idle_annotations(idle_path)
    info = get_video_info(str(video_path))
    fps = info["fps"]
    site = annotations.get("site", "unknown")
    idle_periods = annotations.get("idle_periods", [])

    # Compute motion energy signal
    diffs = compute_frame_diffs(str(video_path), fps, frame_interval)
    if len(diffs) < 2:
        return None

    # Sliding windows
    energy, std, starts = sliding_windows(diffs, fps, frame_interval,
                                          window_size_sec)
    if len(energy) == 0:
        return None

    # Labels
    labels = label_windows(starts, window_size_sec, idle_periods)

    # Save
    stem = video_path.stem
    np.savez_compressed(
        str(output_dir / f"{stem}_idle_windows.npz"),
        motion_energy=energy,
        motion_std=std,
        labels=labels,
        window_starts=starts,
        site=site,
        fps=np.float32(fps),
        window_size_sec=np.float32(window_size_sec),
    )

    # Collect idle-window motion stats for ambient noise profiling
    idle_mask = labels == 1
    idle_energies = energy[idle_mask] if idle_mask.any() else np.array([])
    return {
        "site": site,
        "idle_energies": idle_energies.tolist(),
        "n_windows": len(energy),
        "n_idle": int(idle_mask.sum()),
    }


def compute_ambient_noise_profiles(site_stats: dict) -> dict:
    """Compute per-site ambient noise profile.

    For each site, compute the median and 90th-percentile motion energy
    during annotated idle periods.  This profile is used at inference
    to calibrate the idle threshold for each facility.

    Args:
        site_stats: {site_name: [motion_energy_values, ...]}

    Returns:
        Dict keyed by site with median and p90 motion statistics.
    """
    profiles = {}
    for site, energies_list in site_stats.items():
        all_e = np.array(energies_list, dtype=np.float32)
        if len(all_e) == 0:
            profiles[site] = {"median": 0.0, "p90": 0.0, "n_samples": 0}
            continue
        profiles[site] = {
            "median": float(np.median(all_e)),
            "p90": float(np.percentile(all_e, 90)),
            "n_samples": len(all_e),
        }
    return profiles


def main():
    parser = argparse.ArgumentParser(
        description="Extract sliding-window features for idle detection."
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
        "--idle-dir", type=str, required=True,
        help="Directory containing {stem}_idle.json annotation files"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for window .npz files (default: data/clips/idle)"
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    idle_cfg = config["idle_detection"]
    window_size_sec = idle_cfg["window_size_sec"]
    frame_interval = idle_cfg["frame_interval"]

    # Resolve directories
    video_dir = Path(args.video_dir)
    idle_dir = Path(args.idle_dir)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(config["_config_dir"]).parent / "data" / "clips" / "idle"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not video_dir.is_dir():
        print(f"Error: video directory not found: {video_dir}")
        sys.exit(1)
    if not idle_dir.is_dir():
        print(f"Error: idle annotation directory not found: {idle_dir}")
        sys.exit(1)

    # Discover videos
    videos = sorted(
        p for p in video_dir.iterdir()
        if p.suffix.lower() in VIDEO_EXTENSIONS and p.is_file()
    )
    print(f"Found {len(videos)} video(s) in {video_dir}")
    print(f"Window: {window_size_sec}s, frame interval: {frame_interval}")
    print(f"Output: {output_dir}\n")

    # Process each video
    site_idle_energies = defaultdict(list)
    total_windows = 0
    total_idle = 0

    for i, vpath in enumerate(videos, 1):
        anno_path = idle_dir / f"{vpath.stem}_idle.json"
        if not anno_path.exists():
            print(f"  [{i}/{len(videos)}] {vpath.name} — skipped (no annotation)")
            continue

        result = process_video(vpath, anno_path, output_dir,
                               window_size_sec, frame_interval)
        if result is None:
            print(f"  [{i}/{len(videos)}] {vpath.name} — skipped (too short)")
            continue

        site_idle_energies[result["site"]].extend(result["idle_energies"])
        total_windows += result["n_windows"]
        total_idle += result["n_idle"]
        print(f"  [{i}/{len(videos)}] {vpath.name} — "
              f"{result['n_windows']} windows ({result['n_idle']} idle), "
              f"site={result['site']}")

    # Ambient noise profiles
    profiles = compute_ambient_noise_profiles(site_idle_energies)
    profile_path = output_dir / "ambient_noise.json"
    with open(profile_path, "w") as f:
        json.dump(profiles, f, indent=2)
    print(f"\nAmbient noise profiles ({len(profiles)} sites): {profile_path}")

    print(f"\nDone. {total_windows} windows total ({total_idle} idle) "
          f"saved to {output_dir}")


if __name__ == "__main__":
    main()
