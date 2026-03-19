"""
Video I/O — frame extraction, metadata, writing.
Extracted from trackInstruments/web/video_utils.py
"""

import cv2
import threading
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# VideoCapture cache — avoids reopening video per frame request
# ---------------------------------------------------------------------------

_cap_cache = {}
_cap_lock = threading.Lock()


def _get_cap(video_path):
    key = str(video_path)
    cap = _cap_cache.get(key)
    if cap and cap.isOpened():
        return cap
    cap = cv2.VideoCapture(key)
    _cap_cache[key] = cap
    return cap


def get_video_info(video_path: str) -> dict:
    """Return video metadata: width, height, fps, total_frames, duration_s."""
    with _cap_lock:
        cap = _get_cap(video_path)
        info = {
            "name": Path(video_path).name,
            "stem": Path(video_path).stem,
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": cap.get(cv2.CAP_PROP_FPS) or 30.0,
            "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        }
    info["duration_s"] = round(info["total_frames"] / info["fps"], 2)
    return info


def get_frame_bgr(video_path: str, frame_idx: int) -> np.ndarray:
    """Extract a single frame as BGR numpy array."""
    with _cap_lock:
        cap = _get_cap(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
    return frame if ret else None


def get_frame_jpeg(video_path: str, frame_idx: int, quality: int = 80) -> bytes:
    """Extract a single frame as JPEG bytes."""
    frame = get_frame_bgr(video_path, frame_idx)
    if frame is None:
        return None
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()


def extract_frames_to_dir(video_path: str, output_dir: str,
                           stride: int = 1, max_frames: int = None,
                           resize: tuple = None) -> list[dict]:
    """Extract frames from a video at given stride.

    Args:
        video_path: source video
        output_dir: directory to save JPEGs
        stride: extract every Nth frame (1 = every frame)
        max_frames: stop after this many extracted frames
        resize: (width, height) to resize, or None for original

    Returns: list of {file_name, frame_idx, width, height}
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stem = Path(video_path).stem

    frames = []
    idx = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % stride == 0:
            if resize:
                frame = cv2.resize(frame, resize)
            fname = f"{stem}_frame{idx:06d}.jpg"
            cv2.imwrite(str(out / fname), frame)
            h, w = frame.shape[:2]
            frames.append({
                "file_name": fname,
                "frame_idx": idx,
                "width": w,
                "height": h,
                "source_video": Path(video_path).name,
            })
            saved += 1
            if max_frames and saved >= max_frames:
                break
        idx += 1

    cap.release()
    print(f"Extracted {saved} frames from {Path(video_path).name} "
          f"(stride={stride}, total={total})")
    return frames


def write_video(frames_iter, out_path: str, fps: float, size: tuple):
    """Write frames to an MP4 video.

    Args:
        frames_iter: iterator yielding BGR numpy arrays
        out_path: output path
        fps: frames per second
        size: (width, height)
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, size)
    count = 0
    for frame in frames_iter:
        writer.write(frame)
        count += 1
    writer.release()
    print(f"Wrote {count} frames to {out_path}")
    return count
