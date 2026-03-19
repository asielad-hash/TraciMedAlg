"""
Detailed latency profiling for the privacy engine anonymization pipeline.

Runs the full detect-and-blur pipeline on N test frames and reports
median, p95, p99 latency as well as histogram data.  Uses RetinaFace for
face detection, PaddleOCR for text detection, and shared.cv_utils.anonymize_frame()
for the blur/pixelation step.

Usage:
    python benchmark_latency.py --test-dir ../data/test_frames --num-frames 50
    python benchmark_latency.py --test-dir ../data/test_frames --output latency_report.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Shared library import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from shared.eval_metrics import measure_latency
from shared.cv_utils import anonymize_frame

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

DEFAULT_TEST_DIR = str(Path(__file__).resolve().parent.parent / "data" / "test_frames")


# ---------------------------------------------------------------------------
# Detector helpers (same graceful-fallback pattern as evaluate.py)
# ---------------------------------------------------------------------------

def _load_retinaface():
    """Return a face-detection callable or None."""
    try:
        from retinaface import RetinaFace  # type: ignore

        def detect_faces(frame_bgr: np.ndarray) -> list[list[int]]:
            results = RetinaFace.detect_faces(frame_bgr)
            boxes = []
            if isinstance(results, dict):
                for key in results:
                    area = results[key].get("facial_area", [])
                    if len(area) == 4:
                        boxes.append([int(v) for v in area])
            return boxes

        return detect_faces
    except ImportError:
        return None


def _load_paddleocr():
    """Return a text-detection callable or None."""
    try:
        from paddleocr import PaddleOCR  # type: ignore

        ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)

        def detect_text(frame_bgr: np.ndarray) -> list[list[int]]:
            results = ocr.ocr(frame_bgr, cls=True)
            boxes = []
            if results and results[0]:
                for line in results[0]:
                    polygon = line[0]
                    xs = [p[0] for p in polygon]
                    ys = [p[1] for p in polygon]
                    boxes.append([
                        int(min(xs)), int(min(ys)),
                        int(max(xs)), int(max(ys)),
                    ])
            return boxes

        return detect_text
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def boxes_to_mask(boxes: list[list[int]], shape: tuple) -> np.ndarray:
    """Convert a list of [x1, y1, x2, y2] boxes into a binary mask."""
    mask = np.zeros(shape[:2], dtype=np.uint8)
    for x1, y1, x2, y2 in boxes:
        mask[max(y1, 0):min(y2, shape[0]), max(x1, 0):min(x2, shape[1])] = 255
    return mask


def run_pipeline(frame_bgr: np.ndarray, face_fn, text_fn) -> np.ndarray:
    """Full detect + blur pipeline on a single frame.

    Returns the anonymized frame.
    """
    all_boxes: list[list[int]] = []
    if face_fn:
        all_boxes.extend(face_fn(frame_bgr))
    if text_fn:
        all_boxes.extend(text_fn(frame_bgr))

    if not all_boxes:
        return frame_bgr

    mask = boxes_to_mask(all_boxes, frame_bgr.shape)
    return anonymize_frame(frame_bgr, mask)


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def collect_frames(test_dir: Path, num_frames: int) -> list[np.ndarray]:
    """Load up to *num_frames* images from test_dir.

    If there are fewer images than requested, frames are cycled.
    """
    image_paths = sorted(
        p for p in test_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        return []

    frames: list[np.ndarray] = []
    idx = 0
    while len(frames) < num_frames:
        path = image_paths[idx % len(image_paths)]
        frame = cv2.imread(str(path))
        if frame is not None:
            frames.append(frame)
        idx += 1
        if idx >= num_frames * 2:
            break  # safety valve
    return frames


def compute_histogram(latencies: np.ndarray, bins: int = 20) -> dict:
    """Build histogram data suitable for JSON serialization."""
    counts, edges = np.histogram(latencies, bins=bins)
    return {
        "bin_edges_ms": [round(float(e), 2) for e in edges],
        "counts": [int(c) for c in counts],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark latency of the full anonymization pipeline "
                    "(detect + blur)."
    )
    parser.add_argument(
        "--test-dir", type=str, default=DEFAULT_TEST_DIR,
        help="Directory with test frame images"
    )
    parser.add_argument(
        "--num-frames", type=int, default=50,
        help="Number of frames to benchmark (default: 50, cycles if fewer available)"
    )
    parser.add_argument(
        "--warmup", type=int, default=3,
        help="Warmup iterations before measurement (default: 3)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to save latency report JSON"
    )
    args = parser.parse_args()

    test_dir = Path(args.test_dir)
    if not test_dir.is_dir():
        print(f"Error: test directory not found: {test_dir}")
        sys.exit(1)

    # Load detectors ---------------------------------------------------------
    face_fn = _load_retinaface()
    text_fn = _load_paddleocr()

    if face_fn is None and text_fn is None:
        print("Neither RetinaFace nor PaddleOCR is installed.")
        print("Install at least one detector to run the benchmark:")
        print("  pip install retina-face")
        print("  pip install paddlepaddle paddleocr")
        sys.exit(1)

    available = []
    if face_fn:
        available.append("RetinaFace (faces)")
    if text_fn:
        available.append("PaddleOCR (text/badges)")
    print(f"Detectors loaded: {', '.join(available)}")

    # Load frames ------------------------------------------------------------
    frames = collect_frames(test_dir, args.num_frames)
    if not frames:
        print(f"No images found in {test_dir}")
        sys.exit(1)

    print(f"Loaded {len(frames)} frame(s) for benchmarking")
    h, w = frames[0].shape[:2]
    print(f"Frame resolution: {w}x{h}")
    print()

    # Warmup -----------------------------------------------------------------
    if args.warmup > 0:
        print(f"Running {args.warmup} warmup iteration(s) ...")
        for i in range(min(args.warmup, len(frames))):
            run_pipeline(frames[i], face_fn, text_fn)

    # Benchmark --------------------------------------------------------------
    print(f"Benchmarking {len(frames)} frame(s) ...")
    latencies = []

    for i, frame in enumerate(frames):
        t0 = time.perf_counter()
        run_pipeline(frame, face_fn, text_fn)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed_ms)

        if (i + 1) % 10 == 0 or i == len(frames) - 1:
            print(f"  [{i + 1}/{len(frames)}] last={elapsed_ms:.1f} ms")

    arr = np.array(latencies)

    # Also measure with shared.eval_metrics for consistency
    if len(frames) >= 3:
        shared_stats = measure_latency(
            run_pipeline, frames[0], face_fn, text_fn, n_runs=min(5, len(frames))
        )
    else:
        shared_stats = {}

    # Report -----------------------------------------------------------------
    report = {
        "num_frames": len(latencies),
        "resolution": f"{w}x{h}",
        "detectors": available,
        "median_ms": round(float(np.median(arr)), 2),
        "mean_ms": round(float(np.mean(arr)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "p99_ms": round(float(np.percentile(arr, 99)), 2),
        "min_ms": round(float(np.min(arr)), 2),
        "max_ms": round(float(np.max(arr)), 2),
        "std_ms": round(float(np.std(arr)), 2),
        "histogram": compute_histogram(arr),
        "shared_measure_latency": shared_stats,
    }

    print()
    print("=" * 55)
    print(f"  Frames benchmarked : {report['num_frames']}")
    print(f"  Resolution         : {report['resolution']}")
    print(f"  Median latency     : {report['median_ms']:.2f} ms")
    print(f"  Mean latency       : {report['mean_ms']:.2f} ms")
    print(f"  P95 latency        : {report['p95_ms']:.2f} ms")
    print(f"  P99 latency        : {report['p99_ms']:.2f} ms")
    print(f"  Min / Max          : {report['min_ms']:.2f} / {report['max_ms']:.2f} ms")
    print(f"  Std dev            : {report['std_ms']:.2f} ms")

    target_ms = 200
    status = "PASS" if report["p95_ms"] <= target_ms else "FAIL"
    print(f"  Target (<={target_ms} ms p95) : {status}")
    print("=" * 55)

    report["target_ms"] = target_ms
    report["status"] = status

    # Save -------------------------------------------------------------------
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = test_dir.parent / "latency_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {out_path}")


if __name__ == "__main__":
    main()
