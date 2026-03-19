"""
Main evaluation script for the privacy engine.

Measures face detection recall, badge/text detection recall, false positive
rate, and per-frame latency against the targets defined in config.yaml:
    face_detection_recall  >= 99%
    badge_detection_recall >= 98%
    latency_per_frame_ms   <  200 ms
    false_positive_rate    <= 5%

Loads test frames and ground truth from data/test_frames/ (produced by
data_prep/prepare_test_set.py).  Runs RetinaFace for face detection and
PaddleOCR for text/badge detection, then compares against ground truth
bounding boxes using IoU-based matching.

Usage:
    python evaluate.py --mode all
    python evaluate.py --mode faces --test-dir ../data/test_frames
    python evaluate.py --config ../config/config.yaml --mode latency
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
from shared.config import load_config
from shared.eval_metrics import compute_f1, measure_latency, MetricsCollector

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

DEFAULT_CONFIG = str(Path(__file__).resolve().parent.parent / "config" / "config.yaml")
DEFAULT_TEST_DIR = str(Path(__file__).resolve().parent.parent / "data" / "test_frames")


# ---------------------------------------------------------------------------
# Detector wrappers with graceful fallback
# ---------------------------------------------------------------------------

def _load_retinaface():
    """Try to import and return the RetinaFace detector.

    Returns:
        detect_fn or None  — callable that takes a BGR frame and returns
        a list of [x1, y1, x2, y2] bounding boxes.
    """
    try:
        from retinaface import RetinaFace  # type: ignore

        def detect_faces(frame_bgr: np.ndarray) -> list[list[float]]:
            results = RetinaFace.detect_faces(frame_bgr)
            boxes = []
            if isinstance(results, dict):
                for face_key in results:
                    area = results[face_key].get("facial_area", [])
                    if len(area) == 4:
                        boxes.append([float(v) for v in area])
            return boxes

        return detect_faces
    except ImportError:
        return None


def _load_paddleocr():
    """Try to import and return a PaddleOCR text detector.

    Returns:
        detect_fn or None  — callable that takes a BGR frame and returns
        a list of [x1, y1, x2, y2] axis-aligned bounding boxes.
    """
    try:
        from paddleocr import PaddleOCR  # type: ignore

        ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)

        def detect_text(frame_bgr: np.ndarray) -> list[list[float]]:
            results = ocr.ocr(frame_bgr, cls=True)
            boxes = []
            if results and results[0]:
                for line in results[0]:
                    polygon = line[0]  # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
                    xs = [p[0] for p in polygon]
                    ys = [p[1] for p in polygon]
                    boxes.append([
                        float(min(xs)), float(min(ys)),
                        float(max(xs)), float(max(ys)),
                    ])
            return boxes

        return detect_text
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Matching helpers
# ---------------------------------------------------------------------------

def bbox_iou(a: list, b: list) -> float:
    """IoU between two [x1, y1, x2, y2] boxes."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / max(union, 1e-8)


def match_detections(gt_boxes: list, pred_boxes: list,
                     iou_threshold: float = 0.5) -> tuple[int, int, int]:
    """Greedy IoU-based matching.

    Returns:
        (true_positives, false_positives, false_negatives)
    """
    matched_gt = set()
    tp = 0
    for pred in pred_boxes:
        best_iou = 0.0
        best_idx = -1
        for i, gt in enumerate(gt_boxes):
            if i in matched_gt:
                continue
            iou = bbox_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_idx = i
        if best_iou >= iou_threshold and best_idx >= 0:
            tp += 1
            matched_gt.add(best_idx)

    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp
    return tp, fp, fn


# ---------------------------------------------------------------------------
# Test-set loader
# ---------------------------------------------------------------------------

def load_test_set(test_dir: Path) -> list[dict]:
    """Load image paths and their ground truth annotations.

    Returns:
        List of dicts: {image_path, faces, badges}
    """
    samples = []
    for img_path in sorted(test_dir.iterdir()):
        if not img_path.is_file() or img_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        gt_path = img_path.with_name(img_path.stem + "_gt.json")
        if not gt_path.is_file():
            continue
        with open(gt_path) as f:
            gt = json.load(f)
        samples.append({
            "image_path": img_path,
            "faces": gt.get("faces", []),
            "badges": gt.get("badges", []),
        })
    return samples


# ---------------------------------------------------------------------------
# Evaluation modes
# ---------------------------------------------------------------------------

def evaluate_faces(samples: list[dict], detect_fn) -> dict:
    """Run face detection on all samples and compute recall/precision/F1."""
    collector = MetricsCollector()
    total_tp, total_fp, total_fn = 0, 0, 0
    latencies = []

    for sample in samples:
        frame = cv2.imread(str(sample["image_path"]))
        if frame is None:
            print(f"  Warning: cannot read {sample['image_path'].name}, skipping")
            continue

        t0 = time.perf_counter()
        pred_boxes = detect_fn(frame)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed_ms)

        tp, fp, fn = match_detections(sample["faces"], pred_boxes)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        collector.add("face_latency_ms", elapsed_ms)

    metrics = compute_f1(total_tp, total_fp, total_fn)
    metrics["total_tp"] = total_tp
    metrics["total_fp"] = total_fp
    metrics["total_fn"] = total_fn
    metrics["num_frames"] = len(latencies)
    if latencies:
        arr = np.array(latencies)
        metrics["latency_median_ms"] = float(np.median(arr))
        metrics["latency_p95_ms"] = float(np.percentile(arr, 95))
    return metrics


def evaluate_badges(samples: list[dict], detect_fn) -> dict:
    """Run text detection on all samples and compute badge recall/precision/F1."""
    total_tp, total_fp, total_fn = 0, 0, 0
    latencies = []

    for sample in samples:
        frame = cv2.imread(str(sample["image_path"]))
        if frame is None:
            continue

        t0 = time.perf_counter()
        pred_boxes = detect_fn(frame)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed_ms)

        tp, fp, fn = match_detections(sample["badges"], pred_boxes)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    metrics = compute_f1(total_tp, total_fp, total_fn)
    metrics["total_tp"] = total_tp
    metrics["total_fp"] = total_fp
    metrics["total_fn"] = total_fn
    metrics["num_frames"] = len(latencies)
    if latencies:
        arr = np.array(latencies)
        metrics["latency_median_ms"] = float(np.median(arr))
        metrics["latency_p95_ms"] = float(np.percentile(arr, 95))
    return metrics


def evaluate_latency(samples: list[dict], face_fn, text_fn) -> dict:
    """Measure combined per-frame latency (face + text detection)."""
    latencies = []

    for sample in samples:
        frame = cv2.imread(str(sample["image_path"]))
        if frame is None:
            continue

        t0 = time.perf_counter()
        if face_fn:
            face_fn(frame)
        if text_fn:
            text_fn(frame)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed_ms)

    if not latencies:
        return {"error": "no frames processed"}

    arr = np.array(latencies)
    return {
        "num_frames": len(latencies),
        "median_ms": float(np.median(arr)),
        "mean_ms": float(np.mean(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
    }


# ---------------------------------------------------------------------------
# Pass/fail table
# ---------------------------------------------------------------------------

def print_pass_fail(report: dict, targets: dict):
    """Print a summary table with pass/fail status against config targets."""
    print()
    print(f"{'Metric':<35} {'Value':>10} {'Target':>10} {'Status':>8}")
    print("-" * 67)

    checks = []

    if "faces" in report:
        face_recall = report["faces"].get("recall", 0)
        target = targets.get("face_detection_recall", 0.99)
        status = "PASS" if face_recall >= target else "FAIL"
        checks.append(status)
        print(f"{'Face detection recall':<35} {face_recall:>10.4f} {target:>10.2f} {status:>8}")

        fp_rate = report["faces"]["total_fp"] / max(
            report["faces"]["total_tp"] + report["faces"]["total_fp"], 1)
        fp_target = targets.get("false_positive_rate", 0.05)
        status = "PASS" if fp_rate <= fp_target else "FAIL"
        checks.append(status)
        print(f"{'Face false positive rate':<35} {fp_rate:>10.4f} {'<=' + str(fp_target):>10} {status:>8}")

    if "badges" in report:
        badge_recall = report["badges"].get("recall", 0)
        target = targets.get("badge_detection_recall", 0.98)
        status = "PASS" if badge_recall >= target else "FAIL"
        checks.append(status)
        print(f"{'Badge detection recall':<35} {badge_recall:>10.4f} {target:>10.2f} {status:>8}")

    if "latency" in report:
        p95 = report["latency"].get("p95_ms", 9999)
        lat_target = targets.get("latency_per_frame_ms", 200)
        status = "PASS" if p95 <= lat_target else "FAIL"
        checks.append(status)
        print(f"{'Latency p95 (ms)':<35} {p95:>10.1f} {'<=' + str(lat_target):>10} {status:>8}")

    print("-" * 67)
    overall = "PASS" if all(s == "PASS" for s in checks) else "FAIL"
    print(f"{'OVERALL':<35} {'':>10} {'':>10} {overall:>8}")
    print()
    return overall


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate privacy engine: face/badge detection recall + latency."
    )
    parser.add_argument(
        "--config", type=str, default=DEFAULT_CONFIG,
        help="Path to config YAML"
    )
    parser.add_argument(
        "--test-dir", type=str, default=DEFAULT_TEST_DIR,
        help="Directory containing test frames and *_gt.json files"
    )
    parser.add_argument(
        "--mode", type=str, default="all",
        choices=["faces", "text", "latency", "all"],
        help="Evaluation mode (default: all)"
    )
    parser.add_argument(
        "--iou-threshold", type=float, default=0.5,
        help="IoU threshold for matching (default: 0.5)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to save report JSON (default: eval_report.json next to test dir)"
    )
    args = parser.parse_args()

    # Load config and targets ------------------------------------------------
    config = load_config(args.config)
    targets = config.get("evaluation", {}).get("targets", {})

    # Load test set ----------------------------------------------------------
    test_dir = Path(args.test_dir)
    if not test_dir.is_dir():
        print(f"Error: test directory not found: {test_dir}")
        print("Run data_prep/prepare_test_set.py first to create the test set.")
        sys.exit(1)

    samples = load_test_set(test_dir)
    if not samples:
        print(f"No test samples found in {test_dir}")
        print("Each image needs a matching *_gt.json (e.g. frame_001.jpg + frame_001_gt.json)")
        sys.exit(1)

    print(f"Loaded {len(samples)} test sample(s) from {test_dir}")

    # Load detectors ---------------------------------------------------------
    face_fn = None
    text_fn = None
    run_faces = args.mode in ("faces", "all")
    run_text = args.mode in ("text", "all")
    run_latency = args.mode in ("latency", "all")

    if run_faces or run_latency:
        face_fn = _load_retinaface()
        if face_fn is None:
            print()
            print("WARNING: RetinaFace not installed. Face detection will be skipped.")
            print("  Install with:  pip install retina-face")
            print()

    if run_text or run_latency:
        text_fn = _load_paddleocr()
        if text_fn is None:
            print()
            print("WARNING: PaddleOCR not installed. Text/badge detection will be skipped.")
            print("  Install with:  pip install paddlepaddle paddleocr")
            print()

    # Run evaluation ---------------------------------------------------------
    report: dict = {"num_samples": len(samples)}

    if run_faces and face_fn:
        print("Evaluating face detection ...")
        report["faces"] = evaluate_faces(samples, face_fn)
        print(f"  recall={report['faces']['recall']:.4f}  "
              f"precision={report['faces']['precision']:.4f}  "
              f"f1={report['faces']['f1']:.4f}")

    if run_text and text_fn:
        print("Evaluating badge/text detection ...")
        report["badges"] = evaluate_badges(samples, text_fn)
        print(f"  recall={report['badges']['recall']:.4f}  "
              f"precision={report['badges']['precision']:.4f}  "
              f"f1={report['badges']['f1']:.4f}")

    if run_latency and (face_fn or text_fn):
        print("Measuring combined latency ...")
        report["latency"] = evaluate_latency(samples, face_fn, text_fn)
        print(f"  median={report['latency']['median_ms']:.1f} ms  "
              f"p95={report['latency']['p95_ms']:.1f} ms")

    # Pass/fail table --------------------------------------------------------
    if any(k in report for k in ("faces", "badges", "latency")):
        overall = print_pass_fail(report, targets)
        report["overall"] = overall
    else:
        print("\nNo evaluations were run. Install required packages and retry.")

    # Save report ------------------------------------------------------------
    if args.output:
        report_path = Path(args.output)
    else:
        report_path = test_dir.parent / "eval_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert Path objects for JSON serialization
    serializable = json.loads(json.dumps(report, default=str))
    with open(report_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
