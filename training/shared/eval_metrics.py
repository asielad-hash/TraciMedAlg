"""
Evaluation metrics for all TrackiMed algorithm components.
Implements metrics from TRK-ALGO-001 Section 6.1.
"""

import time
import numpy as np
from collections import defaultdict


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Intersection over Union for binary masks."""
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    return float(intersection / max(union, 1))


def compute_miou(predictions: list, ground_truths: list) -> float:
    """Mean IoU across all mask pairs."""
    ious = [compute_iou(p, g) for p, g in zip(predictions, ground_truths)]
    return float(np.mean(ious)) if ious else 0.0


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def compute_auc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Area Under ROC Curve."""
    sorted_idx = np.argsort(-y_scores)
    y_true = y_true[sorted_idx]
    tps = np.cumsum(y_true)
    fps = np.cumsum(1 - y_true)
    tpr = tps / (tps[-1] + 1e-8)
    fpr = fps / (fps[-1] + 1e-8)
    return float(np.trapz(tpr, fpr))


def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """Average Precision (101-point interpolation)."""
    recall_levels = np.linspace(0, 1, 101)
    interpolated = np.zeros_like(recall_levels)
    for i, r in enumerate(recall_levels):
        mask = recalls >= r
        if mask.any():
            interpolated[i] = precisions[mask].max()
    return float(interpolated.mean())


# ---------------------------------------------------------------------------
# Tracking
# ---------------------------------------------------------------------------

def compute_hota(pred_tracks: dict, gt_tracks: dict) -> dict:
    """HOTA (Higher Order Tracking Accuracy).

    HOTA = sqrt(DetA * AssA)

    Args:
        pred_tracks: {frame_id: [(track_id, bbox), ...]}
        gt_tracks:   {frame_id: [(track_id, bbox), ...]}
    """
    all_frames = set(list(pred_tracks.keys()) + list(gt_tracks.keys()))
    det_scores, ass_scores = [], []

    for frame in all_frames:
        preds = pred_tracks.get(frame, [])
        gts = gt_tracks.get(frame, [])
        if not gts:
            continue
        matched = min(len(preds), len(gts))
        det_scores.append(matched / len(gts))
        if matched > 0:
            ass_scores.append(matched / max(len(preds), len(gts)))

    det_a = float(np.mean(det_scores)) if det_scores else 0
    ass_a = float(np.mean(ass_scores)) if ass_scores else 0
    hota = float(np.sqrt(det_a * ass_a))
    return {"HOTA": hota, "DetA": det_a, "AssA": ass_a}


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def compute_f1(tp: int, fp: int, fn: int) -> dict:
    """Precision, recall, F1 from raw counts."""
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return {"precision": precision, "recall": recall, "f1": f1}


def compute_per_class_f1(y_true: np.ndarray, y_pred: np.ndarray,
                          class_names: list = None) -> dict:
    """Per-class and macro F1."""
    classes = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    results = {}
    f1s = []
    for c in classes:
        tp = ((y_pred == c) & (y_true == c)).sum()
        fp = ((y_pred == c) & (y_true != c)).sum()
        fn = ((y_pred != c) & (y_true == c)).sum()
        m = compute_f1(int(tp), int(fp), int(fn))
        name = class_names[c] if class_names and c < len(class_names) else f"class_{c}"
        results[name] = m
        f1s.append(m["f1"])
    results["macro_f1"] = float(np.mean(f1s))
    return results


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                      num_classes: int = None) -> np.ndarray:
    """Compute confusion matrix."""
    if num_classes is None:
        num_classes = max(y_true.max(), y_pred.max()) + 1
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


# ---------------------------------------------------------------------------
# Count reconciliation
# ---------------------------------------------------------------------------

def compute_count_accuracy(predicted: list[int], actual: list[int]) -> dict:
    """Count accuracy across procedures."""
    correct = sum(1 for p, a in zip(predicted, actual) if p == a)
    total = len(actual)
    return {
        "accuracy": correct / max(total, 1),
        "correct": correct,
        "total": total,
    }


# ---------------------------------------------------------------------------
# ASR
# ---------------------------------------------------------------------------

def compute_wer(predictions: list[str], references: list[str]) -> float:
    """Word Error Rate using edit distance."""
    total_errors = 0
    total_words = 0
    for pred, ref in zip(predictions, references):
        pred_w = pred.lower().split()
        ref_w = ref.lower().split()
        total_words += len(ref_w)
        m, n = len(ref_w), len(pred_w)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if ref_w[i - 1] == pred_w[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
        total_errors += dp[m][n]
    return total_errors / max(total_words, 1)


# ---------------------------------------------------------------------------
# Latency
# ---------------------------------------------------------------------------

def measure_latency(fn, *args, n_runs: int = 10, **kwargs) -> dict:
    """Measure function execution latency."""
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        times.append((time.perf_counter() - t0) * 1000)
    arr = np.array(times)
    return {
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "mean_ms": float(np.mean(arr)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

class MetricsCollector:
    """Accumulate metrics across frames/videos and generate a report."""

    def __init__(self):
        self.metrics = defaultdict(list)

    def add(self, name: str, value: float):
        self.metrics[name].append(value)

    def summary(self) -> dict:
        result = {}
        for name, values in self.metrics.items():
            arr = np.array(values)
            result[name] = {
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "min": float(arr.min()),
                "max": float(arr.max()),
                "count": len(values),
            }
        return result

    def print_report(self, targets: dict = None):
        """Print formatted report with optional pass/fail against targets."""
        summary = self.summary()
        print(f"\n{'Metric':<30} {'Mean':>10} {'Std':>10} {'Target':>10} {'Status':>8}")
        print("-" * 72)
        for name, stats in summary.items():
            target = targets.get(name) if targets else None
            target_str = f"{target:.3f}" if target is not None else "—"
            if target is not None:
                status = "PASS" if stats["mean"] >= target else "FAIL"
            else:
                status = "—"
            print(f"{name:<30} {stats['mean']:>10.4f} {stats['std']:>10.4f} "
                  f"{target_str:>10} {status:>8}")
