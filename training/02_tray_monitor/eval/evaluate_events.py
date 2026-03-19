"""
Evaluate tray-monitor event classification on test clips.

Loads predicted and ground-truth event labels for each test clip, then
computes:
  - Event detection F1 (did we detect an event vs. NO_EVENT?)
  - TAKE_OUT / PUT_BACK classification accuracy
  - False positive rate (NO_EVENT clips misclassified as events)
  - Per-class F1 and confusion matrix

Results are compared against targets from config/config.yaml:
    event_detection_f1        >= 0.85
    event_classification_accuracy >= 0.90
    false_positive_rate_max   <  0.10

Prediction and ground-truth directories should contain one JSON per clip:
    clip_000000.json  ->  {"clip_id": "clip_000000", "event_type": "TAKE_OUT"}

Usage:
    python evaluate_events.py --pred-dir results/predictions --gt-dir data --output results/eval_report.json
    python evaluate_events.py --config ../config/config.yaml --pred-dir results/predictions --gt-dir data
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Shared library import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from shared.config import load_config
from shared.eval_metrics import compute_per_class_f1, confusion_matrix, compute_f1

DEFAULT_CONFIG = str(Path(__file__).resolve().parent.parent / "config" / "config.yaml")

EVENT_TYPES = ["TAKE_OUT", "PUT_BACK", "NO_EVENT"]
EVENT_TO_IDX = {name: idx for idx, name in enumerate(EVENT_TYPES)}


def load_clip_labels(directory: Path, key: str = "event_type") -> dict:
    """Load per-clip label JSONs from a directory.

    Supports two formats:
      1) Individual clip_XXXXXX.json files, each with {clip_id, event_type}
      2) A single test.json file with a list of {clip_id, event_type, ...}

    Returns: {clip_id: event_type_string}
    """
    labels = {}

    # Try individual clip JSONs first
    clip_files = sorted(directory.glob("clip_*.json"))
    if clip_files:
        for f in clip_files:
            with open(f) as fh:
                data = json.load(fh)
            clip_id = data.get("clip_id", f.stem)
            labels[clip_id] = data[key]
        return labels

    # Try consolidated split files (test.json, val.json, etc.)
    for split_file in ["test.json", "val.json"]:
        candidate = directory / split_file
        if candidate.is_file():
            with open(candidate) as fh:
                data = json.load(fh)
            if isinstance(data, list):
                for item in data:
                    clip_id = item.get("clip_id", item.get("id"))
                    if clip_id:
                        labels[clip_id] = item[key]
            break

    return labels


def compute_event_detection_metrics(y_true: np.ndarray,
                                     y_pred: np.ndarray) -> dict:
    """Compute binary event-detection metrics (event vs NO_EVENT).

    Converts 3-class labels to binary: event (TAKE_OUT or PUT_BACK) = 1,
    NO_EVENT = 0.

    Returns dict with precision, recall, f1.
    """
    no_event_idx = EVENT_TO_IDX["NO_EVENT"]

    gt_binary = (y_true != no_event_idx).astype(int)
    pred_binary = (y_pred != no_event_idx).astype(int)

    tp = int(((pred_binary == 1) & (gt_binary == 1)).sum())
    fp = int(((pred_binary == 1) & (gt_binary == 0)).sum())
    fn = int(((pred_binary == 0) & (gt_binary == 1)).sum())

    return compute_f1(tp, fp, fn)


def compute_classification_accuracy(y_true: np.ndarray,
                                     y_pred: np.ndarray) -> dict:
    """Compute TAKE_OUT vs PUT_BACK classification accuracy.

    Only considers clips where ground truth is an actual event
    (not NO_EVENT) and the prediction is also an event.
    """
    no_event_idx = EVENT_TO_IDX["NO_EVENT"]

    # Filter to only event clips
    event_mask = y_true != no_event_idx
    if event_mask.sum() == 0:
        return {"accuracy": 0.0, "correct": 0, "total": 0}

    gt_events = y_true[event_mask]
    pred_events = y_pred[event_mask]

    correct = int((gt_events == pred_events).sum())
    total = int(event_mask.sum())

    return {
        "accuracy": correct / max(total, 1),
        "correct": correct,
        "total": total,
    }


def compute_false_positive_rate(y_true: np.ndarray,
                                 y_pred: np.ndarray) -> dict:
    """Compute false positive rate: fraction of NO_EVENT clips predicted as events."""
    no_event_idx = EVENT_TO_IDX["NO_EVENT"]

    no_event_mask = y_true == no_event_idx
    n_no_event = int(no_event_mask.sum())

    if n_no_event == 0:
        return {"fp_rate": 0.0, "false_positives": 0, "total_negatives": 0}

    false_positives = int((y_pred[no_event_mask] != no_event_idx).sum())

    return {
        "fp_rate": false_positives / n_no_event,
        "false_positives": false_positives,
        "total_negatives": n_no_event,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate tray-monitor event classification on test clips."
    )
    parser.add_argument(
        "--config", type=str, default=DEFAULT_CONFIG,
        help="Path to config YAML (default: ../config/config.yaml)"
    )
    parser.add_argument(
        "--pred-dir", type=str, required=True,
        help="Directory containing prediction JSONs (clip_*.json or test.json)"
    )
    parser.add_argument(
        "--gt-dir", type=str, required=True,
        help="Directory containing ground-truth JSONs (clip_*.json or test.json)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output path for evaluation report JSON (optional)"
    )
    args = parser.parse_args()

    # Load config and targets
    config = load_config(args.config)
    eval_cfg = config.get("evaluation", {})
    targets = eval_cfg.get("targets", {})

    target_f1 = targets.get("event_detection_f1", 0.85)
    target_acc = targets.get("event_classification_accuracy", 0.90)
    target_fp_max = targets.get("false_positive_rate_max", 0.10)

    # Load predictions and ground truth
    pred_dir = Path(args.pred_dir)
    gt_dir = Path(args.gt_dir)

    if not pred_dir.exists():
        print(f"Error: prediction directory not found: {pred_dir}")
        sys.exit(1)
    if not gt_dir.exists():
        print(f"Error: ground-truth directory not found: {gt_dir}")
        sys.exit(1)

    pred_labels = load_clip_labels(pred_dir, key="event_type")
    gt_labels = load_clip_labels(gt_dir, key="event_type")

    if not pred_labels:
        print(f"Error: no prediction labels found in {pred_dir}")
        sys.exit(1)
    if not gt_labels:
        print(f"Error: no ground-truth labels found in {gt_dir}")
        sys.exit(1)

    # Match clips present in both predictions and ground truth
    common_ids = sorted(set(pred_labels.keys()) & set(gt_labels.keys()))
    if not common_ids:
        print("Error: no matching clip IDs between predictions and ground truth")
        print(f"  Prediction IDs (sample): {list(pred_labels.keys())[:5]}")
        print(f"  Ground truth IDs (sample): {list(gt_labels.keys())[:5]}")
        sys.exit(1)

    n_pred_only = len(set(pred_labels.keys()) - set(gt_labels.keys()))
    n_gt_only = len(set(gt_labels.keys()) - set(pred_labels.keys()))

    print(f"Matched {len(common_ids)} clips")
    if n_pred_only > 0:
        print(f"  Warning: {n_pred_only} prediction clips have no ground truth")
    if n_gt_only > 0:
        print(f"  Warning: {n_gt_only} ground-truth clips have no prediction")
    print()

    # Convert to numpy arrays
    y_true = np.array([EVENT_TO_IDX[gt_labels[cid]] for cid in common_ids])
    y_pred = np.array([EVENT_TO_IDX[pred_labels[cid]] for cid in common_ids])

    # --- Compute metrics ---

    # 1. Event detection F1 (binary: event vs NO_EVENT)
    det_metrics = compute_event_detection_metrics(y_true, y_pred)

    # 2. TAKE_OUT vs PUT_BACK classification accuracy
    cls_metrics = compute_classification_accuracy(y_true, y_pred)

    # 3. False positive rate
    fp_metrics = compute_false_positive_rate(y_true, y_pred)

    # 4. Per-class F1
    per_class = compute_per_class_f1(y_true, y_pred, class_names=EVENT_TYPES)

    # 5. Confusion matrix
    cm = confusion_matrix(y_true, y_pred, num_classes=len(EVENT_TYPES))

    # --- Print results ---
    print("=" * 70)
    print("TRAY MONITOR EVENT EVALUATION REPORT")
    print("=" * 70)
    print()

    # Event detection
    det_pass = det_metrics["f1"] >= target_f1
    print(f"Event Detection F1:           {det_metrics['f1']:.4f}  "
          f"(target >= {target_f1})  {'PASS' if det_pass else 'FAIL'}")
    print(f"  Precision: {det_metrics['precision']:.4f}  "
          f"Recall: {det_metrics['recall']:.4f}")
    print()

    # Classification accuracy
    acc_pass = cls_metrics["accuracy"] >= target_acc
    print(f"Classification Accuracy:      {cls_metrics['accuracy']:.4f}  "
          f"(target >= {target_acc})  {'PASS' if acc_pass else 'FAIL'}")
    print(f"  Correct: {cls_metrics['correct']}/{cls_metrics['total']} event clips")
    print()

    # False positive rate
    fp_pass = fp_metrics["fp_rate"] < target_fp_max
    print(f"False Positive Rate:          {fp_metrics['fp_rate']:.4f}  "
          f"(target < {target_fp_max})  {'PASS' if fp_pass else 'FAIL'}")
    print(f"  FP: {fp_metrics['false_positives']}/{fp_metrics['total_negatives']} "
          f"NO_EVENT clips")
    print()

    # Per-class F1
    print("Per-Class F1:")
    for cls_name in EVENT_TYPES:
        if cls_name in per_class:
            m = per_class[cls_name]
            print(f"  {cls_name:<15} P={m['precision']:.4f}  "
                  f"R={m['recall']:.4f}  F1={m['f1']:.4f}")
    print(f"  {'Macro F1':<15} {per_class.get('macro_f1', 0):.4f}")
    print()

    # Confusion matrix
    print("Confusion Matrix:")
    header = "              " + "  ".join(f"{n:>10}" for n in EVENT_TYPES)
    print(header)
    for i, row_name in enumerate(EVENT_TYPES):
        row_vals = "  ".join(f"{cm[i, j]:>10d}" for j in range(len(EVENT_TYPES)))
        print(f"  {row_name:<12} {row_vals}")
    print()

    # Overall verdict
    all_pass = det_pass and acc_pass and fp_pass
    verdict = "ALL TARGETS MET" if all_pass else "TARGETS NOT MET"
    print(f"{'=' * 70}")
    print(f"VERDICT: {verdict}")
    print(f"{'=' * 70}")

    # --- Save report ---
    report = {
        "num_clips_evaluated": len(common_ids),
        "event_detection": det_metrics,
        "event_detection_pass": det_pass,
        "classification": cls_metrics,
        "classification_pass": acc_pass,
        "false_positive": fp_metrics,
        "false_positive_pass": fp_pass,
        "per_class_f1": per_class,
        "confusion_matrix": cm.tolist(),
        "confusion_labels": EVENT_TYPES,
        "targets": {
            "event_detection_f1": target_f1,
            "event_classification_accuracy": target_acc,
            "false_positive_rate_max": target_fp_max,
        },
        "all_pass": all_pass,
    }

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to {output_path}")

    # Exit with non-zero status if targets not met
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
