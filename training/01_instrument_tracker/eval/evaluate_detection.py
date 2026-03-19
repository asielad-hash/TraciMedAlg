"""
Evaluate instrument detection AUC-ROC and per-class F1.

Reads prediction and ground-truth files (JSON or CSV) containing
per-image class labels and confidence scores, then computes AUC-ROC,
precision, recall, and F1 per class. Prints pass/fail against a
configurable target AUC (default >= 0.90).

Expected JSON format for --predictions and --ground-truth:
    [
        {"image": "frame_000001.jpg", "class": 0, "confidence": 0.95},
        {"image": "frame_000001.jpg", "class": 2, "confidence": 0.80},
        ...
    ]

Ground-truth entries should have confidence=1.0 (or the field can be
omitted, in which case presence is inferred from the entry existing).

Usage:
    python evaluate_detection.py \
        --predictions preds.json \
        --ground-truth gt.json \
        --config ../../config/config.yaml
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Shared library
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from shared.eval_metrics import compute_auc, compute_per_class_f1, MetricsCollector


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_json_or_csv(filepath: str) -> list[dict]:
    """Load predictions/ground-truth from JSON or CSV."""
    p = Path(filepath)
    if p.suffix == ".csv":
        rows = []
        with open(p, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append({
                    "image": row["image"],
                    "class": int(row["class"]),
                    "confidence": float(row.get("confidence", 1.0)),
                })
        return rows
    else:
        with open(p) as f:
            data = json.load(f)
        # Normalise: ensure confidence exists
        for entry in data:
            entry.setdefault("confidence", 1.0)
            entry["class"] = int(entry["class"])
            entry["confidence"] = float(entry["confidence"])
        return data


def _collect_images(entries: list[dict]) -> set[str]:
    """Return unique image names."""
    return {e["image"] for e in entries}


def _collect_classes(entries: list[dict]) -> list[int]:
    """Return sorted unique class IDs."""
    return sorted({e["class"] for e in entries})


def _build_class_names(config: dict | None) -> list[str] | None:
    """Extract ordered class names from config taxonomy."""
    if config is None:
        return None
    taxonomy = config.get("dataset", {}).get("instrument_taxonomy", {})
    if not taxonomy:
        return None
    names = []
    for priority in ("critical", "high", "medium", "low"):
        items = taxonomy.get(priority, [])
        names.extend(items)
    return names if names else None


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate_detection(
    pred_entries: list[dict],
    gt_entries: list[dict],
    target_auc: float = 0.90,
    class_names: list[str] | None = None,
) -> dict:
    """Run full detection evaluation.

    For each class, computes:
      - AUC-ROC (binary: is class present in image or not)
      - Precision, Recall, F1 at confidence >= 0.5

    Returns dict with per-class results and overall summary.
    """
    all_images = sorted(_collect_images(pred_entries) | _collect_images(gt_entries))
    all_classes = sorted(
        set(_collect_classes(pred_entries)) | set(_collect_classes(gt_entries))
    )

    # Build per-image, per-class lookup
    # gt_map[image][class] = 1/0
    gt_map = {}
    for e in gt_entries:
        gt_map.setdefault(e["image"], {})[e["class"]] = 1

    # pred_map[image][class] = max confidence
    pred_map = {}
    for e in pred_entries:
        img = e["image"]
        cls = e["class"]
        conf = e["confidence"]
        if img not in pred_map:
            pred_map[img] = {}
        pred_map[img][cls] = max(pred_map[img].get(cls, 0.0), conf)

    per_class = {}
    y_true_all = []
    y_pred_all = []

    for cls in all_classes:
        y_true = []
        y_scores = []

        for img in all_images:
            gt_present = gt_map.get(img, {}).get(cls, 0)
            pred_conf = pred_map.get(img, {}).get(cls, 0.0)
            y_true.append(gt_present)
            y_scores.append(pred_conf)

        y_true_np = np.array(y_true, dtype=np.int32)
        y_scores_np = np.array(y_scores, dtype=np.float64)

        # AUC
        if y_true_np.sum() > 0 and y_true_np.sum() < len(y_true_np):
            auc = compute_auc(y_true_np, y_scores_np)
        else:
            auc = float("nan")

        # P/R/F1 at threshold 0.5
        y_pred_bin = (y_scores_np >= 0.5).astype(np.int32)
        tp = int(((y_pred_bin == 1) & (y_true_np == 1)).sum())
        fp = int(((y_pred_bin == 1) & (y_true_np == 0)).sum())
        fn = int(((y_pred_bin == 0) & (y_true_np == 1)).sum())

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

        name = class_names[cls] if class_names and cls < len(class_names) else f"class_{cls}"
        status = "PASS" if (not np.isnan(auc) and auc >= target_auc) else "FAIL"

        per_class[name] = {
            "class_id": cls,
            "auc_roc": round(auc, 4) if not np.isnan(auc) else None,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn,
            "n_gt_positive": int(y_true_np.sum()),
            "status": status,
        }

        y_true_all.extend(y_true)
        y_pred_all.extend(y_pred_bin.tolist())

    # Macro averages
    valid_aucs = [v["auc_roc"] for v in per_class.values() if v["auc_roc"] is not None]
    macro_auc = float(np.mean(valid_aucs)) if valid_aucs else float("nan")

    y_true_all_np = np.array(y_true_all, dtype=np.int32)
    y_pred_all_np = np.array(y_pred_all, dtype=np.int32)
    class_f1 = compute_per_class_f1(y_true_all_np, y_pred_all_np)

    overall_status = "PASS" if (not np.isnan(macro_auc) and macro_auc >= target_auc) else "FAIL"

    return {
        "per_class": per_class,
        "macro_auc_roc": round(macro_auc, 4) if not np.isnan(macro_auc) else None,
        "macro_f1": round(class_f1.get("macro_f1", 0.0), 4),
        "target_auc": target_auc,
        "overall_status": overall_status,
        "num_images": len(all_images),
        "num_classes": len(all_classes),
    }


def print_report(results: dict):
    """Print formatted detection evaluation report."""
    target = results["target_auc"]
    print(f"\n{'Class':<30} {'AUC':>8} {'Prec':>8} {'Rec':>8} {'F1':>8} "
          f"{'GT+':>6} {'Status':>8}")
    print("-" * 80)

    for name, metrics in results["per_class"].items():
        auc_str = f"{metrics['auc_roc']:.4f}" if metrics["auc_roc"] is not None else "N/A"
        print(f"{name:<30} {auc_str:>8} {metrics['precision']:>8.4f} "
              f"{metrics['recall']:>8.4f} {metrics['f1']:>8.4f} "
              f"{metrics['n_gt_positive']:>6} {metrics['status']:>8}")

    print("-" * 80)
    macro_auc_str = (f"{results['macro_auc_roc']:.4f}"
                     if results["macro_auc_roc"] is not None else "N/A")
    print(f"{'MACRO':<30} {macro_auc_str:>8} {'':>8} {'':>8} "
          f"{results['macro_f1']:>8.4f} {'':>6} {results['overall_status']:>8}")
    print(f"\nTarget AUC >= {target:.2f}  |  "
          f"Images: {results['num_images']}  |  Classes: {results['num_classes']}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate instrument detection AUC-ROC per class."
    )
    parser.add_argument(
        "--predictions", required=True,
        help="Prediction file (JSON or CSV) with image, class, confidence.",
    )
    parser.add_argument(
        "--ground-truth", required=True,
        help="Ground truth file (JSON or CSV) with image, class.",
    )
    parser.add_argument(
        "--config", default=None,
        help="Component config.yaml for class names and target AUC.",
    )
    parser.add_argument(
        "--target-auc", type=float, default=None,
        help="Target AUC-ROC for pass/fail (overrides config, default: 0.90).",
    )
    parser.add_argument(
        "--output", default="detection_eval_report.json",
        help="Output report JSON path (default: detection_eval_report.json).",
    )
    args = parser.parse_args()

    # Load data
    pred_entries = _load_json_or_csv(args.predictions)
    gt_entries = _load_json_or_csv(args.ground_truth)
    print(f"Loaded {len(pred_entries)} predictions, {len(gt_entries)} ground-truth entries.")

    # Load config for class names and target
    config = None
    class_names = None
    target_auc = 0.90

    if args.config:
        from shared.config import load_config
        config = load_config(args.config)
        class_names = _build_class_names(config)
        cfg_target = config.get("evaluation", {}).get("targets", {}).get("detection_auc")
        if cfg_target is not None:
            target_auc = cfg_target
    else:
        # Try loading default config
        default_cfg = (
            Path(__file__).resolve().parent.parent / "config" / "config.yaml"
        )
        if default_cfg.exists():
            from shared.config import load_config
            config = load_config(str(default_cfg))
            class_names = _build_class_names(config)
            cfg_target = config.get("evaluation", {}).get("targets", {}).get("detection_auc")
            if cfg_target is not None:
                target_auc = cfg_target

    # CLI override takes precedence
    if args.target_auc is not None:
        target_auc = args.target_auc

    # Evaluate
    results = evaluate_detection(
        pred_entries, gt_entries,
        target_auc=target_auc,
        class_names=class_names,
    )

    print_report(results)

    # Save report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nReport saved to {output_path}")


if __name__ == "__main__":
    main()
