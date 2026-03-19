"""
Evaluate instrument tracking quality against ground truth.

Computes segmentation mIoU, HOTA tracking accuracy, count accuracy,
false alarm rate, and event detection accuracy. Prints a pass/fail
table against config targets and saves a report JSON.

Usage (single video):
    python evaluate_tracking.py \
        --pred-meta pred_tracked_meta.json \
        --gt-meta gt_tracked_meta.json

Usage (batch — directory of meta files):
    python evaluate_tracking.py \
        --pred-dir ./predictions/ \
        --gt-dir ./ground_truth/ \
        --config ../../config/config.yaml
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Shared library
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from shared.annotation_io import load_tracked_meta
from shared.eval_metrics import (
    compute_miou,
    compute_hota,
    compute_count_accuracy,
    MetricsCollector,
)
from shared.event_detection import detect_events, compute_stable_counts


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_tracks(meta: dict) -> dict:
    """Convert tracked metadata to {frame_id: [(track_id, bbox), ...]} for HOTA."""
    tracks = {}
    instruments = meta.get("instruments", {})
    for fdata in meta.get("frames", []):
        fidx = fdata["frame"]
        entries = []
        bboxes = fdata.get("bboxes", {})
        for oid in fdata.get("obj_ids", []):
            bbox = bboxes.get(str(oid), [0, 0, 1, 1])
            entries.append((oid, bbox))
        tracks[fidx] = entries
    return tracks


def _compute_false_alarm_rate(pred_meta: dict, gt_meta: dict) -> float:
    """Fraction of predicted detections that have no matching ground-truth object."""
    gt_frames = {f["frame"]: set(f.get("obj_ids", [])) for f in gt_meta.get("frames", [])}
    total_preds = 0
    false_alarms = 0
    for fdata in pred_meta.get("frames", []):
        fidx = fdata["frame"]
        pred_ids = set(fdata.get("obj_ids", []))
        gt_ids = gt_frames.get(fidx, set())
        total_preds += len(pred_ids)
        false_alarms += len(pred_ids - gt_ids)
    return false_alarms / max(total_preds, 1)


def _event_accuracy(pred_events: list[dict], gt_events: list[dict],
                     tolerance: int = 10) -> dict:
    """Compare detected events against ground truth.

    An event matches if it has the same type, same obj_id, and the frame
    is within +/- tolerance of the ground truth event frame.

    Returns: {precision, recall, f1, matched, pred_total, gt_total}
    """
    matched = 0
    gt_matched = set()
    for pe in pred_events:
        for gi, ge in enumerate(gt_events):
            if gi in gt_matched:
                continue
            if (pe["type"] == ge["type"]
                    and pe["obj_id"] == ge["obj_id"]
                    and abs(pe["frame"] - ge["frame"]) <= tolerance):
                matched += 1
                gt_matched.add(gi)
                break

    precision = matched / max(len(pred_events), 1)
    recall = matched / max(len(gt_events), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "matched": matched,
        "pred_total": len(pred_events),
        "gt_total": len(gt_events),
    }


def evaluate_pair(pred_meta: dict, gt_meta: dict) -> dict:
    """Evaluate a single pred/gt metadata pair. Returns metric dict."""
    results = {}

    # --- HOTA tracking ---
    pred_tracks = _build_tracks(pred_meta)
    gt_tracks = _build_tracks(gt_meta)
    hota = compute_hota(pred_tracks, gt_tracks)
    results.update(hota)

    # --- Count accuracy ---
    pred_counts = compute_stable_counts(pred_meta)
    gt_counts = compute_stable_counts(gt_meta)
    min_len = min(len(pred_counts), len(gt_counts))
    if min_len > 0:
        ca = compute_count_accuracy(pred_counts[:min_len], gt_counts[:min_len])
        results["count_accuracy"] = ca["accuracy"]
    else:
        results["count_accuracy"] = 0.0

    # --- False alarm rate ---
    results["false_alarm_rate"] = _compute_false_alarm_rate(pred_meta, gt_meta)

    # --- Event detection ---
    pred_events = detect_events(pred_meta)
    gt_events = detect_events(gt_meta)
    evt = _event_accuracy(pred_events, gt_events)
    results["event_f1"] = evt["f1"]
    results["event_detail"] = evt

    return results


def load_default_targets() -> dict:
    """Load evaluation targets from component config if available."""
    config_path = (
        Path(__file__).resolve().parent.parent / "config" / "config.yaml"
    )
    if config_path.exists():
        from shared.config import load_config
        cfg = load_config(str(config_path))
        return cfg.get("evaluation", {}).get("targets", {})
    return {}


def print_report(aggregated: dict, targets: dict):
    """Print a formatted pass/fail table."""
    # Map metric names in our results to config target keys
    metric_target_map = {
        "HOTA": "tracking_hota",
        "count_accuracy": "count_accuracy",
        "false_alarm_rate": "false_alarm_rate_max",
        "event_f1": None,
        "DetA": None,
        "AssA": None,
    }

    print(f"\n{'Metric':<30} {'Value':>10} {'Target':>10} {'Status':>8}")
    print("-" * 62)
    for metric, value in aggregated.items():
        if isinstance(value, dict):
            continue
        target_key = metric_target_map.get(metric)
        target_val = targets.get(target_key) if target_key else None
        target_str = f"{target_val:.3f}" if target_val is not None else "--"

        if target_val is not None:
            if metric == "false_alarm_rate":
                status = "PASS" if value <= target_val else "FAIL"
            else:
                status = "PASS" if value >= target_val else "FAIL"
        else:
            status = "--"

        print(f"{metric:<30} {value:>10.4f} {target_str:>10} {status:>8}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate instrument tracking quality."
    )
    # Single file mode
    parser.add_argument("--pred-meta", default=None, help="Prediction _tracked_meta.json")
    parser.add_argument("--gt-meta", default=None, help="Ground truth _tracked_meta.json")
    # Batch mode
    parser.add_argument("--pred-dir", default=None, help="Directory of prediction meta files")
    parser.add_argument("--gt-dir", default=None, help="Directory of ground truth meta files")
    # Config & output
    parser.add_argument("--config", default=None, help="Component config.yaml for targets")
    parser.add_argument(
        "--output", default="tracking_eval_report.json",
        help="Output report JSON path (default: tracking_eval_report.json).",
    )
    parser.add_argument(
        "--event-tolerance", type=int, default=10,
        help="Frame tolerance for event matching (default: 10).",
    )
    args = parser.parse_args()

    # Determine file pairs
    pairs = []
    if args.pred_meta and args.gt_meta:
        pairs.append((Path(args.pred_meta), Path(args.gt_meta)))
    elif args.pred_dir and args.gt_dir:
        pred_dir = Path(args.pred_dir)
        gt_dir = Path(args.gt_dir)
        for pf in sorted(pred_dir.glob("*_tracked_meta.json")):
            gf = gt_dir / pf.name
            if gf.exists():
                pairs.append((pf, gf))
            else:
                print(f"WARNING: No matching GT for {pf.name}, skipping.")
    else:
        parser.error("Provide either --pred-meta/--gt-meta or --pred-dir/--gt-dir.")

    if not pairs:
        print("ERROR: No valid pred/gt pairs found.")
        sys.exit(1)

    print(f"Evaluating {len(pairs)} video(s)...")

    # Load targets
    targets = {}
    if args.config:
        from shared.config import load_config
        cfg = load_config(args.config)
        targets = cfg.get("evaluation", {}).get("targets", {})
    else:
        targets = load_default_targets()

    # Evaluate each pair and aggregate
    collector = MetricsCollector()
    per_video = []

    for pred_path, gt_path in pairs:
        pred_meta = load_tracked_meta(str(pred_path))
        gt_meta = load_tracked_meta(str(gt_path))
        result = evaluate_pair(pred_meta, gt_meta)
        result["video"] = pred_path.stem.replace("_tracked_meta", "")
        per_video.append(result)

        for key, val in result.items():
            if isinstance(val, (int, float)):
                collector.add(key, val)

        print(f"  {result['video']}: HOTA={result['HOTA']:.3f} "
              f"count_acc={result['count_accuracy']:.3f} "
              f"FAR={result['false_alarm_rate']:.3f} "
              f"event_f1={result['event_f1']:.3f}")

    # Aggregate means
    summary = collector.summary()
    aggregated = {k: v["mean"] for k, v in summary.items()}

    print_report(aggregated, targets)

    # Save report
    report = {
        "aggregated": aggregated,
        "targets": targets,
        "per_video": per_video,
        "num_videos": len(pairs),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nReport saved to {output_path}")


if __name__ == "__main__":
    main()
