"""
Evaluate surgical workflow models: phase recognition and idle detection.

Phase recognition metrics:
    - Per-class F1 score (target from config: 0.88)
    - Phase boundary accuracy within +/-30 seconds

Idle detection metrics:
    - Precision and recall (targets: 0.90 / 0.85)
    - Duration error percentage (target: 10%)

Loads trained models, runs inference on the test split, computes metrics,
and prints a pass/fail report against config targets.

Usage:
    python evaluate.py --config ../config/config.yaml \
        --data-dir ../data/splits --mode all
    python evaluate.py --config ../config/config.yaml \
        --data-dir ../data/splits --mode phase
    python evaluate.py --config ../config/config.yaml \
        --data-dir ../data/splits --mode idle
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

# Shared library import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from shared.config import load_config
from shared.eval_metrics import compute_per_class_f1, compute_f1

DEFAULT_CONFIG = str(Path(__file__).resolve().parent.parent / "config" / "config.yaml")


# ---------------------------------------------------------------------------
# Phase evaluation
# ---------------------------------------------------------------------------

def load_phase_model(checkpoint_path: str, device: torch.device):
    """Load a trained PhaseClassifier from checkpoint.

    Imports the model class from the training script to avoid duplication.
    """
    # Import model definition
    train_dir = Path(__file__).resolve().parent.parent / "train"
    sys.path.insert(0, str(train_dir))
    from train_phase import PhaseClassifier

    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    phases = ckpt["phases"]
    model = PhaseClassifier(num_phases=len(phases))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, phases


def evaluate_phase_clips(model, test_manifest: str, clip_dir: str,
                         phases: list, device: torch.device) -> dict:
    """Run phase inference on test clips and compute metrics.

    Returns dict with per-class F1, macro F1, and boundary accuracy.
    """
    import cv2

    with open(test_manifest) as f:
        manifest = json.load(f)

    clip_dir = Path(clip_dir)
    all_true = []
    all_pred = []

    RESIZE = (224, 224)
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    for fn in manifest["files"]:
        fp = clip_dir / fn
        if not fp.exists():
            continue
        data = np.load(str(fp))
        label = int(data["majority_phase"])

        # Preprocess frames
        frames = data["frames"]
        processed = []
        for frame in frames:
            frame = cv2.resize(frame, RESIZE)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.astype(np.float32) / 255.0
            frame = (frame - MEAN) / STD
            processed.append(frame.transpose(2, 0, 1))

        x = torch.from_numpy(np.stack(processed)[np.newaxis])  # (1, T, 3, H, W)
        x = x.to(device)
        with torch.no_grad():
            logits = model(x)
        pred = logits.argmax(dim=1).item()

        all_true.append(label)
        all_pred.append(pred)

    if not all_true:
        return {"error": "no test clips found"}

    y_true = np.array(all_true)
    y_pred = np.array(all_pred)

    # Per-class F1
    f1_results = compute_per_class_f1(y_true, y_pred, class_names=phases)

    return f1_results


def evaluate_phase_boundaries(test_manifest: str, clip_dir: str,
                              model, phases: list, device: torch.device,
                              fps: float = 30.0) -> dict:
    """Evaluate phase boundary detection accuracy.

    Groups clips by video, sorts by start frame, detects predicted
    phase transitions, and compares against ground truth transitions.

    Returns boundary accuracy metrics.
    """
    import cv2

    with open(test_manifest) as f:
        manifest = json.load(f)

    clip_dir = Path(clip_dir)
    RESIZE = (224, 224)
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    # Collect per-video predictions
    video_clips = {}
    for fn in manifest["files"]:
        fp = clip_dir / fn
        if not fp.exists():
            continue
        data = np.load(str(fp), allow_pickle=True)
        stem = str(data.get("video_stem", "unknown"))
        start = int(data.get("start_frame", 0))
        gt_label = int(data["majority_phase"])

        frames = data["frames"]
        processed = []
        for frame in frames:
            frame = cv2.resize(frame, RESIZE)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.astype(np.float32) / 255.0
            frame = (frame - MEAN) / STD
            processed.append(frame.transpose(2, 0, 1))

        x = torch.from_numpy(np.stack(processed)[np.newaxis]).to(device)
        with torch.no_grad():
            pred = model(x).argmax(dim=1).item()

        video_clips.setdefault(stem, []).append({
            "start_frame": start, "gt": gt_label, "pred": pred,
        })

    if not video_clips:
        return {"boundary_accuracy": 0.0, "n_boundaries": 0}

    # Detect boundaries per video
    boundary_errors = []
    for stem, clips in video_clips.items():
        clips.sort(key=lambda c: c["start_frame"])
        gt_boundaries = []
        pred_boundaries = []
        for i in range(1, len(clips)):
            if clips[i]["gt"] != clips[i - 1]["gt"]:
                gt_boundaries.append(clips[i]["start_frame"])
            if clips[i]["pred"] != clips[i - 1]["pred"]:
                pred_boundaries.append(clips[i]["start_frame"])

        # Match predicted boundaries to nearest GT boundary
        for pb in pred_boundaries:
            if gt_boundaries:
                nearest = min(gt_boundaries, key=lambda gb: abs(gb - pb))
                error_sec = abs(pb - nearest) / fps
                boundary_errors.append(error_sec)

    if not boundary_errors:
        return {"boundary_accuracy": 1.0, "mean_error_sec": 0.0,
                "n_boundaries": 0}

    errors = np.array(boundary_errors)
    return {
        "boundary_accuracy": float((errors <= 30.0).mean()),
        "mean_error_sec": float(errors.mean()),
        "median_error_sec": float(np.median(errors)),
        "n_boundaries": len(errors),
    }


# ---------------------------------------------------------------------------
# Idle evaluation
# ---------------------------------------------------------------------------

def load_idle_model(checkpoint_path: str, device: torch.device):
    """Load a trained IdleDetectorMLP from checkpoint."""
    train_dir = Path(__file__).resolve().parent.parent / "train"
    sys.path.insert(0, str(train_dir))
    from train_idle import IdleDetectorMLP

    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    model = IdleDetectorMLP()
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def evaluate_idle(model, test_manifest: str, window_dir: str,
                  window_size_sec: float, device: torch.device) -> dict:
    """Evaluate idle detection on test windows.

    Returns precision, recall, F1, and duration error statistics.
    """
    with open(test_manifest) as f:
        manifest = json.load(f)

    window_dir = Path(window_dir)
    all_true = []
    all_pred = []
    all_starts = []
    all_stems = []

    for fn in manifest["files"]:
        fp = window_dir / fn
        if not fp.exists():
            continue
        data = np.load(str(fp), allow_pickle=True)
        energy = data["motion_energy"]
        std = data["motion_std"]
        labels = data["labels"]
        starts = data["window_starts"]
        stem = str(data.get("site", "unknown"))

        features = np.stack([energy, std], axis=1).astype(np.float32)
        x = torch.from_numpy(features).to(device)
        with torch.no_grad():
            preds = model(x).argmax(dim=1).cpu().numpy()

        all_true.append(labels)
        all_pred.append(preds)
        all_starts.append(starts)
        all_stems.extend([stem] * len(labels))

    if not all_true:
        return {"error": "no test windows found"}

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    w_starts = np.concatenate(all_starts)

    # Precision, recall, F1
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    pr_metrics = compute_f1(tp, fp, fn)

    # Duration error: compare total predicted idle time vs ground truth
    gt_idle_sec = float(y_true.sum()) * window_size_sec
    pred_idle_sec = float((y_pred == 1).sum()) * window_size_sec
    if gt_idle_sec > 0:
        duration_error_pct = abs(pred_idle_sec - gt_idle_sec) / gt_idle_sec * 100
    else:
        duration_error_pct = 0.0 if pred_idle_sec == 0 else 100.0

    return {
        "precision": pr_metrics["precision"],
        "recall": pr_metrics["recall"],
        "f1": pr_metrics["f1"],
        "tp": tp, "fp": fp, "fn": fn,
        "gt_idle_sec": gt_idle_sec,
        "pred_idle_sec": pred_idle_sec,
        "duration_error_pct": duration_error_pct,
        "n_windows": len(y_true),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_phase_report(f1_results: dict, boundary_results: dict,
                       targets: dict) -> bool:
    """Print phase evaluation report; return True if all targets pass."""
    print("\n" + "=" * 70)
    print("PHASE RECOGNITION EVALUATION")
    print("=" * 70)

    all_pass = True

    # Per-class F1
    f1_target = targets.get("f1_per_phase", 0.0)
    print(f"\nPer-class F1 (target >= {f1_target:.2f}):")
    print(f"  {'Phase':<30} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Status':>8}")
    print(f"  {'-' * 68}")
    for key, val in f1_results.items():
        if key == "macro_f1":
            continue
        status = "PASS" if val["f1"] >= f1_target else "FAIL"
        if val["f1"] < f1_target:
            all_pass = False
        print(f"  {key:<30} {val['precision']:>10.4f} {val['recall']:>10.4f} "
              f"{val['f1']:>10.4f} {status:>8}")

    macro = f1_results.get("macro_f1", 0.0)
    macro_status = "PASS" if macro >= f1_target else "FAIL"
    if macro < f1_target:
        all_pass = False
    print(f"  {'MACRO F1':<30} {'':>10} {'':>10} {macro:>10.4f} {macro_status:>8}")

    # Boundary accuracy
    boundary_target = targets.get("phase_boundary_accuracy_sec", 30)
    ba = boundary_results.get("boundary_accuracy", 0.0)
    ba_status = "PASS" if ba >= 0.80 else "FAIL"
    if ba < 0.80:
        all_pass = False
    print(f"\nPhase boundary accuracy (+/-{boundary_target}s): "
          f"{ba:.2%} ({boundary_results.get('n_boundaries', 0)} boundaries) "
          f"[{ba_status}]")
    if "mean_error_sec" in boundary_results:
        print(f"  Mean boundary error: {boundary_results['mean_error_sec']:.1f}s, "
              f"median: {boundary_results.get('median_error_sec', 0):.1f}s")

    return all_pass


def print_idle_report(idle_results: dict, targets: dict) -> bool:
    """Print idle detection report; return True if all targets pass."""
    print("\n" + "=" * 70)
    print("IDLE DETECTION EVALUATION")
    print("=" * 70)

    all_pass = True
    prec_target = targets.get("precision", 0.0)
    recall_target = targets.get("recall", 0.0)
    dur_target = targets.get("duration_error_pct", 100.0)

    prec = idle_results.get("precision", 0.0)
    rec = idle_results.get("recall", 0.0)
    dur_err = idle_results.get("duration_error_pct", 100.0)

    prec_status = "PASS" if prec >= prec_target else "FAIL"
    rec_status = "PASS" if rec >= recall_target else "FAIL"
    dur_status = "PASS" if dur_err <= dur_target else "FAIL"

    if prec < prec_target:
        all_pass = False
    if rec < recall_target:
        all_pass = False
    if dur_err > dur_target:
        all_pass = False

    print(f"\n  {'Metric':<30} {'Value':>10} {'Target':>10} {'Status':>8}")
    print(f"  {'-' * 58}")
    print(f"  {'Precision':<30} {prec:>10.4f} {'>=' + f'{prec_target:.2f}':>10} {prec_status:>8}")
    print(f"  {'Recall':<30} {rec:>10.4f} {'>=' + f'{recall_target:.2f}':>10} {rec_status:>8}")
    print(f"  {'F1':<30} {idle_results.get('f1', 0):>10.4f} {'':>10} {'':>8}")
    print(f"  {'Duration error %':<30} {dur_err:>10.1f} {'<=' + f'{dur_target:.0f}':>10} {dur_status:>8}")
    print(f"\n  GT idle time: {idle_results.get('gt_idle_sec', 0):.0f}s, "
          f"predicted: {idle_results.get('pred_idle_sec', 0):.0f}s")
    print(f"  Windows: {idle_results.get('n_windows', 0)} "
          f"(TP={idle_results.get('tp', 0)}, "
          f"FP={idle_results.get('fp', 0)}, "
          f"FN={idle_results.get('fn', 0)})")

    return all_pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate surgical workflow models (phase + idle)."
    )
    parser.add_argument(
        "--config", type=str, default=DEFAULT_CONFIG,
        help="Path to config YAML"
    )
    parser.add_argument(
        "--data-dir", type=str, required=True,
        help="Base splits directory (contains phase/ and idle/ subdirs)"
    )
    parser.add_argument(
        "--mode", type=str, default="all", choices=["phase", "idle", "all"],
        help="Which evaluation to run (default: all)"
    )
    parser.add_argument(
        "--phase-checkpoint", type=str, default=None,
        help="Path to phase model checkpoint (default: checkpoints/phase/best_phase_model.pt)"
    )
    parser.add_argument(
        "--idle-checkpoint", type=str, default=None,
        help="Path to idle model checkpoint (default: checkpoints/idle/best_idle_model.pt)"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device: cuda / cpu (default: auto-detect)"
    )
    args = parser.parse_args()

    # Config
    config = load_config(args.config)
    config_dir = Path(config["_config_dir"]).parent

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_dir = Path(args.data_dir)
    overall_pass = True

    # --- Phase evaluation ---
    if args.mode in ("phase", "all"):
        phase_cfg = config["phase_recognition"]
        phases = phase_cfg["phases"]
        phase_targets = phase_cfg["evaluation"]["targets"]

        phase_ckpt = args.phase_checkpoint or str(
            config_dir / "checkpoints" / "phase" / "best_phase_model.pt")
        phase_split_dir = data_dir / "phase"
        phase_clip_dir = config_dir / "data" / "clips" / "phase"
        test_manifest = str(phase_split_dir / "test.json")

        if not Path(phase_ckpt).exists():
            print(f"\nPhase checkpoint not found: {phase_ckpt}")
            print("  Skipping phase evaluation.")
            overall_pass = False
        elif not Path(test_manifest).exists():
            print(f"\nPhase test manifest not found: {test_manifest}")
            print("  Skipping phase evaluation.")
            overall_pass = False
        else:
            model, phases_loaded = load_phase_model(phase_ckpt, device)
            f1_results = evaluate_phase_clips(
                model, test_manifest, str(phase_clip_dir), phases_loaded, device)
            boundary_results = evaluate_phase_boundaries(
                test_manifest, str(phase_clip_dir), model, phases_loaded,
                device)
            phase_pass = print_phase_report(f1_results, boundary_results,
                                            phase_targets)
            if not phase_pass:
                overall_pass = False

    # --- Idle evaluation ---
    if args.mode in ("idle", "all"):
        idle_cfg = config["idle_detection"]
        idle_targets = idle_cfg["evaluation"]["targets"]
        window_size_sec = idle_cfg["window_size_sec"]

        idle_ckpt = args.idle_checkpoint or str(
            config_dir / "checkpoints" / "idle" / "best_idle_model.pt")
        idle_split_dir = data_dir / "idle"
        idle_window_dir = config_dir / "data" / "clips" / "idle"
        test_manifest = str(idle_split_dir / "test.json")

        if not Path(idle_ckpt).exists():
            print(f"\nIdle checkpoint not found: {idle_ckpt}")
            print("  Skipping idle evaluation.")
            overall_pass = False
        elif not Path(test_manifest).exists():
            print(f"\nIdle test manifest not found: {test_manifest}")
            print("  Skipping idle evaluation.")
            overall_pass = False
        else:
            idle_model = load_idle_model(idle_ckpt, device)
            idle_results = evaluate_idle(
                idle_model, test_manifest, str(idle_window_dir),
                window_size_sec, device)
            idle_pass = print_idle_report(idle_results, idle_targets)
            if not idle_pass:
                overall_pass = False

    # --- Overall ---
    print("\n" + "=" * 70)
    status = "ALL TARGETS PASSED" if overall_pass else "SOME TARGETS FAILED"
    print(f"OVERALL: {status}")
    print("=" * 70)

    sys.exit(0 if overall_pass else 1)


if __name__ == "__main__":
    main()
