"""
Train a YOLOv8-seg instrument detector using Ultralytics.

Takes the data.yaml produced by export_yolo_dataset.py and fine-tunes
a YOLOv8 segmentation model for surgical instrument detection.

Usage:
    python train_detector.py \
        --data-yaml ./data/yolo_dataset/data.yaml \
        --epochs 50 \
        --imgsz 640 \
        --batch-size 4 \
        --model yolov8m-seg.pt
"""

import argparse
import json
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Shared library
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from shared.config import load_config


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLOv8-seg instrument detector."
    )
    parser.add_argument(
        "--data-yaml", required=True,
        help="Path to data.yaml from export_yolo_dataset.py.",
    )
    parser.add_argument(
        "--model", default="yolov8m-seg.pt",
        help="Pretrained model checkpoint (default: yolov8m-seg.pt).",
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Number of training epochs (default: 50).",
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="Input image size (default: 640).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=4,
        help="Batch size (default: 4).",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Initial learning rate (default: 1e-3).",
    )
    parser.add_argument(
        "--patience", type=int, default=10,
        help="Early stopping patience in epochs (default: 10).",
    )
    parser.add_argument(
        "--project", default=None,
        help="Output project directory (default: runs/segment).",
    )
    parser.add_argument(
        "--name", default="instrument_tracker",
        help="Experiment name within project (default: instrument_tracker).",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume training from last checkpoint.",
    )
    parser.add_argument(
        "--config", default=None,
        help="Optional component config.yaml to override defaults.",
    )
    args = parser.parse_args()

    data_yaml = Path(args.data_yaml)
    if not data_yaml.exists():
        print(f"ERROR: data.yaml not found at {data_yaml}")
        sys.exit(1)

    # Optionally load component config for training hyper-params
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    patience = args.patience

    if args.config:
        cfg = load_config(args.config)
        tcfg = cfg.get("training", {})
        epochs = tcfg.get("epochs", epochs)
        batch_size = tcfg.get("batch_size", batch_size)
        lr = tcfg.get("learning_rate", lr)
        patience = tcfg.get("early_stopping_patience", patience)
        print(f"Loaded training overrides from {args.config}")

    # Import ultralytics here so the CLI help works without it installed
    try:
        from ultralytics import YOLO
    except ImportError:
        print(
            "ERROR: ultralytics is not installed.\n"
            "Install with: pip install ultralytics"
        )
        sys.exit(1)

    print("=" * 60)
    print("Instrument Tracker — YOLOv8-seg Training")
    print("=" * 60)
    print(f"  Model:      {args.model}")
    print(f"  Data:       {data_yaml}")
    print(f"  Epochs:     {epochs}")
    print(f"  Image size: {args.imgsz}")
    print(f"  Batch size: {batch_size}")
    print(f"  LR:         {lr}")
    print(f"  Patience:   {patience}")
    print(f"  Resume:     {args.resume}")
    print("=" * 60)

    model = YOLO(args.model)

    t0 = time.time()
    results = model.train(
        data=str(data_yaml.resolve()),
        epochs=epochs,
        imgsz=args.imgsz,
        batch=batch_size,
        lr0=lr,
        patience=patience,
        project=args.project,
        name=args.name,
        resume=args.resume,
        save=True,
        save_period=10,
        plots=True,
        verbose=True,
    )
    elapsed = time.time() - t0

    # Locate best weights
    best_path = None
    if hasattr(results, "save_dir"):
        candidate = Path(results.save_dir) / "weights" / "best.pt"
        if candidate.exists():
            best_path = candidate

    print("\n" + "=" * 60)
    print("Training complete")
    print(f"  Elapsed:    {elapsed / 60:.1f} min")
    if best_path:
        print(f"  Best model: {best_path}")
    print("=" * 60)

    # Save a training summary JSON alongside the best weights
    summary = {
        "model": args.model,
        "data_yaml": str(data_yaml.resolve()),
        "epochs": epochs,
        "imgsz": args.imgsz,
        "batch_size": batch_size,
        "lr0": lr,
        "patience": patience,
        "elapsed_min": round(elapsed / 60, 2),
        "best_weights": str(best_path) if best_path else None,
    }
    if best_path:
        summary_path = best_path.parent.parent / "training_summary.json"
    else:
        summary_path = Path("training_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"  Summary:    {summary_path}")


if __name__ == "__main__":
    main()
