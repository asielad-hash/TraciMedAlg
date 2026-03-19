"""
Fine-tune YOLOv8 for hand / surgical-glove detection within the tray ROI.

This trains a YOLO object detector to locate hands entering the instrument
tray area.  The detections are one of the three signals used by the tray
monitor's event classification pipeline (alongside frame differencing and
count delta).

The training data should follow standard YOLO format:
    data_yaml points to a dataset with:
        images/train/*, images/val/*
        labels/train/*, labels/val/*
    Classes: 0=hand, 1=gloved_hand (or as defined in the data YAML)

Usage:
    python train_hand_detector.py --data-yaml data/hand_detection/data.yaml --epochs 50
    python train_hand_detector.py --data-yaml data/hand_detection/data.yaml --model yolov8s.pt --imgsz 640
"""

import argparse
import sys
from pathlib import Path

# Shared library import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from shared.config import load_config

DEFAULT_CONFIG = str(Path(__file__).resolve().parent.parent / "config" / "config.yaml")


def validate_data_yaml(data_yaml: Path) -> dict:
    """Load and validate the YOLO dataset YAML.

    Returns the parsed YAML content.
    """
    import yaml

    if not data_yaml.is_file():
        print(f"Error: data YAML not found: {data_yaml}")
        sys.exit(1)

    with open(data_yaml) as f:
        data_cfg = yaml.safe_load(f)

    required_keys = ["train", "val", "nc", "names"]
    for key in required_keys:
        if key not in data_cfg:
            print(f"Error: data YAML missing required key '{key}'")
            sys.exit(1)

    print(f"Dataset YAML: {data_yaml}")
    print(f"  Classes ({data_cfg['nc']}): {data_cfg['names']}")
    print(f"  Train: {data_cfg['train']}")
    print(f"  Val:   {data_cfg['val']}")
    return data_cfg


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLOv8 for hand/surgical-glove detection in tray ROI."
    )
    parser.add_argument(
        "--data-yaml", type=str, required=True,
        help="Path to YOLO dataset YAML (images + labels)"
    )
    parser.add_argument(
        "--model", type=str, default="yolov8n.pt",
        help="Pretrained YOLO model to fine-tune (default: yolov8n.pt)"
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Number of training epochs (default: 50)"
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="Input image size (default: 640)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="Batch size (default: 16)"
    )
    parser.add_argument(
        "--lr", type=float, default=0.01,
        help="Initial learning rate (default: 0.01)"
    )
    parser.add_argument(
        "--device", type=str, default="",
        help="Device: 'cpu', '0', '0,1', or '' for auto (default: auto)"
    )
    parser.add_argument(
        "--project", type=str, default=None,
        help="Output project directory (default: checkpoints/hand_detector)"
    )
    parser.add_argument(
        "--name", type=str, default="run",
        help="Experiment name within project dir (default: run)"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume training from last checkpoint"
    )
    parser.add_argument(
        "--config", type=str, default=DEFAULT_CONFIG,
        help="Path to component config YAML for reference (default: ../config/config.yaml)"
    )
    args = parser.parse_args()

    # Resolve project directory
    if args.project is None:
        component_dir = Path(__file__).resolve().parent.parent
        project_dir = component_dir / "checkpoints" / "hand_detector"
    else:
        project_dir = Path(args.project)

    # Validate data YAML
    data_yaml = Path(args.data_yaml).resolve()
    data_cfg = validate_data_yaml(data_yaml)

    print()
    print("=" * 60)
    print("HAND DETECTOR TRAINING")
    print("=" * 60)
    print(f"  Model:      {args.model}")
    print(f"  Epochs:     {args.epochs}")
    print(f"  Image size: {args.imgsz}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  LR:         {args.lr}")
    print(f"  Device:     {args.device or 'auto'}")
    print(f"  Project:    {project_dir}")
    print(f"  Name:       {args.name}")
    print(f"  Resume:     {args.resume}")
    print("=" * 60)
    print()

    # Import ultralytics (deferred so --help works without it installed)
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: ultralytics package not installed.")
        print("Install with: pip install ultralytics")
        sys.exit(1)

    # Load model
    print(f"Loading model: {args.model}")
    model = YOLO(args.model)

    # Train
    print("Starting training...")
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch_size,
        lr0=args.lr,
        device=args.device or None,
        project=str(project_dir),
        name=args.name,
        exist_ok=True,
        resume=args.resume,
        verbose=True,
        plots=True,
        save=True,
        save_period=10,
    )

    # Print summary
    print()
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    # Best model path
    best_path = project_dir / args.name / "weights" / "best.pt"
    last_path = project_dir / args.name / "weights" / "last.pt"
    print(f"  Best weights: {best_path}")
    print(f"  Last weights: {last_path}")

    # Run validation on best model
    print()
    print("Running validation on best model...")
    best_model = YOLO(str(best_path))
    val_results = best_model.val(data=str(data_yaml), imgsz=args.imgsz)

    if hasattr(val_results, "box"):
        box = val_results.box
        print(f"  mAP50:    {box.map50:.4f}")
        print(f"  mAP50-95: {box.map:.4f}")
        if hasattr(box, "maps") and box.maps is not None:
            for i, name in enumerate(data_cfg.get("names", [])):
                if i < len(box.maps):
                    print(f"  AP({name}): {box.maps[i]:.4f}")

    print()
    print("Done. Use the best weights for inference in the event detection pipeline.")


if __name__ == "__main__":
    main()
