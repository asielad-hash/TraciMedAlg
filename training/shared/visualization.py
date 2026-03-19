"""
Visualization utilities — overlays, plotting, comparison videos.
Extracted from trackInstruments/python/alert_overlay.py + new code.
"""

import cv2
import numpy as np
from pathlib import Path


def draw_alert_banner(frame_bgr: np.ndarray, text: str,
                       color_bgr: tuple, position: str = "top-right",
                       font_scale: float = 0.7) -> np.ndarray:
    """Draw an alert banner on a frame.

    Args:
        color_bgr: banner background color
        position: "top-right" or "top-left"
    """
    overlay = frame_bgr.copy()
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
    pad = 8
    h, w = frame_bgr.shape[:2]

    if position == "top-right":
        x2 = w - 10
        x1 = x2 - tw - 2 * pad
    else:
        x1 = 10
        x2 = x1 + tw + 2 * pad

    y1, y2 = 10, 10 + th + 2 * pad
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color_bgr, -1)
    cv2.putText(overlay, text, (x1 + pad, y2 - pad),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)
    return overlay


def draw_count_badge(frame_bgr: np.ndarray, count: int,
                      position: str = "top-left") -> np.ndarray:
    """Draw instrument count badge on frame."""
    overlay = frame_bgr.copy()
    text = f"Count: {count}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    pad = 10
    x1, y1 = 10, 10
    x2, y2 = x1 + tw + 2 * pad, y1 + th + 2 * pad
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (40, 40, 40), -1)
    cv2.putText(overlay, text, (x1 + pad, y2 - pad),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return overlay


def create_side_by_side(frame_a: np.ndarray, frame_b: np.ndarray,
                         label_a: str = "Original", label_b: str = "Tracked") -> np.ndarray:
    """Create side-by-side comparison of two frames."""
    h_a, w_a = frame_a.shape[:2]
    h_b, w_b = frame_b.shape[:2]

    # Match heights
    target_h = max(h_a, h_b)
    if h_a != target_h:
        scale = target_h / h_a
        frame_a = cv2.resize(frame_a, (int(w_a * scale), target_h))
    if h_b != target_h:
        scale = target_h / h_b
        frame_b = cv2.resize(frame_b, (int(w_b * scale), target_h))

    # Add labels
    cv2.putText(frame_a, label_a, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame_b, label_b, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Concatenate
    combined = np.hstack([frame_a, frame_b])
    return combined


# ---------------------------------------------------------------------------
# Matplotlib-based plotting (optional — only if matplotlib available)
# ---------------------------------------------------------------------------

def plot_training_curves(train_losses: list, val_losses: list,
                          output_path: str = None, title: str = "Training Curves"):
    """Plot training and validation loss curves."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train_losses, label="Train Loss", color="blue")
    ax.plot(val_losses, label="Val Loss", color="red")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {output_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, class_names: list = None,
                           output_path: str = None, title: str = "Confusion Matrix"):
    """Plot a confusion matrix."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)

    n = cm.shape[0]
    labels = class_names if class_names else [str(i) for i in range(n)]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_ylabel("True")
    ax.set_xlabel("Predicted")

    # Annotate cells
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {output_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_metrics_report(metrics: dict, targets: dict = None,
                         output_path: str = None):
    """Bar chart comparing achieved metrics vs targets."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return

    names = list(metrics.keys())
    values = [metrics[n] for n in names]

    fig, ax = plt.subplots(figsize=(12, 5))
    x = range(len(names))
    bars = ax.bar(x, values, color="steelblue", alpha=0.8, label="Achieved")

    if targets:
        target_vals = [targets.get(n, 0) for n in names]
        ax.bar(x, target_vals, color="none", edgecolor="red",
               linewidth=2, linestyle="--", label="Target")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Evaluation Metrics vs Targets")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)
