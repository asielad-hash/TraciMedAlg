"""
Train binary idle-period detector.

Architecture: simple MLP classifier over sliding-window motion features
(motion_energy, motion_std) produced by extract_idle_windows.py.

Usage:
    python train_idle.py --config ../config/config.yaml \
        --data-dir ../data/splits/idle --output-dir ../checkpoints/idle
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Shared library import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from shared.config import load_config

DEFAULT_CONFIG = str(Path(__file__).resolve().parent.parent / "config" / "config.yaml")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class IdleWindowDataset(Dataset):
    """Dataset of sliding-window motion features for idle detection.

    Each sample is a 2-dim feature vector (motion_energy, motion_std)
    with a binary label (1=idle, 0=active).
    """

    def __init__(self, manifest_path: str, data_dir: str):
        with open(manifest_path) as f:
            manifest = json.load(f)
        data_dir = Path(data_dir)

        features_list = []
        labels_list = []
        for fn in manifest["files"]:
            fp = data_dir / fn
            if not fp.exists():
                continue
            data = np.load(str(fp))
            energy = data["motion_energy"]
            std = data["motion_std"]
            labels = data["labels"]
            feats = np.stack([energy, std], axis=1)  # (N, 2)
            features_list.append(feats)
            labels_list.append(labels)

        if features_list:
            self.features = np.concatenate(features_list, axis=0).astype(np.float32)
            self.labels = np.concatenate(labels_list, axis=0).astype(np.int64)
        else:
            self.features = np.zeros((0, 2), dtype=np.float32)
            self.labels = np.zeros((0,), dtype=np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (torch.from_numpy(self.features[idx]),
                torch.tensor(self.labels[idx], dtype=torch.long))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class IdleDetectorMLP(nn.Module):
    """Two-layer MLP for binary idle detection.

    Input: 2-dim (motion_energy, motion_std)
    Output: 2-dim logits (active, idle)
    """

    def __init__(self, input_dim: int = 2, hidden_dim: int = 64,
                 dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train one epoch; return loss and accuracy."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Validate; return loss and accuracy."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)
        logits = model(features)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train binary idle-period detector (MLP)."
    )
    parser.add_argument(
        "--config", type=str, default=DEFAULT_CONFIG,
        help="Path to config YAML"
    )
    parser.add_argument(
        "--data-dir", type=str, required=True,
        help="Directory containing split manifests (train.json, val.json) "
             "and the parent idle-window directory"
    )
    parser.add_argument(
        "--window-dir", type=str, default=None,
        help="Directory containing .npz window files (default: inferred from data-dir)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory for checkpoints (default: config model.output_dir)"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device: cuda / cpu (default: auto-detect)"
    )
    args = parser.parse_args()

    # Config
    config = load_config(args.config)
    idle_cfg = config["idle_detection"]
    train_cfg = idle_cfg["training"]

    epochs = train_cfg["epochs"]
    batch_size = train_cfg["batch_size"]
    lr = train_cfg["learning_rate"]

    # Directories
    data_dir = Path(args.data_dir)
    window_dir = Path(args.window_dir) if args.window_dir else data_dir.parent
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(config["_config_dir"]).parent / "checkpoints" / "idle"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Epochs: {epochs}, batch_size: {batch_size}, lr: {lr}")

    # Datasets
    train_ds = IdleWindowDataset(str(data_dir / "train.json"), str(window_dir))
    val_ds = IdleWindowDataset(str(data_dir / "val.json"), str(window_dir))
    print(f"Train: {len(train_ds)} windows, Val: {len(val_ds)} windows")

    # Class balance info
    if len(train_ds) > 0:
        n_idle = (train_ds.labels == 1).sum()
        n_active = (train_ds.labels == 0).sum()
        print(f"  Train balance: {n_active} active, {n_idle} idle "
              f"({100 * n_idle / len(train_ds):.1f}% idle)")

    # Use class weights to handle imbalance
    if len(train_ds) > 0 and n_idle > 0 and n_active > 0:
        weight_active = len(train_ds) / (2 * n_active)
        weight_idle = len(train_ds) / (2 * n_idle)
        class_weights = torch.tensor([weight_active, weight_idle],
                                     dtype=torch.float32).to(device)
    else:
        class_weights = None
    print()

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)

    # Model
    model = IdleDetectorMLP().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8
    )

    # Training loop
    best_val_loss = float("inf")
    history = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion,
                                                optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        elapsed = time.time() - t0

        record = {
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc, "time_s": round(elapsed, 1),
        }
        history.append(record)

        if epoch % 5 == 1 or epoch == epochs:
            print(f"Epoch {epoch:3d}/{epochs} | "
                  f"train_loss={train_loss:.4f} acc={train_acc:.3f} | "
                  f"val_loss={val_loss:.4f} acc={val_acc:.3f} | "
                  f"{elapsed:.1f}s")

        # Checkpoint best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = output_dir / "best_idle_model.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "config": train_cfg,
            }, str(ckpt_path))

    # Save final model and history
    torch.save(model.state_dict(), str(output_dir / "final_idle_model.pt"))
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best val_loss: {best_val_loss:.4f}")
    print(f"Checkpoints: {output_dir}")


if __name__ == "__main__":
    main()
