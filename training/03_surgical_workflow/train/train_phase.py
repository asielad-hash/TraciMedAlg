"""
Train temporal phase classifier for surgical workflow recognition.

Architecture: ResNet-18 (frozen backbone) per-frame feature extraction
-> LSTM temporal model -> linear classification head.

Loads 64-frame phase clips produced by extract_phase_clips.py and trains
with CrossEntropyLoss on the majority-vote phase label.

Usage:
    python train_phase.py --config ../config/config.yaml \
        --data-dir ../data/splits/phase --output-dir ../checkpoints/phase
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

class PhaseClipDataset(Dataset):
    """Dataset of pre-extracted phase clips stored as .npz files.

    Each sample returns:
        frames: (T, 3, H', W') float32 tensor (resized, normalised)
        label:  int64 majority phase ID
    """

    RESIZE = (224, 224)
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, manifest_path: str, data_dir: str):
        with open(manifest_path) as f:
            manifest = json.load(f)
        self.data_dir = Path(data_dir)
        self.files = [self.data_dir / fn for fn in manifest["files"]]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        import cv2  # deferred import for workers
        data = np.load(str(self.files[idx]))
        frames = data["frames"]  # (T, H, W, 3) uint8 BGR
        label = int(data["majority_phase"])

        # Resize, BGR->RGB, normalise, (T,H,W,3) -> (T,3,H,W)
        processed = []
        for frame in frames:
            frame = cv2.resize(frame, self.RESIZE)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.astype(np.float32) / 255.0
            frame = (frame - self.MEAN) / self.STD
            processed.append(frame.transpose(2, 0, 1))  # (3, H, W)

        frames_t = torch.from_numpy(np.stack(processed, axis=0))  # (T, 3, H, W)
        return frames_t, torch.tensor(label, dtype=torch.long)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class PhaseClassifier(nn.Module):
    """ResNet-18 feature extractor + LSTM + linear head.

    The ResNet backbone is frozen; only the LSTM and classifier head
    are trained.  This keeps GPU memory manageable on surgical-length
    clips while still learning temporal patterns.
    """

    def __init__(self, num_phases: int, hidden_dim: int = 256,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        import torchvision.models as models
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Remove final FC — output is 512-dim feature
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.feature_dim = 512

        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_phases)

    def forward(self, x):
        """
        Args:
            x: (B, T, 3, H, W)
        Returns:
            logits: (B, num_phases)
        """
        B, T = x.shape[:2]
        # Extract per-frame features
        frames = x.reshape(B * T, *x.shape[2:])      # (B*T, 3, H, W)
        with torch.no_grad():
            feats = self.backbone(frames)              # (B*T, 512, 1, 1)
        feats = feats.squeeze(-1).squeeze(-1)          # (B*T, 512)
        feats = feats.reshape(B, T, self.feature_dim)  # (B, T, 512)

        # Temporal modelling
        lstm_out, _ = self.lstm(feats)                 # (B, T, hidden)
        final = lstm_out[:, -1, :]                     # (B, hidden)
        logits = self.classifier(self.dropout(final))  # (B, num_phases)
        return logits


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch; return average loss and accuracy."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for frames, labels in loader:
        frames = frames.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(frames)
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
    """Validate; return average loss and accuracy."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for frames, labels in loader:
        frames = frames.to(device)
        labels = labels.to(device)
        logits = model(frames)
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
        description="Train temporal phase classifier (ResNet + LSTM)."
    )
    parser.add_argument(
        "--config", type=str, default=DEFAULT_CONFIG,
        help="Path to config YAML"
    )
    parser.add_argument(
        "--data-dir", type=str, required=True,
        help="Directory containing split manifests (train.json, val.json) "
             "and the parent clip directory"
    )
    parser.add_argument(
        "--clip-dir", type=str, default=None,
        help="Directory containing .npz clip files (default: inferred from data-dir)"
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
    phase_cfg = config["phase_recognition"]
    train_cfg = phase_cfg["training"]
    phases = phase_cfg["phases"]
    num_phases = len(phases)

    epochs = train_cfg["epochs"]
    batch_size = train_cfg["batch_size"]
    lr = train_cfg["learning_rate"]

    # Directories
    data_dir = Path(args.data_dir)
    clip_dir = Path(args.clip_dir) if args.clip_dir else data_dir.parent
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(config["_config_dir"]).parent / "checkpoints" / "phase"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Phases ({num_phases}): {phases}")
    print(f"Epochs: {epochs}, batch_size: {batch_size}, lr: {lr}")

    # Datasets
    train_ds = PhaseClipDataset(str(data_dir / "train.json"), str(clip_dir))
    val_ds = PhaseClipDataset(str(data_dir / "val.json"), str(clip_dir))
    print(f"Train: {len(train_ds)} clips, Val: {len(val_ds)} clips\n")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)

    # Model
    model = PhaseClassifier(num_phases=num_phases).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
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

        print(f"Epoch {epoch:3d}/{epochs} | "
              f"train_loss={train_loss:.4f} acc={train_acc:.3f} | "
              f"val_loss={val_loss:.4f} acc={val_acc:.3f} | "
              f"{elapsed:.1f}s")

        # Checkpoint best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = output_dir / "best_phase_model.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "phases": phases,
                "config": train_cfg,
            }, str(ckpt_path))
            print(f"  -> saved best checkpoint (val_loss={val_loss:.4f})")

    # Save final model and history
    torch.save(model.state_dict(), str(output_dir / "final_phase_model.pt"))
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best val_loss: {best_val_loss:.4f}")
    print(f"Checkpoints: {output_dir}")


if __name__ == "__main__":
    main()
