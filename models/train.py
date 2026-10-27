"""
Training and Fine-Tuning Pipeline for Model B.
Fine-tunes the deepfake detection model EXCLUSIVELY on the 80% training split:
- Loads data/splits/train_manifest.json (held-out test samples are strictly excluded)
- Applies waveform data augmentation (noise injection, random time shifting)
- Optimizes BCEWithLogitsLoss using AdamW with Cosine Annealing LR schedule
- Saves fine-tuned weights to models/saved/fine_tuned_model_b.pt
- Logs actual loss, accuracy, and training duration into reports/training_history.json
"""

import os
import json
import time
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from models.architecture import DeepfakeAudioClassifier
from models.feature_extractor import MelSpectrogramExtractor


class AudioDeepfakeDataset(Dataset):
    def __init__(self, manifest_path: str, augment: bool = True):
        with open(manifest_path, "r", encoding="utf-8") as f:
            self.records = json.load(f)
        self.augment = augment
        self.extractor = MelSpectrogramExtractor()

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        file_path = rec["segment_path"]
        y, sr = sf.read(file_path, dtype="float32")

        # Label: REAL -> 0.0, FAKE -> 1.0
        label_val = 1.0 if rec["label"] == "FAKE" else 0.0

        # Augmentation during training only
        if self.augment:
            # 1. Subtle random time roll / shift
            shift = np.random.randint(-int(sr * 0.2), int(sr * 0.2))
            y = np.roll(y, shift)

            # 2. Subtle Gaussian noise (SNR ~ 30 dB)
            if np.random.rand() > 0.5:
                noise = np.random.normal(0, 0.005, len(y)).astype(np.float32)
                y = y + noise

        # Extract 2D Mel-Spectrogram (64, T)
        mel_spec = self.extractor.extract_mel_spectrogram(y)
        # Add channel dimension: (1, 64, T)
        tensor = torch.from_numpy(mel_spec).unsqueeze(0)
        label_tensor = torch.tensor([label_val], dtype=torch.float32)

        return tensor, label_tensor, rec["label"], rec["parent_file"]


def train_model_b(
    train_manifest: str = "data/splits/train_manifest.json",
    output_weights: str = "models/saved/fine_tuned_model_b.pt",
    pretrained_weights: str = "models/saved/pretrained_model_a.pt",
    history_path: str = "reports/training_history.json",
    epochs: int = 25,
    batch_size: int = 8,
    learning_rate: float = 8e-4,
    seed: int = 42
) -> dict:
    """Train Model B strictly on training data."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"[Train] Initializing Model B training...")
    print(f"        Train manifest: {train_manifest}")
    print(f"        Epochs: {epochs}, Batch size: {batch_size}, LR: {learning_rate}")

    dataset = AudioDeepfakeDataset(train_manifest, augment=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    model = DeepfakeAudioClassifier(dropout_rate=0.35)
    if os.path.exists(pretrained_weights):
        print(f"[Train] Initializing from pretrained base weights: {pretrained_weights}")
        model.load_weights(pretrained_weights)
    else:
        print("[Train] Initializing fresh weights...")

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    if device_str == "cpu":
        torch.set_num_threads(min(4, os.cpu_count() or 1))
    print(f"[Train] Training device: {device} ({'GPU Acceleration Active' if device_str == 'cuda' else 'Multi-threaded CPU'})")
    model.to(device)
    model.train()

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    history = {
        "epochs": [],
        "loss": [],
        "train_accuracy": [],
        "learning_rates": [],
        "total_train_samples": len(dataset),
        "total_epochs": epochs,
        "seed": seed
    }

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        correct = 0
        total = 0

        for tensors, labels, _, _ in dataloader:
            tensors = tensors.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(tensors)
            loss = criterion(logits, labels)
            loss.backward()

            # Gradient clipping to stabilize training
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * tensors.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        scheduler.step()

        epoch_loss = round(running_loss / total, 4)
        epoch_acc = round(correct / total * 100.0, 2)
        current_lr = scheduler.get_last_lr()[0]

        history["epochs"].append(epoch)
        history["loss"].append(epoch_loss)
        history["train_accuracy"].append(epoch_acc)
        history["learning_rates"].append(current_lr)

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(f"  Epoch [{epoch:02d}/{epochs:02d}] - Loss: {epoch_loss:.4f} - Train Acc: {epoch_acc:.1f}% - LR: {current_lr:.6f}")

    total_duration = round(time.time() - start_time, 2)
    history["training_time_seconds"] = total_duration

    # Save fine-tuned weights
    os.makedirs(os.path.dirname(os.path.abspath(output_weights)), exist_ok=True)
    model.save_weights(output_weights)
    print(f"[Train] Model B fine-tuning completed in {total_duration}s!")
    print(f"[Train] Weights saved to: {output_weights}")

    # Save history
    os.makedirs(os.path.dirname(os.path.abspath(history_path)), exist_ok=True)
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"[Train] Training history saved to: {history_path}")

    return history


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train / Fine-tune Model B on audio dataset")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size (e.g., 8, 16, 32)")
    parser.add_argument("--lr", type=float, default=8e-4, help="Initial learning rate")
    parser.add_argument("--manifest", type=str, default="data/splits/train_manifest.json", help="Path to train manifest")
    args = parser.parse_args()

    train_model_b(train_manifest=args.manifest, epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr)
