"""
Deepfake Detection Model Architectures.
Provides support for:
1. DeepfakeCNNBiLSTMClassifier: Temporal CNN-BiLSTM hybrid for large-scale audio deepfake detection.
2. DeepfakeCNNClassifier: 2D Convolutional neural network for Mel-spectrogram binary classification.
3. Universal loader: Automatically determines architecture from saved checkpoint keys.
"""

import os
import torch
import torch.nn as nn


class DeepfakeCNNBiLSTMClassifier(nn.Module):
    """
    CNN-BiLSTM Hybrid Architecture for Voice Deepfake Detection.
    Captures localized frequency artifacts via 4-layer 2D CNN and sequential
    temporal inconsistencies via bidirectional LSTM.
    """
    def __init__(self, num_mels: int = 64, lstm_hidden: int = 128, dropout_rate: float = 0.35):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, None))
        )
        self.lstm = nn.LSTM(
            input_size=256 * 8,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate
        )
        self.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(lstm_hidden * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        B, C, F_dim, T_dim = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B, T_dim, C * F_dim)
        lstm_out, _ = self.lstm(x)
        pooled = torch.mean(lstm_out, dim=1)
        logits = self.fc(pooled)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logit = self.forward(x)
            prob = torch.sigmoid(logit)
        return prob

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def save_weights(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(self.state_dict(), path)

    def load_weights(self, path: str, map_location: str = "cpu"):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Weights file not found: {path}")
        state = torch.load(path, map_location=map_location, weights_only=True)
        self.load_state_dict(state)
        self.eval()


class DeepfakeCNNClassifier(nn.Module):
    """
    Standard 4-block 2D CNN feature extractor and classification head.
    """
    def __init__(self, dropout_rate: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate / 2),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.features(x)
        logit = self.classifier(feats)
        return logit

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logit = self.forward(x)
            prob = torch.sigmoid(logit)
        return prob

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def save_weights(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(self.state_dict(), path)

    def load_weights(self, path: str, map_location: str = "cpu"):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Weights file not found: {path}")
        state = torch.load(path, map_location=map_location, weights_only=True)
        self.load_state_dict(state)
        self.eval()


# Alias for backward compatibility
DeepfakeAudioClassifier = DeepfakeCNNBiLSTMClassifier


def load_classifier(weights_path: str, map_location: str = "cpu") -> nn.Module:
    """
    Universal classifier loader. Inspects the checkpoint state_dict and dynamically
    instantiates either DeepfakeCNNBiLSTMClassifier or DeepfakeCNNClassifier.
    """
    if not os.path.exists(weights_path):
        # Default to CNN-BiLSTM if initializing without existing weights
        return DeepfakeCNNBiLSTMClassifier()

    state = torch.load(weights_path, map_location=map_location, weights_only=True)

    # Check key signatures
    if "lstm.weight_ih_l0" in state or "conv1.0.weight" in state:
        model = DeepfakeCNNBiLSTMClassifier()
        model.load_state_dict(state)
    elif "features.0.weight" in state or "classifier.1.weight" in state:
        model = DeepfakeCNNClassifier()
        model.load_state_dict(state)
    else:
        # Fallback
        model = DeepfakeCNNBiLSTMClassifier()
        model.load_state_dict(state, strict=False)

    model.eval()
    return model
