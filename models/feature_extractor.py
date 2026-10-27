"""
Audio Feature Extraction Module for Voice Impersonation Detection.
Extracts normalized Log-Mel Spectrograms for deepfake detection models
and computes acoustic diagnostics (spectral centroid, rolloff, flatness).
"""

import numpy as np
import librosa
import torch
from typing import Dict, Any


class MelSpectrogramExtractor:
    def __init__(
        self,
        sr: int = 16000,
        n_mels: int = 64,
        n_fft: int = 1024,
        hop_length: int = 512,
        fmin: float = 20.0,
        fmax: float = 8000.0,
    ):
        self.sr = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.fmin = fmin
        self.fmax = fmax

    def extract_mel_spectrogram(self, y: np.ndarray) -> np.ndarray:
        """
        Extract Log Mel-Spectrogram (dB scale) normalized to zero mean, unit variance.
        Output shape: (n_mels, time_steps)
        """
        mel = librosa.feature.melspectrogram(
            y=y,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
            power=2.0
        )
        log_mel = librosa.power_to_db(mel, ref=np.max)

        # Standardization
        mean = np.mean(log_mel)
        std = np.std(log_mel)
        if std > 1e-6:
            norm_mel = (log_mel - mean) / std
        else:
            norm_mel = log_mel - mean

        return norm_mel.astype(np.float32)

    def extract_tensor(self, y: np.ndarray) -> torch.Tensor:
        """
        Convert audio waveform to 4D PyTorch tensor:
        Shape: (1, 1, n_mels, time_steps)
        """
        spec = self.extract_mel_spectrogram(y)
        tensor = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0)
        return tensor

    def compute_acoustic_diagnostics(self, y: np.ndarray) -> Dict[str, Any]:
        """
        Compute acoustic diagnostic metrics that differentiate real human speech
        from vocoded / synthetic deepfakes:
        - Spectral Centroid: average frequency weighted by amplitude
        - Spectral Rolloff: frequency below which 85% of energy resides
        - Zero-Crossing Rate: rate of sign-changes in signal
        - Spectral Flatness: measure of tone vs noisiness
        """
        try:
            centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=self.sr, hop_length=self.hop_length)))
            rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=self.sr, hop_length=self.hop_length)))
            zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=y, hop_length=self.hop_length)))
            flatness = float(np.mean(librosa.feature.spectral_flatness(y=y, hop_length=self.hop_length)))
        except Exception:
            centroid, rolloff, zcr, flatness = 0.0, 0.0, 0.0, 0.0

        return {
            "spectral_centroid_hz": round(centroid, 1),
            "spectral_rolloff_hz": round(rolloff, 1),
            "zero_crossing_rate": round(zcr, 4),
            "spectral_flatness": round(flatness, 6)
        }
