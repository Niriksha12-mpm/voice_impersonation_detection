"""
Inference Engine for Voice Impersonation Detection.
Validates uploaded files, preprocesses audio, executes inference with
Model A (Pretrained) or Model B (Fine-Tuned), and computes diagnostics.
"""

import io
import os
import numpy as np
import soundfile as sf
import librosa
import torch
from typing import Dict, Any, Tuple
from models.architecture import DeepfakeAudioClassifier, load_classifier
from models.feature_extractor import MelSpectrogramExtractor
from dataset.preprocessor import AudioPreprocessor


class InferenceEngine:
    def __init__(
        self,
        pretrained_weights: str = "models/saved/pretrained_model_a.pt",
        fine_tuned_weights: str = "models/saved/fine_tuned_model_b_lakhs.pt" if os.path.exists("models/saved/fine_tuned_model_b_lakhs.pt") else "models/saved/fine_tuned_model_b.pt",
        device: str = "cpu"
    ):
        self.device = torch.device(device)
        self.preprocessor = AudioPreprocessor(target_sr=16000, target_duration=3.0)
        self.extractor = MelSpectrogramExtractor()

        # Load Model A (Pretrained)
        self.model_a = load_classifier(pretrained_weights, map_location=device)
        self.model_a.to(self.device).eval()

        # Load Model B (Fine-Tuned)
        self.model_b = load_classifier(fine_tuned_weights, map_location=device)
        self.model_b.to(self.device).eval()

        print("[InferenceEngine] Initialized Model A and Model B successfully.")

    def decode_and_validate(self, audio_bytes: bytes, filename: str) -> Tuple[np.ndarray, int, Dict[str, Any]]:
        """
        Safely decode raw audio bytes and validate signal integrity.
        Supports WAV, MP3, FLAC, OGG.
        """
        ext = os.path.splitext(filename)[1].lower()
        allowed_exts = [".wav", ".mp3", ".flac", ".ogg", ".m4a"]
        if ext not in allowed_exts:
            raise ValueError(f"Unsupported file format '{ext}'. Please upload WAV, MP3, or FLAC.")

        if not audio_bytes or len(audio_bytes) < 100:
            raise ValueError("The uploaded file is empty or corrupted.")

        # Decode via soundfile or librosa BytesIO
        y = None
        sr = None
        try:
            buffer = io.BytesIO(audio_bytes)
            y, sr = sf.read(buffer, dtype="float32")
        except Exception:
            try:
                buffer = io.BytesIO(audio_bytes)
                y, sr = librosa.load(buffer, sr=None, mono=False)
            except Exception as e:
                raise ValueError(f"Failed to decode audio file. The file may be damaged or invalid: {e}")

        # Validate
        valid, msg = self.preprocessor.validate_audio(y, sr)
        if not valid:
            raise ValueError(f"Audio validation failed: {msg}")

        # Basic metadata
        raw_duration = round(y.shape[-1] / sr if y.ndim > 1 else len(y) / sr, 3)
        raw_channels = y.shape[0] if y.ndim > 1 else 1

        return y, sr, {
            "filename": filename,
            "format": ext.replace(".", "").upper(),
            "original_sr": sr,
            "channels": raw_channels,
            "duration_seconds": raw_duration
        }

    def predict(self, audio_bytes: bytes, filename: str, model_type: str = "fine_tuned") -> Dict[str, Any]:
        """
        Run complete prediction pipeline.
        Args:
            audio_bytes: raw file bytes
            filename: original file name
            model_type: 'fine_tuned' (Model B) or 'pretrained' (Model A)
        """
        y_raw, orig_sr, meta = self.decode_and_validate(audio_bytes, filename)

        # Standardize preprocessing to 16kHz mono, trimmed and normalized
        processed_y = self.preprocessor.process_waveform(y_raw, orig_sr)
        segments = self.preprocessor.segment_or_pad(processed_y)

        # Select model
        if model_type == "pretrained":
            active_model = self.model_a
            active_model_name = "Model A (Pretrained Model)"
            model_desc = "Baseline audio deepfake detector evaluated out-of-the-box without dataset fine-tuning."
        else:
            active_model = self.model_b
            active_model_name = "Model B (Lakhs-Trained CNN-BiLSTM)"
            model_desc = "Fine-tuned CNN-BiLSTM hybrid deepfake detector trained on 104,800+ WaveFake & ASVspoof utterances via Google Colab."

        # Run inference across segments (average probability for multi-segment files)
        seg_probs = []
        with torch.no_grad():
            for seg in segments:
                tensor = self.extractor.extract_tensor(seg).to(self.device)
                logit = active_model(tensor).item()
                p_fake = 1.0 / (1.0 + np.exp(-logit))
                seg_probs.append(p_fake)

        avg_prob_fake = float(np.mean(seg_probs))

        # Binary label: ONLY REAL or FAKE
        if avg_prob_fake >= 0.5:
            label = "FAKE"
            confidence = avg_prob_fake
        else:
            label = "REAL"
            confidence = 1.0 - avg_prob_fake

        # Extract acoustic diagnostics
        diagnostics = self.extractor.compute_acoustic_diagnostics(processed_y)

        # Downsample waveform for frontend visualization (120 points)
        downsample_factor = max(1, len(processed_y) // 120)
        downsampled_waveform = [round(float(x), 3) for x in processed_y[::downsample_factor][:120]]

        return {
            "prediction": {
                "label": label,  # Strictly "REAL" or "FAKE"
                "confidence_score": round(confidence, 4),
                "confidence_percent": f"{round(confidence * 100.0, 1)}%",
                "probability_fake": round(avg_prob_fake, 4),
                "probability_real": round(1.0 - avg_prob_fake, 4),
                "model_used": active_model_name,
                "model_description": model_desc,
                "confidence_disclaimer": "Confidence is an algorithmic estimate based on acoustic feature analysis and is not an absolute guarantee of correctness."
            },
            "audio_metadata": {
                "filename": filename,
                "format": meta["format"],
                "original_sample_rate": meta["original_sr"],
                "target_sample_rate": 16000,
                "channels": meta["channels"],
                "duration_seconds": meta["duration_seconds"],
                "num_segments_analyzed": len(segments)
            },
            "acoustic_diagnostics": diagnostics,
            "waveform_preview": downsampled_waveform
        }
