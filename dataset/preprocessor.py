"""
Audio Preprocessing and Standardization Module.
Standardizes audio files to:
- Sample rate: 16,000 Hz
- Channels: Mono (1 channel)
- Silence trimming: Remove leading and trailing silence (< -40 dB)
- Amplitude normalization: Peak / RMS normalization (-1.0 to 1.0)
- Fixed duration framing: Standard 3.0 second clips (48,000 samples)
  (Pad with reflection/zero if short, chunk into multiple 3s segments if long)
- Validation: Validates audio validity, rejects corrupted files, silent files, and unsupported formats.
"""

import os
import json
import numpy as np
import soundfile as sf
import librosa
from typing import List, Dict, Tuple, Any, Optional


class AudioPreprocessor:
    def __init__(self, target_sr: int = 16000, target_duration: float = 3.0, top_db: float = 40.0):
        self.target_sr = target_sr
        self.target_duration = target_duration
        self.target_samples = int(target_sr * target_duration)
        self.top_db = top_db

    def validate_audio(self, y: np.ndarray, sr: int) -> Tuple[bool, str]:
        """Validate audio signal. Reject silence, NaN, empty arrays, or flat signals."""
        if y is None or y.size == 0:
            return False, "Audio is empty or could not be decoded."
        if np.isnan(y).any() or np.isinf(y).any():
            return False, "Audio contains NaN or Inf values."
        peak = float(np.max(np.abs(y)))
        if peak < 1e-4:
            return False, "Audio is silent (RMS energy below audible threshold)."
        num_samples = y.shape[-1] if y.ndim > 1 else len(y)
        if num_samples < int(sr * 0.2):  # less than 200ms
            return False, f"Audio duration too short ({num_samples/sr:.2f}s). Minimum 0.2s required."
        return True, "Valid"

    def process_waveform(self, y: np.ndarray, orig_sr: int) -> np.ndarray:
        """Resample to 16kHz, convert to mono, trim silence, and normalize."""
        # Convert multi-channel to mono
        if y.ndim > 1:
            y = np.mean(y, axis=0 if y.shape[0] < y.shape[1] else 1)

        # Resample to 16kHz if needed
        if orig_sr != self.target_sr:
            y = librosa.resample(y, orig_sr=orig_sr, target_sr=self.target_sr)

        # Trim leading and trailing silence
        trimmed_y, _ = librosa.effects.trim(y, top_db=self.top_db)
        if len(trimmed_y) >= int(self.target_sr * 0.2):
            y = trimmed_y

        # Peak normalization with headroom
        max_val = np.max(np.abs(y))
        if max_val > 1e-5:
            y = y / max_val * 0.95

        return y.astype(np.float32)

    def segment_or_pad(self, y: np.ndarray) -> List[np.ndarray]:
        """
        Produce one or more fixed 3.0s (48,000 samples) segments:
        - If shorter than 3s, pad with zero / repeat padding.
        - If longer, slice into non-overlapping 3s segments.
        """
        segments = []
        if len(y) < self.target_samples:
            # Pad
            padded = np.zeros(self.target_samples, dtype=np.float32)
            padded[:len(y)] = y
            segments.append(padded)
        else:
            # Slice non-overlapping segments
            num_segments = len(y) // self.target_samples
            for i in range(num_segments):
                seg = y[i * self.target_samples : (i + 1) * self.target_samples]
                segments.append(seg)

            # If remaining tail is at least 1.5s, pad the tail to 3s
            tail_len = len(y) % self.target_samples
            if tail_len >= int(self.target_sr * 1.5):
                tail = np.zeros(self.target_samples, dtype=np.float32)
                tail[:tail_len] = y[-tail_len:]
                segments.append(tail)

        return segments

    def process_file(self, filepath: str) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """Load, validate, process, and return 3s segments and metadata."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        # Check extension
        ext = os.path.splitext(filepath)[1].lower()
        if ext not in [".wav", ".mp3", ".flac", ".ogg", ".m4a"]:
            raise ValueError(f"Unsupported audio format: {ext}. Supported: WAV, MP3, FLAC, OGG.")

        try:
            y, sr = librosa.load(filepath, sr=None, mono=False)
        except Exception as e:
            raise ValueError(f"Corrupted or unreadable audio file: {e}")

        valid, msg = self.validate_audio(y, sr)
        if not valid:
            raise ValueError(f"Audio validation failed: {msg}")

        processed_y = self.process_waveform(y, sr)
        segments = self.segment_or_pad(processed_y)

        meta = {
            "original_sr": sr,
            "target_sr": self.target_sr,
            "raw_duration": round(len(y) / sr if y.ndim == 1 else y.shape[-1] / sr, 3),
            "processed_duration": round(len(processed_y) / self.target_sr, 3),
            "num_segments": len(segments),
            "segment_samples": self.target_samples
        }
        return segments, meta


def preprocess_clean_dataset(clean_manifest_path: str = "data/raw/clean_manifest.json",
                             output_dir: str = "data/processed") -> List[Dict[str, Any]]:
    """
    Process all files from clean manifest, save standardized 16kHz WAV segments,
    and output processed dataset manifest.
    """
    with open(clean_manifest_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    os.makedirs(os.path.join(output_dir, "real"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "fake"), exist_ok=True)

    preprocessor = AudioPreprocessor(target_sr=16000, target_duration=3.0)
    processed_manifest = []

    print(f"[Preprocessor] Preprocessing {len(records)} clean recordings into 16kHz mono standardized segments...")

    for r in records:
        src_path = r["file_path"]
        label = r["label"]
        speaker_id = r["speaker_id"]
        dataset_source = r["dataset_source"]
        algorithm = r["algorithm"]
        subfolder = "real" if label == "REAL" else "fake"

        try:
            segments, meta = preprocessor.process_file(src_path)
            base_name = os.path.splitext(r["filename"])[0]

            for s_idx, seg in enumerate(segments):
                seg_filename = f"{base_name}_seg{s_idx:02d}.wav"
                seg_path = os.path.join(output_dir, subfolder, seg_filename)

                sf.write(seg_path, seg, 16000, subtype="PCM_16")

                processed_manifest.append({
                    "segment_filename": seg_filename,
                    "segment_path": os.path.abspath(seg_path),
                    "parent_file": r["filename"],
                    "label": label,
                    "speaker_id": speaker_id,
                    "dataset_source": dataset_source,
                    "algorithm": algorithm,
                    "segment_index": s_idx,
                    "sample_rate": 16000,
                    "channels": 1,
                    "duration_seconds": 3.0,
                    "samples": 48000
                })
        except Exception as e:
            print(f"[Preprocessor] Error processing {r['filename']}: {e}")

    out_manifest_path = os.path.join(output_dir, "processed_manifest.json")
    with open(out_manifest_path, "w", encoding="utf-8") as f:
        json.dump(processed_manifest, f, indent=2)

    real_count = sum(1 for x in processed_manifest if x["label"] == "REAL")
    fake_count = sum(1 for x in processed_manifest if x["label"] == "FAKE")
    print(f"[Preprocessor] Done! Generated {len(processed_manifest)} standardized 3.0s segments.")
    print(f"               REAL: {real_count}, FAKE: {fake_count}")
    print(f"               Manifest saved to {out_manifest_path}")
    return processed_manifest


if __name__ == "__main__":
    preprocess_clean_dataset()
