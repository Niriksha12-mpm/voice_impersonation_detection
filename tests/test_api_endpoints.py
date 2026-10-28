"""
Automated Integration Tests for FastAPI Backend Endpoints.
Verifies compliance with Phase 8:
1. Health check returns 200 and reports healthy models.
2. Models endpoint returns Model A and Model B specifications.
3. Comparison endpoint returns genuine evaluation benchmark.
4. /api/predict endpoint accepts audio uploads, returns strictly 'REAL' or 'FAKE',
   includes confidence scores, acoustic diagnostics, and disclaimers.
5. Error handling for corrupted, empty, or unsupported audio formats.
"""

import os
import io
import json
import numpy as np
import soundfile as sf
import pytest
from fastapi.testclient import TestClient

from backend.app import app

client = TestClient(app)


def generate_test_audio_bytes(freq=440.0, duration=3.0, sr=16000) -> bytes:
    """Generate in-memory WAV audio bytes for testing."""
    samples = int(sr * duration)
    t = np.linspace(0, duration, samples, endpoint=False, dtype=np.float32)
    y = 0.5 * np.sin(2 * np.pi * freq * t)
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV")
    buf.seek(0)
    return buf.read()


class TestApiEndpoints:
    def test_health_check(self):
        """GET /api/health must return status healthy."""
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "models_loaded" in data

    def test_get_models_info(self):
        """GET /api/models must list Model A and Model B with configurations."""
        response = client.get("/api/models")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert len(data["models"]) == 2
        model_ids = [m["id"] for m in data["models"]]
        assert "fine_tuned" in model_ids
        assert "pretrained" in model_ids

    def test_get_model_comparison(self):
        """GET /api/comparison must return genuine benchmark metrics and deltas."""
        response = client.get("/api/comparison")
        assert response.status_code == 200
        data = response.json()
        assert "held_out_test_samples" in data
        assert "model_b_improved" in data
        assert "improvement_delta" in data
        assert data["improvement_delta"]["meets_target_improvement"] is True

    def test_predict_fine_tuned_real_audio(self):
        """POST /api/predict with fine_tuned model must return strictly 'REAL' or 'FAKE'."""
        audio_bytes = generate_test_audio_bytes(freq=300.0, duration=3.0)
        files = {"file": ("test_voice.wav", audio_bytes, "audio/wav")}
        response = client.post("/api/predict?model_type=fine_tuned", files=files)
        assert response.status_code == 200
        res = response.json()

        assert "prediction" in res
        pred = res["prediction"]
        assert pred["label"] in ["REAL", "FAKE"]
        assert 0.0 <= pred["confidence_score"] <= 1.0
        assert "confidence_disclaimer" in pred
        assert "acoustic_diagnostics" in res
        assert "waveform_preview" in res

    def test_predict_pretrained_model(self):
        """POST /api/predict with model_type=pretrained executes baseline."""
        audio_bytes = generate_test_audio_bytes(freq=440.0, duration=2.0)
        files = {"file": ("test_sample.wav", audio_bytes, "audio/wav")}
        response = client.post("/api/predict?model_type=pretrained", files=files)
        assert response.status_code == 200
        res = response.json()
        assert res["prediction"]["label"] in ["REAL", "FAKE"]
        assert "Pretrained" in res["prediction"]["model_used"]

    def test_predict_invalid_model_type(self):
        """Invalid model_type must return 400 Bad Request."""
        audio_bytes = generate_test_audio_bytes()
        files = {"file": ("sample.wav", audio_bytes, "audio/wav")}
        response = client.post("/api/predict?model_type=unknown_model", files=files)
        assert response.status_code == 400

    def test_predict_rejects_empty_corrupt_file(self):
        """Empty file upload must return 422 Unprocessable Entity."""
        files = {"file": ("empty.wav", b"too short", "audio/wav")}
        response = client.post("/api/predict", files=files)
        assert response.status_code == 422

    def test_predict_rejects_unsupported_format(self):
        """Unsupported file extension (e.g. .txt, .pdf) must return 422."""
        files = {"file": ("document.pdf", b"%PDF-1.4 dummy", "application/pdf")}
        response = client.post("/api/predict", files=files)
        assert response.status_code == 422


if __name__ == "__main__":
    pytest.main(["-v", __file__])
