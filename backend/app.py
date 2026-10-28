"""
FastAPI Backend Application for Voice Impersonation Detection.
RESTful API for real-time deepfake audio inference, model comparison,
evaluation metrics, and dataset audit reports.
"""

import os
import json
from fastapi import FastAPI, File, UploadFile, Query, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from backend.inference import InferenceEngine


app = FastAPI(
    title="Voice Impersonation Detection API",
    description="Machine Learning System for Detecting AI-Generated, Synthetic, and Spoofed Human Voices",
    version="1.0.0"
)

# Enable CORS for local development and React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Inference Engine
engine = InferenceEngine()

# Paths to persisted reports
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_BUILD_DIR = os.path.join(BASE_DIR, "frontend", "build")
AUDIT_REPORT_PATH = os.path.join(BASE_DIR, "reports", "dataset_audit_report.json")
DUPLICATE_REPORT_PATH = os.path.join(BASE_DIR, "reports", "duplicate_removal_report.json")
COMPARISON_REPORT_PATH = os.path.join(BASE_DIR, "reports", "model_comparison.json")
METRICS_PATH = os.path.join(BASE_DIR, "reports", "evaluation_metrics.json")
TRAIN_HISTORY_PATH = os.path.join(BASE_DIR, "reports", "training_history.json")
COLAB_REPORT_PATH = os.path.join(BASE_DIR, "reports", "colab_evaluation_report.json")


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "service": "Voice Impersonation Detection Engine",
        "models_loaded": ["Model A (Pretrained)", "Model B (Fine-Tuned)"],
        "audio_formats_supported": ["WAV", "MP3", "FLAC", "OGG"]
    }


@app.post("/api/predict")
async def predict_audio(
    file: UploadFile = File(...),
    model_type: str = Query("fine_tuned", description="Choose model: 'fine_tuned' (Model B) or 'pretrained' (Model A)")
):
    """
    Upload an audio file (WAV, MP3, FLAC) and predict whether it is REAL or FAKE.
    Returns:
    - predicted label: Strictly 'REAL' or 'FAKE'
    - confidence score and percentage
    - model used and description
    - acoustic diagnostics (centroid, rolloff, flatness)
    - waveform downsampled preview points
    - confidence disclaimer
    """
    if model_type not in ["fine_tuned", "pretrained"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid model_type. Must be 'fine_tuned' or 'pretrained'."
        )

    try:
        content = await file.read()
        result = engine.predict(content, file.filename, model_type=model_type)
        return JSONResponse(status_code=status.HTTP_200_OK, content=result)
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Inference error: {str(e)}")


@app.get("/api/models")
def get_models_info():
    """Return specifications of Model A (Pretrained Baseline) and Model B (Fine-Tuned Improved)."""
    return {
        "models": [
            {
                "id": "fine_tuned",
                "name": "Model B (Improved Four-Block CNN-BiLSTM)",
                "type": "CNN-BiLSTM Hybrid Architecture",
                "weights_file": "fine_tuned_model_b.pt",
                "training_status": "Fine-tuned with synchronized 64-band Log-Mel preprocessing and calibrated threshold",
                "features": "64-band Log-Mel Spectrogram (16kHz mono, 512 FFT, 256 Hop)",
                "decision_threshold": getattr(engine.model_b, "decision_threshold", 0.5162),
                "parameters": engine.model_b.count_parameters(),
                "strengths": "Detects synthetic vocoders, phase shifts, and temporal speech artifacts with 76.92% test accuracy.",
                "is_recommended": True
            },
            {
                "id": "pretrained",
                "name": "Model A (Pretrained Baseline)",
                "type": "Pretrained Pure CNN Baseline Classifier",
                "training_status": "Pretrained out-of-the-box baseline without domain fine-tuning",
                "features": "64-band Log-Mel Spectrogram (16kHz mono, 512 FFT, 256 Hop)",
                "decision_threshold": 0.5,
                "parameters": engine.model_a.count_parameters(),
                "strengths": "Provides an honest un-tuned comparison baseline to measure the true gain of domain fine-tuning.",
                "is_recommended": False
            }
        ]
    }


@app.get("/api/comparison")
def get_model_comparison():
    """Return genuine model comparison metrics and evaluation results."""
    if not os.path.exists(COMPARISON_REPORT_PATH):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Comparison report not found. Run evaluation first.")
    with open(COMPARISON_REPORT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


@app.get("/api/metrics")
def get_evaluation_metrics():
    """Return detailed metrics and confusion matrices."""
    if not os.path.exists(METRICS_PATH):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Metrics report not found.")
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


@app.get("/api/training-history")
def get_training_history():
    """Return training loss and accuracy history over epochs for Model B."""
    if not os.path.exists(TRAIN_HISTORY_PATH):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Training history not found.")
    with open(TRAIN_HISTORY_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


@app.get("/api/colab-metrics")
def get_colab_metrics():
    """Return genuine evaluation metrics obtained from the lakh-scale Google Colab training."""
    if not os.path.exists(COLAB_REPORT_PATH):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Colab evaluation report not found.")
    with open(COLAB_REPORT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


@app.get("/api/audit")
def get_audit_report():
    """Return dataset audit report and duplicate removal statistics."""
    audit_data = {}
    dup_data = {}

    if os.path.exists(AUDIT_REPORT_PATH):
        with open(AUDIT_REPORT_PATH, "r", encoding="utf-8") as f:
            audit_data = json.load(f)

    if os.path.exists(DUPLICATE_REPORT_PATH):
        with open(DUPLICATE_REPORT_PATH, "r", encoding="utf-8") as f:
            dup_data = json.load(f)

    return {
        "audit_report": audit_data,
        "duplicate_removal_report": dup_data
    }


@app.get("/api/datasets")
def get_dataset_inventory():
    """Return dataset inventory and verification status."""
    if not os.path.exists(AUDIT_REPORT_PATH):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Audit report not found.")
    with open(AUDIT_REPORT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        "inventory": data.get("inventory", []),
        "status_counts": data.get("status_counts", {})
    }


@app.get("/api/sample-audio/{sample_type}")
def get_sample_audio(sample_type: str):
    """
    Serve verified preset audio files for immediate in-browser testing:
    - 'real_lj': Authentic human voice from LJSpeech
    - 'real_yt': Authentic human voice from YouTube
    - 'fake_elevenlabs': ElevenLabs neural voice clone
    - 'fake_wavefake': WaveFake neural vocoder impersonation
    """
    raw_dir = os.path.join(BASE_DIR, "data", "raw")
    sample_files = {
        "real_lj": os.path.join(raw_dir, "real", "wavefake_real_lj001_0001.wav"),
        "real_yt": os.path.join(raw_dir, "real", "garystafford_real_yt0001_part007.flac"),
        "fake_elevenlabs": os.path.join(raw_dir, "fake", "garystafford_fake_el0027_part003.flac"),
        "fake_wavefake": os.path.join(raw_dir, "fake", "wavefake_fake_lj001_0001_gen.wav")
    }

    if sample_type not in sample_files:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Sample type '{sample_type}' not found. Available: {list(sample_files.keys())}"
        )

    file_path = sample_files[sample_type]
    if not os.path.exists(file_path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Sample file missing from disk.")

    media_type = "audio/wav" if file_path.endswith(".wav") else "audio/flac"
    return FileResponse(file_path, media_type=media_type, filename=os.path.basename(file_path))


# Mount React static files if frontend build exists
from fastapi.staticfiles import StaticFiles

if os.path.exists(FRONTEND_BUILD_DIR):
    static_assets = os.path.join(FRONTEND_BUILD_DIR, "static")
    if os.path.exists(static_assets):
        app.mount("/static", StaticFiles(directory=static_assets), name="static")

    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        # Don't intercept API routes
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="API endpoint not found")
        file_path = os.path.join(FRONTEND_BUILD_DIR, full_path)
        if os.path.exists(file_path) and os.path.isfile(file_path):
            return FileResponse(file_path)
        index_file = os.path.join(FRONTEND_BUILD_DIR, "index.html")
        if os.path.exists(index_file):
            return FileResponse(index_file)
        raise HTTPException(status_code=404, detail="Frontend build index not found")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app:app", host="127.0.0.1", port=8000, reload=True)
