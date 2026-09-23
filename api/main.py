"""
api/main.py - FastAPI service that serves the trained CatDogCNN model.

Endpoints:
    GET  /         -> HTML upload page
    GET  /health   -> {"status": "ok", "model_loaded": true}
    POST /predict  -> multipart image upload -> {"label", "confidence"}

Run locally:
    uvicorn api.main:app --reload --port 8000
"""

import json
import os
import sys

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.predict import load_model, predict_image

app = FastAPI(title="Cats vs Dogs Classifier", version="1.0.0",
              description="From-scratch CNN binary image classifier")

_model = None
_device = None
_norm = None
_metrics_cache = None


def get_model():
    """Lazily load the trained model once and reuse it across requests."""
    global _model, _device, _norm
    if _model is None:
        _model, _device, _norm = load_model()
    return _model, _device, _norm


@app.get("/")
def index() -> FileResponse:
    return FileResponse(os.path.join(os.path.dirname(__file__), "static", "index.html"))


@app.get("/dashboard")
def dashboard() -> FileResponse:
    """Beautiful metrics + live-prediction dashboard."""
    return FileResponse(
        os.path.join(os.path.dirname(__file__), "static", "dashboard.html"))


@app.get("/api/metrics")
def api_metrics() -> JSONResponse:
    """All evaluation metrics for the dashboard (generated on first call)."""
    global _metrics_cache
    if _metrics_cache is not None:
        return JSONResponse(_metrics_cache)

    json_path = os.path.join(config.OUTPUTS_DIR, "metrics.json")
    if not os.path.exists(json_path):
        # Generate once: evaluate the model and dump outputs/metrics.json
        # (On hosts without the dataset this can't run; return 503 with a hint.)
        try:
            from src.evaluate import generate_metrics_json
            config.ensure_dirs()
            generate_metrics_json(json_path)
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail="Metrics are not available yet on this host. "
                       "Run `python src/evaluate.py` and redeploy, or check logs.")

    try:
        with open(json_path) as f:
            _metrics_cache = json.load(f)
    except FileNotFoundError:
        raise HTTPException(status_code=404,
                            detail="No metrics available. Train the model first.")

    return JSONResponse(_metrics_cache)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    if not file.filename or not (file.content_type or "").startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file received")

    config.ensure_dirs()
    tmp = os.path.join(config.OUTPUTS_DIR, "_upload_tmp.jpg")
    try:
        with open(tmp, "wb") as f:
            f.write(data)
        model, device, norm = get_model()
        label, confidence = predict_image(tmp, model=model, device=device, norm=norm)
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001 - surface any image/decoding error
        raise HTTPException(status_code=422, detail=f"Could not process image: {e}")
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

    return {"label": label, "confidence": round(float(confidence), 2)}


# UI static assets
_static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(_static_dir):
    app.mount("/static", StaticFiles(directory=_static_dir), name="static")

# Diagnostic images (sample predictions, confusion matrix) for the dashboard
if os.path.isdir(config.OUTPUTS_DIR):
    app.mount("/outputs", StaticFiles(directory=config.OUTPUTS_DIR), name="outputs")

# Sample test images used by the dashboard quickly-try buttons
if os.path.isdir(config.TEST_IMAGES_DIR):
    app.mount("/test", StaticFiles(directory=config.TEST_IMAGES_DIR), name="test")