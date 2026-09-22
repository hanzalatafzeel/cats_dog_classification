"""
api/main.py - FastAPI service that serves the trained CatDogCNN model.

Endpoints:
    GET  /         -> HTML upload page
    GET  /health   -> {"status": "ok", "model_loaded": true}
    POST /predict  -> multipart image upload -> {"label", "confidence"}

Run locally:
    uvicorn api.main:app --reload --port 8000
"""

import os
import sys

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.predict import load_model, predict_image

app = FastAPI(title="Cats vs Dogs Classifier", version="1.0.0",
              description="From-scratch CNN binary image classifier")

_model = None
_device = None
_norm = None


def get_model():
    """Lazily load the trained model once and reuse it across requests."""
    global _model, _device, _norm
    if _model is None:
        _model, _device, _norm = load_model()
    return _model, _device, _norm


@app.get("/")
def index() -> FileResponse:
    return FileResponse(os.path.join(os.path.dirname(__file__), "static", "index.html"))


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