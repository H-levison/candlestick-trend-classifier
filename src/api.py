"""FastAPI backend: prediction, bulk data upload, retrain trigger, insights."""

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src import database
from src.prediction import predict_image, reload_model, DEFAULT_MODEL_PATH
from src.model import retrain as run_retraining

APP_START_TIME = time.time()
TRAIN_DIR = "data/train"
TEST_DIR = "data/test"
MODEL_PATH = DEFAULT_MODEL_PATH
VALID_LABELS = {"Up", "Down"}

app = FastAPI(title="Crypto Candlestick Classifier API")

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# In-memory state -- fine for a single-process demo deployment; a real
# multi-worker deployment would move this into the SQLite DB instead.
_state = {"last_trained_at": None, "is_retraining": False, "model_version": 1}


@app.on_event("startup")
def on_startup():
    database.init_db()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": Path(MODEL_PATH).exists(),
        "is_retraining": _state["is_retraining"],
    }


@app.get("/uptime")
def uptime():
    seconds = time.time() - APP_START_TIME
    return {
        "uptime_seconds": round(seconds, 1),
        "started_at": datetime.fromtimestamp(APP_START_TIME, tz=timezone.utc).isoformat(),
        "model_version": _state["model_version"],
        "last_trained_at": _state["last_trained_at"],
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ("image/png", "image/jpeg", "image/jpg"):
        raise HTTPException(400, "Only PNG/JPG images are accepted")
    image_bytes = await file.read()
    try:
        return predict_image(image_bytes, model_path=MODEL_PATH)
    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {e}")


@app.post("/upload-data")
async def upload_data(label: str = Form(...), files: List[UploadFile] = File(...)):
    if label not in VALID_LABELS:
        raise HTTPException(400, "label must be 'Up' or 'Down'")

    target_dir = Path(TRAIN_DIR) / label
    target_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for f in files:
        contents = await f.read()
        dest = target_dir / f.filename
        dest.write_bytes(contents)
        database.insert_upload(f.filename, str(dest), label)
        saved += 1

    return {"saved": saved, "label": label, **database.get_upload_stats()}


@app.post("/retrain")
def retrain():
    """Fine-tunes the current model on data/train/ (manual trigger from the
    UI button, or called after get_upload_stats() flags auto_retrain_recommended).
    Runs synchronously -- FastAPI executes sync path functions in a threadpool,
    so /health stays responsive to polling while this runs. Deliberately not a
    background job queue: fine-tuning on this dataset size finishes in well
    under a minute, so the added complexity isn't worth it for this scope.
    """
    if _state["is_retraining"]:
        raise HTTPException(409, "Retraining already in progress")

    _state["is_retraining"] = True
    try:
        pending_ids = [p["id"] for p in database.get_pending_samples()]
        metrics = run_retraining(MODEL_PATH, TRAIN_DIR, TEST_DIR)
        reload_model(MODEL_PATH)
        database.mark_as_retrained(pending_ids)
        _state["last_trained_at"] = datetime.now(timezone.utc).isoformat()
        _state["model_version"] += 1
    finally:
        _state["is_retraining"] = False

    return {
        "status": "retrained",
        "model_version": _state["model_version"],
        "samples_used": len(pending_ids),
        "metrics_before": metrics["before"],
        "metrics_after": metrics["after"],
    }


@app.get("/insights")
def insights():
    def count(path):
        p = Path(path)
        return len(list(p.glob("*"))) if p.exists() else 0

    return {
        "train": {"Up": count(f"{TRAIN_DIR}/Up"), "Down": count(f"{TRAIN_DIR}/Down")},
        "test": {"Up": count(f"{TEST_DIR}/Up"), "Down": count(f"{TEST_DIR}/Down")},
        "uploads": database.get_upload_stats(),
    }
