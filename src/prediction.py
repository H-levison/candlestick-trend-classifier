"""Single-image inference against the trained candlestick model."""

import os

from src.model import DECISION_THRESHOLD, load_model
from src.preprocessing import preprocess_image

# .keras (native Keras 3 format) rather than .h5: legacy H5 full-model saving
# hits a "cannot pickle 'module' object" error partway through in this
# Keras/TF version (a known Keras 3 legacy-H5 serialization issue), and
# save_format="tf" SavedModel export was removed outright ("deprecated in
# Keras 3"). .keras is the actively-maintained native format.
DEFAULT_MODEL_PATH = os.environ.get("MODEL_PATH", "models/candlestick_model.keras")
CLASS_NAMES = ["Down", "Up"]  # sigmoid output: 0 -> Down, 1 -> Up

_cached_model = None
_cached_model_path = None


def get_model(model_path=DEFAULT_MODEL_PATH):
    """Loads the model once and reuses it across requests. Call
    reload_model() after retraining so the new weights take effect
    without restarting the API process."""
    global _cached_model, _cached_model_path
    if _cached_model is None or _cached_model_path != model_path:
        _cached_model = load_model(model_path)
        _cached_model_path = model_path
    return _cached_model


def reload_model(model_path=DEFAULT_MODEL_PATH):
    global _cached_model, _cached_model_path
    _cached_model = load_model(model_path)
    _cached_model_path = model_path
    return _cached_model


def predict_image(image_bytes_or_path, model_path=DEFAULT_MODEL_PATH):
    """Returns predicted class ("Up"/"Down"), a confidence score, and the
    raw sigmoid probability of the "Up" class (in [0, 1]). Uses
    DECISION_THRESHOLD (validation-tuned, see model.py), not a naive 0.5
    cutoff -- see model.py for why that matters for this model."""
    model = get_model(model_path)
    arr = preprocess_image(image_bytes_or_path)
    raw_probability = float(model.predict(arr, verbose=0)[0][0])

    is_up = raw_probability >= DECISION_THRESHOLD
    prediction = CLASS_NAMES[1] if is_up else CLASS_NAMES[0]
    # Confidence is not just distance from 0.5 anymore since the decision
    # boundary moved -- express it as distance from the decision threshold,
    # rescaled into the class's own probability range.
    if is_up:
        confidence = (raw_probability - DECISION_THRESHOLD) / (1.0 - DECISION_THRESHOLD) * 0.5 + 0.5
    else:
        confidence = (DECISION_THRESHOLD - raw_probability) / DECISION_THRESHOLD * 0.5 + 0.5

    return {
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "raw_probability": round(raw_probability, 4),
    }
