"""Single-image inference against the trained candlestick model."""

from src.model import load_model
from src.preprocessing import preprocess_image

DEFAULT_MODEL_PATH = "models/candlestick_model.h5"
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
    """Returns predicted class ("Up"/"Down"), a confidence score (probability
    of the predicted class, in [0.5, 1]), and the raw sigmoid probability of
    the "Up" class (in [0, 1])."""
    model = get_model(model_path)
    arr = preprocess_image(image_bytes_or_path)
    raw_probability = float(model.predict(arr, verbose=0)[0][0])

    prediction = CLASS_NAMES[1] if raw_probability >= 0.5 else CLASS_NAMES[0]
    confidence = raw_probability if raw_probability >= 0.5 else 1.0 - raw_probability

    return {
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "raw_probability": round(raw_probability, 4),
    }
