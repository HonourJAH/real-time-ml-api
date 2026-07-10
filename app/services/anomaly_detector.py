import os

import joblib
from sklearn.ensemble import IsolationForest

MODEL_PATH = os.getenv("MODEL_PATH", "models/isolation_forest.joblib")

# Order matters — this exact order is used both when training and when
# scoring a live event, so the feature vector always means the same thing
# to the model regardless of which code path built it.
FEATURE_NAMES = [
    "amount",
    "hour",
    "merchant_category",
    "distance_from_home_km",
    "velocity_last_hour",
]

# In-memory cache so score() doesn't hit disk on every single call — the
# model is loaded once and reused, similar in spirit to a shared client
# elsewhere in this portfolio, just for a local file instead of a network
# connection.
_model: IsolationForest | None = None


class ModelNotTrainedError(RuntimeError):
    """Raised when score() is called before any model has been trained
    or persisted to disk — a clear, specific error rather than a
    confusing FileNotFoundError surfacing from inside joblib.
    """


def _extract_features(event: dict) -> list[float]:
    return [float(event[name]) for name in FEATURE_NAMES]


def train(events: list[dict], contamination: float = 0.05) -> dict:
    """Fit a new IsolationForest on a batch of events and persist it to
    disk. `contamination` is IsolationForest's expected proportion of
    anomalies in the data — should roughly match the anomaly_rate used
    to generate the training batch.

    Unsupervised: events are plain feature dicts, no labels needed or
    used, even if the caller's generator happens to know which ones were
    actually anomalous (that's for evaluation only, never fed to the model).
    """
    X = [_extract_features(e) for e in events]

    model = IsolationForest(contamination=contamination, random_state=42)
    model.fit(X)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    global _model
    _model = model

    return {
        "n_samples": len(events),
        "contamination": contamination,
        "model_path": MODEL_PATH,
    }


def _load_model() -> IsolationForest:
    global _model
    if _model is not None:
        return _model

    if not os.path.exists(MODEL_PATH):
        raise ModelNotTrainedError(
            f"No trained model found at '{MODEL_PATH}'. Call /train first."
        )

    _model = joblib.load(MODEL_PATH)
    return _model


def is_trained() -> bool:
    return _model is not None or os.path.exists(MODEL_PATH)


def score(event: dict) -> dict:
    """Score a single event. Raises ModelNotTrainedError if no model has
    been trained/persisted yet.

    anomaly_score: higher = more anomalous (this is decision_function()
    negated — IsolationForest's own convention is the opposite, positive
    for normal, negative for anomalous, which reads backwards for a field
    literally named "anomaly_score").
    is_anomaly: IsolationForest's own classification, contamination-rate
    dependent (this is what predict() == -1 means).
    """
    model = _load_model()
    X = [_extract_features(event)]

    raw_decision = model.decision_function(X)[0]
    prediction = model.predict(X)[0]

    return {
        "anomaly_score": round(float(-raw_decision), 4),
        "is_anomaly": bool(prediction == -1),
    }
