import os

import pytest

from app.services.anomaly_detector import (
    train,
    score,
    is_trained,
    ModelNotTrainedError,
)
from app.services.event_generator import generate_batch

# Deliberately extreme, hand-built feature vectors — not generated via
# event_generator — so these tests aren't relying on the generator's own
# correctness (that's covered separately in test_event_generator.py).
CLEARLY_NORMAL_EVENT = {
    "amount": 30.0,
    "hour": 13,
    "merchant_category": 2,
    "distance_from_home_km": 3.0,
    "velocity_last_hour": 1,
}
CLEARLY_ANOMALOUS_EVENT = {
    "amount": 4000.0,
    "hour": 3,
    "merchant_category": 8,
    "distance_from_home_km": 2500.0,
    "velocity_last_hour": 18,
}


def _train_on_synthetic_batch(n=1500, anomaly_rate=0.05):
    batch = generate_batch(n, anomaly_rate=anomaly_rate)
    events_only = [event for event, _ in batch]
    return train(events_only, contamination=anomaly_rate)


class TestBeforeTraining:
    def test_is_trained_false_when_no_model_exists(self):
        assert is_trained() is False

    def test_score_raises_clear_error_before_training(self):
        with pytest.raises(ModelNotTrainedError, match="No trained model found"):
            score(CLEARLY_NORMAL_EVENT)


class TestTraining:
    def test_train_persists_model_file_to_disk(self):
        result = _train_on_synthetic_batch()
        assert os.path.exists(result["model_path"])

    def test_train_returns_correct_metadata(self):
        result = _train_on_synthetic_batch(n=1234, anomaly_rate=0.07)
        assert result["n_samples"] == 1234
        assert result["contamination"] == 0.07

    def test_is_trained_true_immediately_after_training(self):
        _train_on_synthetic_batch()
        assert is_trained() is True

    def test_is_trained_true_from_disk_even_after_cache_cleared(self):
        """Confirms is_trained() checks disk, not just the in-memory
        cache — important since a fresh process restart would have no
        in-memory cache but should still recognize an existing model.
        """
        _train_on_synthetic_batch()
        import app.services.anomaly_detector as detector_module

        detector_module._model = None
        assert is_trained() is True


class TestScoring:
    def test_score_returns_expected_keys(self):
        _train_on_synthetic_batch()
        result = score(CLEARLY_NORMAL_EVENT)
        assert set(result.keys()) == {"anomaly_score", "is_anomaly"}

    def test_clearly_normal_event_scored_as_not_anomalous(self):
        _train_on_synthetic_batch()
        result = score(CLEARLY_NORMAL_EVENT)
        assert result["is_anomaly"] is False

    def test_clearly_anomalous_event_scored_as_anomalous(self):
        _train_on_synthetic_batch()
        result = score(CLEARLY_ANOMALOUS_EVENT)
        assert result["is_anomaly"] is True

    def test_anomaly_score_is_higher_for_anomalous_than_normal(self):
        """The core sign-convention contract: higher anomaly_score must
        mean more anomalous, verified against IsolationForest's own
        decision_function() (which uses the opposite sign).
        """
        _train_on_synthetic_batch()
        normal_result = score(CLEARLY_NORMAL_EVENT)
        anomalous_result = score(CLEARLY_ANOMALOUS_EVENT)
        assert anomalous_result["anomaly_score"] > normal_result["anomaly_score"]

    def test_model_loads_correctly_from_disk_after_cache_cleared(self):
        """Confirms score() can load a persisted model fresh from disk —
        simulates what happens after a process restart, where training
        happened in a previous run and this run only ever calls score().
        """
        _train_on_synthetic_batch()
        import app.services.anomaly_detector as detector_module

        detector_module._model = None  # force a real disk load, not cache

        result = score(CLEARLY_ANOMALOUS_EVENT)
        assert result["is_anomaly"] is True

    def test_accuracy_against_generator_ground_truth_is_high(self):
        """End-to-end sanity check: train and score using the SAME
        generator, confirming the whole pipeline actually works together,
        not just each piece in isolation.
        """
        batch = generate_batch(2000, anomaly_rate=0.05)
        events_only = [event for event, _ in batch]
        train(events_only, contamination=0.05)

        correct = sum(
            1 for event, is_actual in batch if score(event)["is_anomaly"] == is_actual
        )
        accuracy = correct / len(batch)
        assert accuracy > 0.95
