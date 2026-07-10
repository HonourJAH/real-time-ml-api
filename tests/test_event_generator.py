import statistics

from app.services.event_generator import (
    generate_normal_event,
    generate_anomalous_event,
    generate_event,
    generate_batch,
)

EXPECTED_FIELDS = {
    "event_id",
    "timestamp",
    "amount",
    "hour",
    "merchant_category",
    "distance_from_home_km",
    "velocity_last_hour",
}


class TestEventSchema:
    def test_normal_event_has_all_expected_fields(self):
        event = generate_normal_event()
        assert set(event.keys()) == EXPECTED_FIELDS

    def test_anomalous_event_has_all_expected_fields(self):
        event = generate_anomalous_event()
        assert set(event.keys()) == EXPECTED_FIELDS

    def test_hour_always_within_valid_range(self):
        for _ in range(200):
            assert 0 <= generate_normal_event()["hour"] <= 23
            assert 0 <= generate_anomalous_event()["hour"] <= 23

    def test_event_ids_are_unique(self):
        events = [generate_normal_event() for _ in range(500)]
        ids = {e["event_id"] for e in events}
        assert len(ids) == 500


class TestAnomalyRate:
    def test_generate_event_respects_anomaly_rate_over_large_sample(self):
        results = [generate_event(anomaly_rate=0.1)[1] for _ in range(5000)]
        actual_rate = sum(results) / len(results)
        # Statistical, not exact — allow a reasonable margin
        assert 0.08 <= actual_rate <= 0.12

    def test_zero_anomaly_rate_produces_no_anomalies(self):
        results = [generate_event(anomaly_rate=0.0)[1] for _ in range(200)]
        assert not any(results)

    def test_generate_batch_returns_requested_count(self):
        batch = generate_batch(250, anomaly_rate=0.05)
        assert len(batch) == 250


class TestStatisticalSeparation:
    """Confirms normal and anomalous events are genuinely distinguishable
    — the entire premise the anomaly detector depends on.
    """

    def test_anomalous_amounts_are_far_higher_than_normal(self):
        normal = [generate_normal_event()["amount"] for _ in range(500)]
        anomalous = [generate_anomalous_event()["amount"] for _ in range(500)]
        assert max(normal) < min(anomalous)

    def test_anomalous_distance_is_far_higher_than_normal(self):
        normal = [generate_normal_event()["distance_from_home_km"] for _ in range(500)]
        anomalous = [
            generate_anomalous_event()["distance_from_home_km"] for _ in range(500)
        ]
        assert max(normal) < min(anomalous)

    def test_anomalous_velocity_is_higher_than_normal(self):
        normal = [generate_normal_event()["velocity_last_hour"] for _ in range(500)]
        anomalous = [
            generate_anomalous_event()["velocity_last_hour"] for _ in range(500)
        ]
        assert statistics.mean(anomalous) > statistics.mean(normal) * 3
