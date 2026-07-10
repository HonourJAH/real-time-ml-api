from fastapi.testclient import TestClient

from app.main import app

NORMAL_EVENT = {
    "amount": 30.0,
    "hour": 13,
    "merchant_category": 2,
    "distance_from_home_km": 3.0,
    "velocity_last_hour": 1,
}
ANOMALOUS_EVENT = {
    "amount": 4000.0,
    "hour": 3,
    "merchant_category": 8,
    "distance_from_home_km": 2500.0,
    "velocity_last_hour": 18,
}


class TestTrainEndpoint:
    def test_train_returns_201_with_correct_shape(self):
        client = TestClient(app)
        response = client.post("/train", json={"n_samples": 1000, "anomaly_rate": 0.05})
        assert response.status_code == 201
        body = response.json()
        assert body["n_samples"] == 1000
        assert body["contamination"] == 0.05
        assert "model_path" in body

    def test_train_uses_defaults_when_body_omitted(self):
        client = TestClient(app)
        response = client.post("/train", json={})
        assert response.status_code == 201
        assert response.json()["n_samples"] == 3000

    def test_train_rejects_invalid_anomaly_rate(self):
        client = TestClient(app)
        response = client.post("/train", json={"anomaly_rate": 1.5})
        assert response.status_code == 422


class TestHealthEndpoint:
    def test_health_false_before_training(self):
        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy", "model_trained": False}

    def test_health_true_after_training(self):
        client = TestClient(app)
        client.post("/train", json={"n_samples": 1000})
        response = client.get("/health")
        assert response.json()["model_trained"] is True


class TestWebSocketStreaming:
    def test_scores_normal_event_correctly(self):
        client = TestClient(app)
        client.post("/train", json={"n_samples": 2000, "anomaly_rate": 0.05})

        with client.websocket_connect("/ws/stream") as ws:
            ws.send_json(NORMAL_EVENT)
            response = ws.receive_json()

        assert response["is_anomaly"] is False
        assert "anomaly_score" in response
        assert "latency_ms" in response
        assert response["event"] == NORMAL_EVENT

    def test_scores_anomalous_event_correctly(self):
        client = TestClient(app)
        client.post("/train", json={"n_samples": 2000, "anomaly_rate": 0.05})

        with client.websocket_connect("/ws/stream") as ws:
            ws.send_json(ANOMALOUS_EVENT)
            response = ws.receive_json()

        assert response["is_anomaly"] is True

    def test_anomaly_score_ordering_matches_expectation(self):
        client = TestClient(app)
        client.post("/train", json={"n_samples": 2000, "anomaly_rate": 0.05})

        with client.websocket_connect("/ws/stream") as ws:
            ws.send_json(NORMAL_EVENT)
            normal_response = ws.receive_json()
            ws.send_json(ANOMALOUS_EVENT)
            anomalous_response = ws.receive_json()

        assert anomalous_response["anomaly_score"] > normal_response["anomaly_score"]

    def test_invalid_event_returns_error_but_keeps_connection_alive(self):
        client = TestClient(app)
        client.post("/train", json={"n_samples": 1000})

        with client.websocket_connect("/ws/stream") as ws:
            ws.send_json({"amount": 10.0})  # missing required fields
            error_response = ws.receive_json()
            assert "error" in error_response

            # connection must still work after the error
            ws.send_json(NORMAL_EVENT)
            good_response = ws.receive_json()
            assert good_response["is_anomaly"] is False

    def test_scoring_before_training_returns_error_not_crash(self):
        client = TestClient(app)
        # deliberately no /train call in this test

        with client.websocket_connect("/ws/stream") as ws:
            ws.send_json(NORMAL_EVENT)
            response = ws.receive_json()

        assert "error" in response
        assert "No trained model found" in response["error"]

    def test_multiple_events_in_sequence_on_one_connection(self):
        client = TestClient(app)
        client.post("/train", json={"n_samples": 2000, "anomaly_rate": 0.05})

        with client.websocket_connect("/ws/stream") as ws:
            results = []
            for event in [NORMAL_EVENT, ANOMALOUS_EVENT, NORMAL_EVENT]:
                ws.send_json(event)
                results.append(ws.receive_json())

        assert [r["is_anomaly"] for r in results] == [False, True, False]
