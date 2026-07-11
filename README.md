# Real-Time ML API

A real-time anomaly detection service for streaming transaction events — built with FastAPI, scikit-learn's IsolationForest, and native WebSockets. No labeled data, no external services, sub-20ms inference per event.

---

## How It Works

```
POST /train              →  generate synthetic training data, fit an IsolationForest
WS   /ws/stream           →  send one event at a time, get back a live anomaly score
GET  /health              →  health check + whether a model is currently trained
```

---

## Table of Contents

- [Why Unsupervised Anomaly Detection?](#why-unsupervised-anomaly-detection)
- [Why WebSockets, Not Kafka?](#why-websockets-not-kafka)
- [Event Schema](#event-schema)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Getting Started](#getting-started)
- [Running Tests](#running-tests)
- [API Endpoints](#api-endpoints)
- [Request & Response Schemas](#request--response-schemas)
- [Example Usage](#example-usage)
- [Docker](#docker)

---

## Why Unsupervised Anomaly Detection?

Real fraud datasets are large, severely class-imbalanced, and need careful handling (SMOTE, class weights) just to train a supervised classifier correctly — a heavy lift for what this project is actually demonstrating. `IsolationForest` needs no labels at all: it learns what "normal" looks like from the data's own structure, and flags anything that's an outlier relative to that — exactly the anomaly-detection framing this project is built around.

```
Training data  → mostly normal events, ~5% synthetic anomalies mixed in
                  (contamination rate the model is told to expect)
IsolationForest → learns the shape of "normal" without ever seeing a label
Live event      → scored against that learned shape, not against fixed rules
```

Every event is scored independently — the model has no memory of previous events, so latency stays flat regardless of how long the stream has been running.

---

## Why WebSockets, Not Kafka?

Kafka needs its own broker (JVM-based, more moving parts, real memory overhead) to demonstrate exactly the same core skill this project is after: real-time, low-latency inference on a stream of events. FastAPI's native WebSocket support gets there with zero extra infrastructure — one persistent connection, one event in, one scored result out, no broker to run or maintain.

---

## Event Schema

Every event — synthetic or real — is a flat set of numeric features:

| Field | Type | Description |
|---|---|---|
| `amount` | float | Transaction amount |
| `hour` | int (0-23) | Hour of day |
| `merchant_category` | int (0-9) | Merchant category code |
| `distance_from_home_km` | float | Distance from the cardholder's home |
| `velocity_last_hour` | int | Number of transactions in the past hour |

**Synthetic data generation**, used both for training and demoing:

| | Normal events | Anomalous events |
|---|---|---|
| `amount` | Lognormal, mostly $15-$150 | Uniform, $500-$5000 |
| `hour` | Weighted toward daytime | Overnight (1am-4am) |
| `distance_from_home_km` | Exponential, mostly a few km | Uniform, 200-3000km |
| `velocity_last_hour` | 0-3, weighted toward 0-1 | 5-20 |

These ranges are deliberately non-overlapping — verified directly (`max(normal) < min(anomalous)` on every feature) so the model has a genuinely learnable signal, not noise.

---

## Project Structure

```
real-time-ml-api/
├── .github/
│   └── workflows/
│       └── ci.yml                  — GitHub Actions CI pipeline
├── app/
│   ├── main.py                      — FastAPI app: /train, /ws/stream, /health
│   ├── schemas.py                   — Request schemas, incl. WebSocket event validation
│   └── services/
│       ├── event_generator.py       — synthetic normal/anomalous event generator
│       └── anomaly_detector.py      — IsolationForest wrapper: train(), score()
├── models/                          — trained model persisted here (gitignored)
├── scripts/
│   └── simulate_stream.py           — standalone WebSocket client for live demos
├── tests/
│   ├── conftest.py                  — model/state isolation fixture
│   ├── test_event_generator.py
│   ├── test_anomaly_detector.py
│   └── test_main.py                 — includes real WebSocket connection tests
├── .dockerignore
├── .gitignore
├── Dockerfile
├── pytest.ini
├── README.md
└── requirements.txt
```

---

## Requirements

- Python 3.12+
- Docker (optional, for containerized runs)

No external services — no broker, no cache, no model API. Everything runs in a single process.

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/HonourJAH/real-time-ml-api.git
cd real-time-ml-api
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv venv
```

**Linux / macOS:**
```bash
source venv/bin/activate
```

**Windows (PowerShell):**
```powershell
venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the API

```bash
uvicorn app.main:app --reload
```

### 5. Train the model

```bash
curl -X POST http://localhost:8000/train -H "Content-Type: application/json" -d '{}'
```

### 6. Stream live events

```bash
python3 scripts/simulate_stream.py
```

API available at `http://localhost:8000`
Interactive docs at `http://localhost:8000/docs`

---

## Running Tests

Every test runs against the real `IsolationForest`/`event_generator` code — nothing here needs mocking, since there's no external service in the loop at all. Model state (the in-memory cache and the on-disk model path) is fully isolated per test via an autouse fixture, so tests never leak a trained model into each other.

```bash
pip install pytest
pytest -v
```

---

## API Endpoints

| Method | Endpoint | Description | Status Code |
|---|---|---|---|
| `POST` | `/train` | Generate synthetic data and fit a new IsolationForest | `201 Created` |
| `WS` | `/ws/stream` | Stream events, get back live anomaly scores | — |
| `GET` | `/health` | Health check + whether a model is currently trained | `200 OK` |

---

## Request & Response Schemas

### `POST /train`

**Request body** — both fields optional:

```json
{
  "n_samples": 3000,
  "anomaly_rate": 0.05
}
```

**Response:**

```json
{
  "n_samples": 3000,
  "contamination": 0.05,
  "model_path": "models/isolation_forest.joblib"
}
```

---

### `WS /ws/stream`

**Send** (matches the [Event Schema](#event-schema) exactly):

```json
{
  "amount": 4483.80,
  "hour": 3,
  "merchant_category": 7,
  "distance_from_home_km": 1842.5,
  "velocity_last_hour": 12
}
```

**Receive:**

```json
{
  "event": { "...": "the event you sent, echoed back" },
  "anomaly_score": 0.0536,
  "is_anomaly": true,
  "latency_ms": 23.06
}
```

`anomaly_score` — higher means more anomalous. Invalid input or scoring before any model is trained both return `{"error": "..."}` over the same connection rather than closing it, so a client can correct course and keep streaming.

---

### `GET /health`

```json
{
  "status": "healthy",
  "model_trained": true
}
```

---

## Example Usage

### Train, then stream a live demo

```bash
curl -X POST http://localhost:8000/train -H "Content-Type: application/json" -d '{}'
python3 scripts/simulate_stream.py
```

### Customize the demo stream

```bash
python3 scripts/simulate_stream.py --count 50 --rate 5 --anomaly-rate 0.15
```

### Point the demo client at a different host

```bash
python3 scripts/simulate_stream.py --url ws://your-host:8000/ws/stream
```

---

## Docker

No compose file — this project has no external service to orchestrate, just one self-contained image.

### Build and run

```bash
docker build -t real-time-ml-api .
docker run -p 8000:8000 real-time-ml-api
```

### Persisting the trained model across container recreation

Without a volume, the trained model lives inside the container's writable layer — it survives a `docker restart`, but not a `docker rm` + recreate. To persist it:

```bash
docker run -d -p 8000:8000 -v real-time-ml-models:/app/models --name real-time-ml-api real-time-ml-api
```

### A note on CI

Unlike other projects in this portfolio that depend on host-based infrastructure (e.g. a locally-running Ollama instance) CI can't fully verify, this project has **zero external dependencies** — so its CI pipeline can and does prove the entire pipeline end-to-end: it builds the image, boots it, trains the model over HTTP, then connects over a genuine WebSocket and asserts correct classification of both a normal and an anomalous event. Nothing in this project's CI is a partial or best-effort check.

---

## License

MIT
