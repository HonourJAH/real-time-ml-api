import time

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, status
from pydantic import ValidationError

from app.schemas import TrainRequest, TrainResponse, EventInput
from app.services.event_generator import generate_batch
from app.services.anomaly_detector import (
    train as train_model,
    score as score_event,
    is_trained,
    ModelNotTrainedError,
)

app = FastAPI(
    title="Real-Time ML API",
    description="Real-time anomaly detection on a stream of transaction events via WebSocket",
)


@app.post("/train", response_model=TrainResponse, status_code=status.HTTP_201_CREATED)
async def train_route(request: TrainRequest):
    batch = generate_batch(request.n_samples, anomaly_rate=request.anomaly_rate)
    events_only = [event for event, _ in batch]

    result = train_model(events_only, contamination=request.anomaly_rate)
    return TrainResponse(**result)


@app.websocket("/ws/stream")
async def stream_events(websocket: WebSocket):
    """Accepts a persistent connection; the client sends one event (JSON)
    per message, matching EventInput's shape, and receives back a scored
    result — including per-event latency — over the same connection.

    Errors (bad input shape, or scoring before any model is trained)
    are sent back as a JSON error message rather than closing the
    connection, so a client can correct course and keep streaming.
    """
    await websocket.accept()
    try:
        while True:
            raw_event = await websocket.receive_json()

            try:
                event = EventInput(**raw_event)
            except ValidationError as exc:
                await websocket.send_json({"error": f"Invalid event: {exc.errors()}"})
                continue

            start = time.perf_counter()
            try:
                result = score_event(event.model_dump())
            except ModelNotTrainedError as exc:
                await websocket.send_json({"error": str(exc)})
                continue
            latency_ms = round((time.perf_counter() - start) * 1000, 3)

            await websocket.send_json(
                {
                    "event": event.model_dump(),
                    "anomaly_score": result["anomaly_score"],
                    "is_anomaly": result["is_anomaly"],
                    "latency_ms": latency_ms,
                }
            )
    except WebSocketDisconnect:
        pass


@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_trained": is_trained()}
