# Builder stage
FROM python:3.12-slim AS builder

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --user --timeout 100 --retries 5 -r requirements.txt


# Runtime stage
FROM python:3.12-slim

RUN groupadd -g 1001 appuser \
    && useradd -u 1001 -g appuser -m -s /usr/sbin/nologin appuser

WORKDIR /app

COPY --from=builder --chown=appuser:appuser /root/.local /home/appuser/.local
COPY --chown=appuser:appuser app ./app

# /train persists the trained model here at runtime (isolation_forest.joblib)
# — needs to exist and be writable by the non-root user before the app runs.
RUN mkdir -p /app/models && chown -R appuser:appuser /app/models

ENV PATH=/home/appuser/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
