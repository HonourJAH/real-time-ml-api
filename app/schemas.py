from pydantic import BaseModel, Field


class TrainRequest(BaseModel):
    n_samples: int = Field(default=3000, gt=0)
    anomaly_rate: float = Field(default=0.05, gt=0, lt=1)


class TrainResponse(BaseModel):
    n_samples: int
    contamination: float
    model_path: str


class EventInput(BaseModel):
    """Shape every WebSocket message must match — same fields, same
    order of importance as anomaly_detector.FEATURE_NAMES.
    """

    amount: float = Field(ge=0)
    hour: int = Field(ge=0, le=23)
    merchant_category: int = Field(ge=0)
    distance_from_home_km: float = Field(ge=0)
    velocity_last_hour: int = Field(ge=0)
