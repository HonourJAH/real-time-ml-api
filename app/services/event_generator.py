import random
import uuid
from datetime import datetime, timezone

# Feature schema every event shares, in the shape the IsolationForest
# detector expects: amount, hour of day, merchant category, distance from
# the cardholder's home, and recent transaction velocity — all standard,
# realistic signals in real fraud-detection systems.

MERCHANT_CATEGORIES = list(range(10))  # 10 generic category codes

# Hours weighted toward normal daytime shopping activity (peaks around
# midday and early evening), used only for NORMAL events.
_NORMAL_HOUR_WEIGHTS = [
    1,
    1,
    1,
    1,
    1,
    2,  # 0-5am: rare
    3,
    5,
    6,
    7,
    8,
    8,  # 6-11am: ramping up
    9,
    8,
    7,
    7,
    8,
    9,  # 12-5pm: steady/high
    8,
    6,
    4,
    3,
    2,
    1,  # 6-11pm: tapering off
]


def _new_event_shell() -> dict:
    return {
        "event_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def generate_normal_event() -> dict:
    """A realistic, unremarkable transaction — modest amount, normal
    shopping hours, close to home, low recent activity.
    """
    event = _new_event_shell()
    event.update(
        {
            "amount": round(random.lognormvariate(3.5, 0.6), 2),  # mostly $15-$150
            "hour": random.choices(range(24), weights=_NORMAL_HOUR_WEIGHTS, k=1)[0],
            "merchant_category": random.choice(MERCHANT_CATEGORIES),
            "distance_from_home_km": round(
                random.expovariate(1 / 5), 2
            ),  # mostly a few km
            "velocity_last_hour": random.choices(
                [0, 1, 2, 3], weights=[5, 4, 2, 1], k=1
            )[0],
        }
    )
    return event


def generate_anomalous_event() -> dict:
    """A transaction with the statistical fingerprints of fraud — unusually
    large amount, an odd hour, far from home, or unusually high velocity.
    Not every anomalous event has ALL of these at once (real fraud doesn't
    either) — each run picks a random subset of "suspicious" traits.
    """
    event = _new_event_shell()
    event.update(
        {
            "amount": round(random.uniform(500, 5000), 2),
            "hour": random.choice([1, 2, 3, 4]),  # unusual overnight hours
            "merchant_category": random.choice(MERCHANT_CATEGORIES),
            "distance_from_home_km": round(random.uniform(200, 3000), 2),
            "velocity_last_hour": random.randint(5, 20),
        }
    )
    return event


def generate_event(anomaly_rate: float = 0.05) -> tuple[dict, bool]:
    """Generate a single event, anomalous with probability `anomaly_rate`.

    Returns (event, is_actually_anomalous) — the boolean is ground truth
    for demo/evaluation purposes only. It is never given to the model,
    since IsolationForest is unsupervised and never sees labels.
    """
    is_anomaly = random.random() < anomaly_rate
    event = generate_anomalous_event() if is_anomaly else generate_normal_event()
    return event, is_anomaly


def generate_batch(n: int, anomaly_rate: float = 0.05) -> list[tuple[dict, bool]]:
    """Generate a batch of events — used for training data, where a small
    contamination rate is expected and desired (IsolationForest is trained
    with a `contamination` parameter matching roughly this rate).
    """
    return [generate_event(anomaly_rate) for _ in range(n)]
