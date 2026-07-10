"""
Shared fixtures for the Real-Time ML API test suite.

anomaly_detector.py keeps a module-level in-memory model cache and reads
from a fixed MODEL_PATH on disk — necessary for real production use (the
whole point is not re-loading the model from disk on every single score()
call), but it means tests MUST reset that state between runs, or an
earlier test's trained model would leak into a later test that expects
no model to exist yet.
"""

import pytest

import app.services.anomaly_detector as detector_module


@pytest.fixture(autouse=True)
def isolated_model_state(tmp_path, monkeypatch):
    """Applied to every test automatically. Points MODEL_PATH at a fresh
    per-test temp file and clears the in-memory model cache before AND
    after each test, so tests never see another test's trained model
    and never leave one behind for the next test to accidentally rely on.
    """
    model_path = str(tmp_path / "isolation_forest.joblib")
    monkeypatch.setattr(detector_module, "MODEL_PATH", model_path)
    detector_module._model = None
    yield
    detector_module._model = None
