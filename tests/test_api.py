from fastapi.testclient import TestClient
import numpy as np

from dyslexia_api.api.app import app


class DummyModel:
    def predict_proba(self, X):
        return np.array([[0.1, 0.9]])


def test_index_endpoint():
    client = TestClient(app)
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"welcome": "Welcome to DyslexIA API Server"}


def test_healthcheck_endpoint():
    client = TestClient(app)
    response = client.get("/healthcheck")

    assert response.status_code == 200
    assert response.json() == {"ok": "True"}


def test_predict_returns_dyslexia_decision(monkeypatch):
    monkeypatch.setattr(
        "dyslexia_api.api.app.XGBoostModel.from_file",
        classmethod(lambda cls, path: DummyModel()),
    )
    client = TestClient(app)
    csv_bytes = (
        "time,x_left,y_left,x_right,y_right\n"
        "0,0,0,0,0\n"
        "1,1,1,1,1\n"
        "2,2,2,2,2\n"
        "3,3,3,3,3\n"
    ).encode("utf-8")

    response = client.post(
        "/predict",
        files={"csv_file": ("data.csv", csv_bytes, "text/csv")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["Prediction"] == "Dyslexique"
    assert isinstance(payload["Probability"], float)


def test_generate_text_fallback(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(
        "dyslexia_api.processing.text_generation.random.choice",
        lambda choices: "fallback passage",
    )

    client = TestClient(app)
    response = client.get("/passage")

    assert response.status_code == 200
    assert response.json() == {"text": "fallback passage"}


def test_predict_non_dyslexique(monkeypatch):
    class LowProbaModel:
        def predict_proba(self, X):
            return np.array([[0.8, 0.2]])

    monkeypatch.setattr(
        "dyslexia_api.api.app.XGBoostModel.from_file",
        classmethod(lambda cls, path: LowProbaModel()),
    )
    client = TestClient(app)
    csv_bytes = (
        "time,x_left,y_left,x_right,y_right\n"
        "0,0,0,0,0\n"
        "1,1,1,1,1\n"
    ).encode("utf-8")

    response = client.post(
        "/predict",
        files={"csv_file": ("data.csv", csv_bytes, "text/csv")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["Prediction"] == "Non-dyslexique"


def test_predict_malformed_csv_returns_400(monkeypatch):
    # No need to patch model; parsing/processing should fail before model load
    client = TestClient(app)
    bad_csv = ("foo,bar\n1,2\n").encode("utf-8")

    response = client.post(
        "/predict",
        files={"csv_file": ("bad.csv", bad_csv, "text/csv")},
    )

    assert response.status_code == 400
    assert "detail" in response.json()
