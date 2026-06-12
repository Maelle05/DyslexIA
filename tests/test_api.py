from fastapi.testclient import TestClient
import numpy as np

from dyslexia_api.api.app import app
from dyslexia_api.db.database import fetch_records, insert_record


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


def test_predict_stores_result_in_database(monkeypatch):
    monkeypatch.setattr(
        "dyslexia_api.api.app.XGBoostModel.from_file",
        classmethod(lambda cls, path: DummyModel()),
    )
    client = TestClient(app)
    csv_bytes = (
        "time,x_left,y_left,x_right,y_right\n"
        "0,0,0,0,0\n"
        "1,1,1,1,1\n"
    ).encode("utf-8")

    response = client.post(
        "/predict",
        data={"user_id": "user-42"},
        files={"csv_file": ("data.csv", csv_bytes, "text/csv")},
    )

    assert response.status_code == 200
    records = fetch_records("user-42")
    assert len(records) == 1
    assert records[0]["prediction"] == "Dyslexique"
    assert records[0]["raw_gaze_data"] == csv_bytes.decode("utf-8")


def test_get_results_returns_stored_predictions():
    insert_record(
        user_id="user-7",
        raw_gaze_data="time,x_left,y_left,x_right,y_right\n0,0,0,0,0\n",
        prediction="Non-dyslexique",
        probability=0.3,
    )
    client = TestClient(app)

    response = client.get("/results/user-7")

    assert response.status_code == 200
    payload = response.json()
    assert payload["user_id"] == "user-7"
    assert len(payload["results"]) == 1
    assert payload["results"][0]["prediction"] == "Non-dyslexique"
    assert payload["results"][0]["probability"] == 0.3


def test_get_all_results_returns_records_for_every_user():
    insert_record(
        user_id="user-1",
        raw_gaze_data="time,x_left,y_left,x_right,y_right\n0,0,0,0,0\n",
        prediction="Non-dyslexique",
        probability=0.3,
    )
    insert_record(
        user_id="user-2",
        raw_gaze_data="time,x_left,y_left,x_right,y_right\n0,0,0,0,0\n",
        prediction="Dyslexique",
        probability=0.9,
    )
    client = TestClient(app)

    response = client.get("/results")

    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) == 2
    assert [r["user_id"] for r in results] == ["user-2", "user-1"]


def test_get_all_results_empty_database_returns_empty_list():
    client = TestClient(app)

    response = client.get("/results")

    assert response.status_code == 200
    assert response.json() == {"results": []}


def test_get_results_unknown_user_returns_404():
    client = TestClient(app)

    response = client.get("/results/unknown-user")

    assert response.status_code == 404
    assert "unknown-user" in response.json()["detail"]


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
