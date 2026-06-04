import csv
import io
import pytest
from fastapi.testclient import TestClient

from dyslexia.api.app import app

client = TestClient(app)


def _make_csv_bytes(rows):
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue().encode("utf-8")


class TestIndex:
    def test_status_200(self):
        resp = client.get("/")
        assert resp.status_code == 200

    def test_returns_app_link_key(self):
        resp = client.get("/")
        assert "app_link" in resp.json()


class TestHealthcheck:
    def test_status_200(self):
        resp = client.get("/healthcheck")
        assert resp.status_code == 200

    def test_ok_is_true(self):
        resp = client.get("/healthcheck")
        assert resp.json()["ok"] == "True"


class TestPredict:
    def test_status_200_with_valid_csv(self):
        data = _make_csv_bytes([{"col1": 1.0, "col2": 2.0}, {"col1": 3.0, "col2": 4.0}])
        resp = client.post("/predict", files={"csv_file": ("data.csv", data, "text/csv")})
        assert resp.status_code == 200

    def test_response_contains_data_key(self):
        # Need ≥2 rows: np.genfromtxt returns a 0-d array for a single data row,
        # which pd.DataFrame cannot wrap.
        data = _make_csv_bytes([{"a": 1, "b": 2}, {"a": 3, "b": 4}])
        resp = client.post("/predict", files={"csv_file": ("data.csv", data, "text/csv")})
        assert "data" in resp.json()

    def test_columns_preserved_in_response(self):
        rows = [{"fix_x": 100, "fix_y": 200, "time": 1.0},
                {"fix_x": 150, "fix_y": 250, "time": 2.0}]
        data = _make_csv_bytes(rows)
        resp = client.post("/predict", files={"csv_file": ("data.csv", data, "text/csv")})
        body = resp.json()["data"]
        assert "fix_x" in body and "fix_y" in body and "time" in body

    def test_row_count_matches_csv(self):
        rows = [{"x": i, "y": i * 2} for i in range(5)]
        data = _make_csv_bytes(rows)
        resp = client.post("/predict", files={"csv_file": ("data.csv", data, "text/csv")})
        body = resp.json()["data"]
        first_col = next(iter(body.values()))
        assert len(first_col) == 5

    def test_missing_file_returns_422(self):
        resp = client.post("/predict")
        assert resp.status_code == 422
