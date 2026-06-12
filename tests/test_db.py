"""Tests for the SQLite persistence layer."""
from dyslexia_api.db.database import (
    DEFAULT_DB_PATH,
    fetch_all_records,
    fetch_records,
    get_db_path,
    insert_record,
)

RAW_CSV = "time,x_left,y_left,x_right,y_right\n0,0,0,0,0\n"


def test_get_db_path_uses_environment_variable(isolated_database):
    assert get_db_path() == isolated_database


def test_get_db_path_default(monkeypatch):
    monkeypatch.delenv("DYSLEXIA_DB_PATH", raising=False)
    assert get_db_path() == DEFAULT_DB_PATH


def test_insert_record_returns_row_id():
    record_id = insert_record(
        user_id="user-1",
        raw_gaze_data=RAW_CSV,
        prediction="Dyslexique",
        probability=0.9,
    )

    assert record_id == 1


def test_fetch_records_returns_inserted_data():
    insert_record(
        user_id="user-1",
        raw_gaze_data=RAW_CSV,
        prediction="Dyslexique",
        probability=0.9,
    )

    records = fetch_records("user-1")

    assert len(records) == 1
    record = records[0]
    assert record["user_id"] == "user-1"
    assert record["raw_gaze_data"] == RAW_CSV
    assert record["prediction"] == "Dyslexique"
    assert record["probability"] == 0.9
    assert record["created_at"] is not None


def test_fetch_records_unknown_user_returns_empty_list():
    assert fetch_records("nobody") == []


def test_fetch_all_records_empty_database_returns_empty_list():
    assert fetch_all_records() == []


def test_fetch_all_records_returns_every_user_recent_first():
    first_id = insert_record("user-1", RAW_CSV, "Non-dyslexique", 0.2)
    second_id = insert_record("user-2", RAW_CSV, "Dyslexique", 0.9)

    records = fetch_all_records()

    assert [record["id"] for record in records] == [second_id, first_id]
    assert {record["user_id"] for record in records} == {"user-1", "user-2"}


def test_fetch_records_filters_by_user_and_orders_recent_first():
    first_id = insert_record("user-1", RAW_CSV, "Non-dyslexique", 0.2)
    second_id = insert_record("user-1", RAW_CSV, "Dyslexique", 0.9)
    insert_record("user-2", RAW_CSV, "Dyslexique", 0.8)

    records = fetch_records("user-1")

    assert [record["id"] for record in records] == [second_id, first_id]
    assert all(record["user_id"] == "user-1" for record in records)
