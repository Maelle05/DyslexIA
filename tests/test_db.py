"""Tests for the BigQuery persistence layer."""
import uuid

from dyslexia_api.db.database import (
    DEFAULT_DATASET,
    DEFAULT_TABLE,
    fetch_all_records,
    fetch_records,
    get_dataset_id,
    get_table_id,
    insert_record,
)

RAW_CSV = "time,x_left,y_left,x_right,y_right\n0,0,0,0,0\n"


def test_get_dataset_id_uses_environment_variable(monkeypatch):
    monkeypatch.setenv("BQ_DATASET", "custom_dataset")
    assert get_dataset_id() == "custom_dataset"


def test_get_dataset_id_default(monkeypatch):
    monkeypatch.delenv("BQ_DATASET", raising=False)
    assert get_dataset_id() == DEFAULT_DATASET


def test_get_table_id_uses_environment_variable(monkeypatch):
    monkeypatch.setenv("BQ_TABLE", "custom_table")
    assert get_table_id() == "custom_table"


def test_get_table_id_default(monkeypatch):
    monkeypatch.delenv("BQ_TABLE", raising=False)
    assert get_table_id() == DEFAULT_TABLE


def test_insert_record_returns_uuid():
    record_id = insert_record(
        user_id="user-1",
        raw_gaze_data=RAW_CSV,
        prediction="Dyslexique",
        probability=0.9,
    )

    # The returned id is a valid UUID string.
    assert str(uuid.UUID(record_id)) == record_id


def test_fetch_records_returns_inserted_data():
    record_id = insert_record(
        user_id="user-1",
        raw_gaze_data=RAW_CSV,
        prediction="Dyslexique",
        probability=0.9,
    )

    records = fetch_records("user-1")

    assert len(records) == 1
    record = records[0]
    assert record["id"] == record_id
    assert record["user_id"] == "user-1"
    assert record["raw_gaze_data"] == RAW_CSV
    assert record["prediction"] == "Dyslexique"
    assert record["probability"] == 0.9
    assert record["created_at"] is not None


def test_fetch_records_unknown_user_returns_empty_list():
    assert fetch_records("nobody") == []


def test_fetch_all_records_empty_table_returns_empty_list():
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
