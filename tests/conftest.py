"""Shared pytest fixtures for the test suite."""
import pytest

from dyslexia_api.db import database


class _FakeQueryJob:
    """Minimal stand-in for a BigQuery query job result."""

    def __init__(self, rows):
        self._rows = rows

    def result(self):
        return list(self._rows)


class FakeBigQueryClient:
    """In-memory fake of the subset of the BigQuery client we use.

    Rows are stored in a list. ``query`` ignores the SQL text and instead
    inspects the job config's query parameters to filter by ``user_id``,
    returning rows ordered most-recent-first using a monotonic insertion
    sequence as a deterministic tie-breaker on ``created_at``.
    """

    project = "test-project"

    def __init__(self):
        self.rows = []
        self._seq = 0

    def create_dataset(self, dataset, exists_ok=False):
        return dataset

    def create_table(self, table, exists_ok=False):
        return table

    def insert_rows_json(self, table, rows):
        for row in rows:
            self._seq += 1
            stored = dict(row)
            stored["_seq"] = self._seq
            self.rows.append(stored)
        return []

    def query(self, query, job_config=None):
        rows = list(self.rows)
        if job_config is not None:
            for param in getattr(job_config, "query_parameters", None) or []:
                if param.name == "user_id":
                    rows = [r for r in rows if r["user_id"] == param.value]

        rows.sort(key=lambda r: (r["created_at"], r["_seq"]), reverse=True)
        return _FakeQueryJob(
            [{k: v for k, v in r.items() if k != "_seq"} for r in rows]
        )


@pytest.fixture(autouse=True)
def fake_bigquery(monkeypatch):
    """Redirect the db module to an in-memory fake BigQuery client.

    This keeps tests independent and avoids any network/credentials
    requirement. The fake is returned so tests can inspect stored rows.
    """
    client = FakeBigQueryClient()
    monkeypatch.setattr(database, "get_client", lambda: client)
    database._ensured_tables.clear()
    return client
