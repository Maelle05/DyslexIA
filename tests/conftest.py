"""Shared pytest fixtures for the test suite."""
import pytest


@pytest.fixture(autouse=True)
def isolated_database(tmp_path, monkeypatch):
    """Redirect the SQLite database to a temporary file for every test.

    This keeps tests independent from each other and prevents them from
    writing to the real ``dyslexia.db`` file in the working directory.

    Returns:
        The path of the temporary database file as a string.
    """
    db_path = tmp_path / "test_dyslexia.db"
    monkeypatch.setenv("DYSLEXIA_DB_PATH", str(db_path))
    return str(db_path)
