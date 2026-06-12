"""SQLite storage for raw gaze data and model prediction results.

The database holds one table, ``gaze_records``, where each row stores the
raw gaze CSV uploaded by a user together with the prediction returned by
the model. The database file location can be overridden with the
``DYSLEXIA_DB_PATH`` environment variable, which is convenient for tests
and deployments.
"""
import os
import sqlite3

DEFAULT_DB_PATH = "dyslexia.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS gaze_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    raw_gaze_data TEXT NOT NULL,
    prediction TEXT NOT NULL,
    probability REAL NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
)
"""


def get_db_path():
    """Return the path of the SQLite database file.

    Returns:
        The value of the ``DYSLEXIA_DB_PATH`` environment variable if set,
        otherwise :data:`DEFAULT_DB_PATH`.
    """
    return os.environ.get("DYSLEXIA_DB_PATH", DEFAULT_DB_PATH)


def get_connection(db_path=None):
    """Open a SQLite connection and make sure the schema exists.

    Args:
        db_path: Optional path to the database file. Defaults to the path
            returned by :func:`get_db_path`.

    Returns:
        An open :class:`sqlite3.Connection` with ``sqlite3.Row`` as row
        factory, so rows can be accessed by column name.
    """
    connection = sqlite3.connect(db_path or get_db_path())
    connection.row_factory = sqlite3.Row
    connection.execute(_SCHEMA)
    return connection


def insert_record(user_id, raw_gaze_data, prediction, probability,
                  db_path=None):
    """Persist a gaze upload and its prediction result.

    Args:
        user_id: Identifier of the user the record belongs to.
        raw_gaze_data: Raw gaze log CSV content as text.
        prediction: Human-readable prediction label.
        probability: Model probability associated with the prediction.
        db_path: Optional path to the database file.

    Returns:
        The integer primary key of the newly inserted row.
    """
    connection = get_connection(db_path)
    try:
        with connection:
            cursor = connection.execute(
                "INSERT INTO gaze_records "
                "(user_id, raw_gaze_data, prediction, probability) "
                "VALUES (?, ?, ?, ?)",
                (user_id, raw_gaze_data, prediction, probability),
            )
            return cursor.lastrowid
    finally:
        connection.close()


def fetch_all_records(db_path=None):
    """Fetch all stored records for every user, most recent first.

    Args:
        db_path: Optional path to the database file.

    Returns:
        A list of dictionaries with the keys ``id``, ``user_id``,
        ``raw_gaze_data``, ``prediction``, ``probability`` and
        ``created_at``. The list is empty when the database holds no
        records.
    """
    connection = get_connection(db_path)
    try:
        rows = connection.execute(
            "SELECT id, user_id, raw_gaze_data, prediction, probability, "
            "created_at FROM gaze_records ORDER BY id DESC"
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        connection.close()


def fetch_records(user_id, db_path=None):
    """Fetch all stored records for a user, most recent first.

    Args:
        user_id: Identifier of the user whose records are requested.
        db_path: Optional path to the database file.

    Returns:
        A list of dictionaries with the keys ``id``, ``user_id``,
        ``raw_gaze_data``, ``prediction``, ``probability`` and
        ``created_at``. The list is empty when the user has no records.
    """
    connection = get_connection(db_path)
    try:
        rows = connection.execute(
            "SELECT id, user_id, raw_gaze_data, prediction, probability, "
            "created_at FROM gaze_records WHERE user_id = ? "
            "ORDER BY id DESC",
            (user_id,),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        connection.close()
