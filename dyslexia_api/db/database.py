"""BigQuery storage for raw gaze data and model prediction results.

Records are persisted to a single BigQuery table (``gaze_records`` by
default), where each row stores the raw gaze CSV uploaded by a user
together with the prediction returned by the model.

Configuration is read from environment variables:

* ``GCP_PROJECT_ID`` (or ``GOOGLE_CLOUD_PROJECT``) - the GCP project that
  owns the dataset.
* ``BQ_DATASET`` - the dataset name. Defaults to :data:`DEFAULT_DATASET`.
* ``BQ_TABLE`` - the table name. Defaults to :data:`DEFAULT_TABLE`.
* ``BQ_LOCATION`` - the dataset location used when auto-creating it.
  Defaults to :data:`DEFAULT_LOCATION`.

The dataset and table are created automatically on first use if they do
not already exist.

Note:
    Rows are written with the streaming API (``insert_rows_json``). Newly
    inserted rows can take a few seconds to become queryable because of
    BigQuery's streaming buffer, so reads are eventually consistent.
"""
import os
import uuid
from datetime import datetime, timezone

from google.cloud import bigquery

DEFAULT_DATASET = "dyslexia"
DEFAULT_TABLE = "gaze_records"
DEFAULT_LOCATION = "europe-west9"

_SCHEMA = [
    bigquery.SchemaField("id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("user_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("raw_gaze_data", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("prediction", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("probability", "FLOAT", mode="REQUIRED"),
    bigquery.SchemaField("created_at", "TIMESTAMP", mode="REQUIRED"),
]

_SELECT_COLUMNS = (
    "id, user_id, raw_gaze_data, prediction, probability, created_at"
)

# Module-level singletons so we reuse a single client and only attempt to
# create the dataset/table once per process.
_client = None
_ensured_tables = set()


def get_project_id():
    """Return the GCP project id from the environment.

    Returns:
        The value of ``GCP_PROJECT_ID`` or, as a fallback,
        ``GOOGLE_CLOUD_PROJECT``. ``None`` when neither is set, in which
        case the BigQuery client falls back to its own default resolution
        (e.g. application default credentials).
    """
    return os.environ.get("GCP_PROJECT_ID") or os.environ.get(
        "GOOGLE_CLOUD_PROJECT"
    )


def get_dataset_id():
    """Return the BigQuery dataset name.

    Returns:
        The value of the ``BQ_DATASET`` environment variable if set,
        otherwise :data:`DEFAULT_DATASET`.
    """
    return os.environ.get("BQ_DATASET", DEFAULT_DATASET)


def get_table_id():
    """Return the BigQuery table name.

    Returns:
        The value of the ``BQ_TABLE`` environment variable if set,
        otherwise :data:`DEFAULT_TABLE`.
    """
    return os.environ.get("BQ_TABLE", DEFAULT_TABLE)


def get_client():
    """Return a cached BigQuery client.

    The client is created lazily on first use and reused afterwards. Tests
    monkeypatch this function to inject an in-memory fake.

    Returns:
        A :class:`google.cloud.bigquery.Client` instance.
    """
    global _client
    if _client is None:
        _client = bigquery.Client(project=get_project_id())
    return _client


def _table_ref(client):
    """Return the fully-qualified ``project.dataset.table`` identifier."""
    return f"{client.project}.{get_dataset_id()}.{get_table_id()}"


def _ensure_table(client):
    """Create the dataset and table if they do not already exist.

    The result is cached per table reference so the (network) existence
    checks only happen once per process.

    Args:
        client: The BigQuery client to use.
    """
    table_ref = _table_ref(client)
    if table_ref in _ensured_tables:
        return

    dataset = bigquery.Dataset(f"{client.project}.{get_dataset_id()}")
    dataset.location = os.environ.get("BQ_LOCATION", DEFAULT_LOCATION)
    client.create_dataset(dataset, exists_ok=True)

    table = bigquery.Table(table_ref, schema=_SCHEMA)
    client.create_table(table, exists_ok=True)

    _ensured_tables.add(table_ref)


def _row_to_dict(row):
    """Convert a BigQuery result row into a plain JSON-serialisable dict."""
    record = dict(row)
    created_at = record.get("created_at")
    if hasattr(created_at, "isoformat"):
        record["created_at"] = created_at.isoformat()
    return record


def insert_record(user_id, raw_gaze_data, prediction, probability):
    """Persist a gaze upload and its prediction result.

    Args:
        user_id: Identifier of the user the record belongs to.
        raw_gaze_data: Raw gaze log CSV content as text.
        prediction: Human-readable prediction label.
        probability: Model probability associated with the prediction.

    Returns:
        The generated UUID string identifying the newly inserted row.

    Raises:
        RuntimeError: If BigQuery reports row-level insertion errors.
    """
    client = get_client()
    _ensure_table(client)

    record_id = str(uuid.uuid4())
    row = {
        "id": record_id,
        "user_id": user_id,
        "raw_gaze_data": raw_gaze_data,
        "prediction": prediction,
        "probability": probability,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    errors = client.insert_rows_json(_table_ref(client), [row])
    if errors:
        raise RuntimeError(
            f"Failed to insert record into BigQuery: {errors}"
        )
    return record_id


def fetch_all_records():
    """Fetch all stored records for every user, most recent first.

    Returns:
        A list of dictionaries with the keys ``id``, ``user_id``,
        ``raw_gaze_data``, ``prediction``, ``probability`` and
        ``created_at``. The list is empty when the table holds no records.
    """
    client = get_client()
    _ensure_table(client)

    query = (
        f"SELECT {_SELECT_COLUMNS} FROM `{_table_ref(client)}` "
        "ORDER BY created_at DESC"
    )
    rows = client.query(query).result()
    return [_row_to_dict(row) for row in rows]


def fetch_records(user_id):
    """Fetch all stored records for a user, most recent first.

    Args:
        user_id: Identifier of the user whose records are requested.

    Returns:
        A list of dictionaries with the keys ``id``, ``user_id``,
        ``raw_gaze_data``, ``prediction``, ``probability`` and
        ``created_at``. The list is empty when the user has no records.
    """
    client = get_client()
    _ensure_table(client)

    query = (
        f"SELECT {_SELECT_COLUMNS} FROM `{_table_ref(client)}` "
        "WHERE user_id = @user_id ORDER BY created_at DESC"
    )
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("user_id", "STRING", user_id),
        ]
    )
    rows = client.query(query, job_config=job_config).result()
    return [_row_to_dict(row) for row in rows]
