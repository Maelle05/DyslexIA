"""FastAPI service exposing the dyslexia-prediction model."""
from typing import Annotated
from io import BytesIO
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException
import shap
import pandas as pd
from dyslexia_api.db.database import (
    fetch_all_records,
    fetch_records,
    insert_record,
)
from dyslexia_api.model.xgboost import XGBoostModel
from dyslexia_api.processing.text_generation import generate_passage
from dyslexia_api.processing.process_v2 import process_data
from dyslexia_api.processing.text_question import get_passage
from fastapi.middleware.cors import CORSMiddleware

OPTIMAL_THRESHOLD = 0.5999999999999998
MODEL_PATH = "dyslexia_api/model/xgboost_dyslexia_model_v3.json"

app = FastAPI()

origins = [
    "https://dyslexia-eyes-detection.netlify.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get('/')
def index():
    """Return a welcome message.

    Returns:
        JSON with an ``welcome`` key.
    """
    return {"welcome": "Welcome to DyslexIA API Server"}


@app.get('/healthcheck')
def health_status():
    """Liveness probe for container orchestrators.

    Returns:
        JSON ``{"ok": "True"}`` when the service is up.
    """
    return {"ok": "True"}


@app.post('/predict')
def predict(
    csv_file: Annotated[bytes, File()],
    user_id: Annotated[str, Form()] = "anonymous",
):
    """Run the dyslexia model on an uploaded gaze log and store the result.

    The endpoint accepts raw gaze-log CSV bytes (produced by the app),
    parses them with NumPy, extracts the model features, and returns the
    prediction. The raw gaze data and the prediction result are persisted
    in the SQLite database so they can later be retrieved through the
    ``/results/{user_id}`` endpoint.

    Args:
        csv_file: Raw bytes of a UTF-8 encoded CSV file with a header row.
        user_id: Optional identifier of the user the gaze log belongs to.
            Defaults to ``"anonymous"``.

    Returns:
        JSON with ``Prediction`` and ``Probability`` keys.
    """
    try:
        array_data = np.genfromtxt(
            BytesIO(csv_file),
            delimiter=',',
            names=True,
            dtype=None,
            encoding='utf-8'
        )
        df = pd.DataFrame(array_data)
        X = process_data(df)

        model = XGBoostModel.from_file(MODEL_PATH)

        prediction_proba = model.predict_proba(X.reshape(1, -1))
        prediction = int(prediction_proba[0][-1] >= OPTIMAL_THRESHOLD)
        label = "Dyslexique" if prediction == 1 else "Non-dyslexique"
        probability = prediction_proba[0][-1].item()

        insert_record(
            user_id=user_id,
            raw_gaze_data=csv_file.decode('utf-8'),
            prediction=label,
            probability=probability,
        )

        return {
            "Prediction": label,
            "Probability": probability
        }

    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=400, detail=str(exc))


@app.get('/results')
def get_all_results():
    """Return all stored gaze uploads and predictions, for every user.

    Records are read from the SQLite database where the ``/predict``
    endpoint stores every processed upload, ordered from the most recent
    to the oldest.

    Returns:
        JSON with a ``results`` list, where each entry contains ``id``,
        ``user_id``, ``raw_gaze_data``, ``prediction``, ``probability``
        and ``created_at``. The list is empty when no record is stored.
    """
    return {
        "results": fetch_all_records()
    }


@app.get('/results/{user_id}')
def get_results(user_id: str):
    """Return the stored gaze uploads and predictions for a user.

    Records are read from the SQLite database where the ``/predict``
    endpoint stores every processed upload, ordered from the most recent
    to the oldest.

    Args:
        user_id: Identifier of the user whose results are requested.

    Returns:
        JSON with the ``user_id`` and a ``results`` list, where each entry
        contains ``id``, ``user_id``, ``raw_gaze_data``, ``prediction``,
        ``probability`` and ``created_at``.

    Raises:
        HTTPException: 404 when no record exists for the user.
    """
    records = fetch_records(user_id)
    if not records:
        raise HTTPException(
            status_code=404,
            detail=f"No results found for user '{user_id}'",
        )
    return {
        "user_id": user_id,
        "results": records
    }


@app.get('/passage')
def generate_text():
    """Return a generated reading passage.

    Uses the passage generator which may call an external API or select a
    fallback passage when unavailable. The endpoint returns a JSON object
    with a single ``text`` key containing the passage.
    """
    text = generate_passage()
    return {
        "text": text
    }


@app.get('/text-question')
def get_text_question():
    """Return a passage together with a short comprehension question.

    The helper :func:`dyslexia_api.processing.text_question.get_passage` is
    used to select a passage and an associated multiple-choice question.
    The endpoint returns a dictionary with ``text`` and ``query`` keys.
    """
    result = get_passage()
    return {
        "text": result['text'],
        "query": result['query']
    }


@app.post('/explain')
def explain_result(csv_file: Annotated[bytes, File()]):
    """Produce a model explanation for a single uploaded gaze log CSV.

    The endpoint computes the feature vector for the provided CSV and asks
    the model's explainer to generate SHAP-like explanation values. The
    returned JSON includes arrays for ``values``, ``base_values`` and the
    original ``data`` used in the explanation.
    """
    array_data = np.genfromtxt(
            BytesIO(csv_file),
            delimiter=',',
            names=True,
            dtype=None,
            encoding='utf-8'
        )
    X = process_data(pd.DataFrame(array_data))

    model = XGBoostModel.from_file(MODEL_PATH)

    explainer = model.explainer()
    explanation = explainer(X.reshape(1, -1))

    print(shap.plots.beeswarm(explanation))

    return {
        "values": list([float(x) for x in explanation.values[0]]),
        "base_values": list([float(x) for x in explanation.base_values]),
        "data": list([float(x) for x in explanation.data[0]])
    }
