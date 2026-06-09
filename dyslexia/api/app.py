"""FastAPI service exposing the dyslexia-prediction model."""
from typing import Annotated
from io import BytesIO
import numpy as np
from fastapi import FastAPI, File
import pandas as pd
from dyslexia.model.xgboost import XGBoostModel
from dyslexia.processing.text_generation import generate_passage
from dyslexia.processing.process_v2 import process_data
from fastapi.middleware.cors import CORSMiddleware

OPTIMAL_THRESHOLD = 0.35

app = FastAPI()

origins = [
    "http://0.0.0.0:8000",
    "http://localhost",
    "http://localhost:8000",
    "http://localhost:5173",
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
    """Return a placeholder link to the front-end application.

    Returns:
        JSON with an ``app_link`` key.
    """
    return {"app_link": "placeholder for app link"}


@app.get('/healthcheck')
def health_status():
    """Liveness probe for container orchestrators.

    Returns:
        JSON ``{"ok": "True"}`` when the service is up.
    """
    return {"ok": "True"}


@app.post('/predict')
def predict(csv_file: Annotated[bytes, File()]):
    """Parse an uploaded CSV and return its contents as a JSON dictionary.

    The endpoint accepts raw gaze-log CSV bytes (produced by the app),
    parses them with NumPy, converts to a DataFrame, and returns all columns as
    a nested dict keyed by column name then row index.

    Args:
        csv_file: Raw bytes of a UTF-8 encoded CSV file with a header row.

    Returns:
        JSON with a ``data`` key containing the DataFrame as a dict-of-dicts.
    """
    array_data = np.genfromtxt(
            BytesIO(csv_file),
            delimiter=',',
            names=True,
            dtype=None,
            encoding='utf-8'
        )
    print(pd.DataFrame(array_data).shape)
    X = process_data(pd.DataFrame(array_data))
    print(X.reshape(1, -1).shape)

    model = XGBoostModel.from_file("dyslexia/model/xgboost_dyslexia_model_v2.json")

    prediction_proba = model.predict_proba(X.reshape(1, -1))
    prediction = int(prediction_proba[0][-1] >= OPTIMAL_THRESHOLD)

    return {
        "Prediction": "Dyslexique" if prediction else "Non-dyslexique",
        "Probability": prediction_proba[0][prediction].item()
        }

@app.get('/passage')
def generate_text():
    text = generate_passage()
    return {
        "text": text
    }
