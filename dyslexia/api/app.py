from typing import Annotated
from io import BytesIO
import numpy as np
from fastapi import FastAPI, File
import pandas as pd

app = FastAPI()

@app.get('/')
def index():
    return {"app_link": "placeholder for app link"}

@app.get('/healthcheck')
def health_status():
    return {"ok": "True"}

@app.post('/predict')
def predict(csv_file: Annotated[bytes, File()]):
    array_data = np.genfromtxt(
    BytesIO(csv_file), delimiter=',', names=True, dtype=None, encoding='utf-8')
    df = pd.DataFrame(array_data)
    return {"data": df.to_dict()}
