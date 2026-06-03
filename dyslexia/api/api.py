from fastapi import FastAPI

app = FastAPI()

@app.get('/')
def index():
    return {"app_link": "placeholder for app link"}

@app.get('/healthcheck')
def health_status():
    return {"ok": "True"}

@app.post('/predict')
def predict(csv_file: str):
    pass #TODO: implement predict function
