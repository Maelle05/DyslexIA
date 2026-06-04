FROM python:3.10.6-slim

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY ./dyslexia ./dyslexia

CMD uvicorn dyslexia.api.app:app --host 0.0.0.0 --port 8000
