install:
	@pip install .

install-dev:
	@pip install -e . -r requirements.txt

test:
	@python -m pytest \
	tests/*

run_api:
	@uvicorn dyslexia.api.app:app

run_app:
	@streamlit run dyslexia-app/app.py
