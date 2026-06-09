install:
	@npm --prefix ./dyslexia_client install
	@pip install .

install-dev:
	@npm --prefix ./dyslexia_client install
	@pip install -e . -r requirements.txt

test:
	@python -m pytest \
	tests/*

run_api:
	@uvicorn dyslexia_api.api.app:app

run_client:
	@npm --prefix ./dyslexia_client run dev
