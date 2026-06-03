install:
    @pip install .

install-dev:
		@pip install -e . -r requirements.txt

test:
	@python -m pytest\
		tests/*.py
