PYTHON ?= python3
VENV ?= .venv

.PHONY: venv install build test clean

venv:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install -U pip
	$(VENV)/bin/pip install -r requirements.txt

build:
	$(PYTHON) -m pipeline.build_reporting_datasets

test:
	pytest

clean:
	rm -rf reports
