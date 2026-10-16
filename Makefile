# cobalt-battery-prognostics
#
# `make demo` goes from a clean checkout to every number in the README.
# `make smoke` does the same in about two minutes on a small fleet.
# `docker compose up` does `make demo` in a container.

PYTHON ?= python
CONFIG ?= config/default.yaml
CI_CONFIG ?= config/ci.yaml
IMAGE ?= cobalt-battery-prognostics:latest

.DEFAULT_GOAL := help
.PHONY: help setup data evaluate report figures stream stream-can serve smoke demo test \
        lint format check docker-build docker-up docker-test docker-down \
        clean clean-all

help:
	@echo "make setup         install the pinned dependencies into .venv"
	@echo "make data          simulate the fleet (about 75 seconds)"
	@echo "make evaluate      every model, every protocol, costs, figures, models"
	@echo "make report        rebuild the report from an existing summary.json"
	@echo "make stream        run a simulated pack through the deployed model"
	@echo "make stream-can    the same pack through a simulated BMS on a CAN bus"
	@echo "make serve         run the fleet service on http://127.0.0.1:8000"
	@echo "make demo          everything, end to end, from nothing"
	@echo "make smoke         the same pipeline small, no README rewrite"
	@echo "make test          the test suite"
	@echo "make lint          ruff"
	@echo "make check         assert the README matches the committed results"
	@echo "make docker-up     make demo in a container"
	@echo "make clean         remove generated data, results, figures and models"

setup:
	bash scripts/setup.sh

data:
	$(PYTHON) -m cobalt.cli generate --config $(CONFIG)

evaluate: data
	$(PYTHON) -m cobalt.cli evaluate --config $(CONFIG) --update-readme

report:
	$(PYTHON) -m cobalt.cli report --config $(CONFIG) --update-readme

figures: report

stream:
	$(PYTHON) -m cobalt.cli stream --config $(CONFIG)

stream-can:
	$(PYTHON) -m cobalt.cli stream --config $(CONFIG) --source can

serve:
	$(PYTHON) -m cobalt.cli serve --config $(CONFIG)

demo:
	bash scripts/run_demo.sh

smoke:
	bash scripts/run_smoke.sh

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check src tests scripts
	$(PYTHON) -m ruff format --check src tests scripts

format:
	$(PYTHON) -m ruff format src tests scripts

check:
	$(PYTHON) -m cobalt.check_readme --config $(CONFIG)

docker-build:
	docker build -f docker/Dockerfile -t $(IMAGE) .

docker-up:
	docker compose up --build

docker-test: docker-build
	docker run --rm $(IMAGE) python -m pytest -q

docker-down:
	docker compose down -v

clean:
	rm -rf data results analysis artifacts
	rm -rf .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

clean-all: clean
	rm -rf .venv *.egg-info src/*.egg-info
