### Object Counter development commands.
### Requires Python 3.11+, uv, and Docker for infrastructure targets.

SHELL := /bin/bash
VENV ?= .venv
PY := $(VENV)/bin/python
UV ?= uv
COMPOSE ?= docker compose

# Local Postgres for `make db-up` / integration tests.
DB_USER ?= counter
DB_PASSWORD ?= counter
DB_NAME ?= object_counter
DB_PORT ?= 5432
DATABASE_URL ?= postgresql+psycopg://$(DB_USER):$(DB_PASSWORD)@localhost:$(DB_PORT)/$(DB_NAME)
TEST_DATABASE_URL ?= postgresql+psycopg://$(DB_USER):$(DB_PASSWORD)@localhost:$(DB_PORT)/$(DB_NAME)_test

# The public RFCN model the original service used, for the TF Serving profile.
MODEL_URL ?= https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_8/rfcn_resnet101_fp32_coco_pretrained_model.tar.gz
MODEL_DIR ?= tmp/model

.DEFAULT_GOAL := help
.PHONY: help setup install run run-prod cli test test-unit test-integration test-e2e \
	    coverage lint format typecheck verify db-up db-down db-shell migrate migrate-down \
	    models-download tfs-up tfs-down docker-build docker-up docker-down docker-logs clean

help: ## Show this help
	@grep -hE '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) \
	    | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}'

## --- setup ---------------------------------------------------------------

$(VENV)/bin/activate: pyproject.toml uv.lock
	$(UV) sync --frozen --extra dev

install: $(VENV)/bin/activate ## Create the virtualenv and install the project

setup: install db-up migrate ## One-shot bootstrap: virtualenv, database, schema
	@echo "Ready. Use 'make run', or 'make tfs-up && make run-prod'."

## --- running -------------------------------------------------------------

run: install ## Run the API with fakes (no model server, no database needed)
	COUNTER_ENV=dev $(PY) -m counter.entrypoints.api.app

run-prod: install ## Run the API against Postgres and TensorFlow Serving
	COUNTER_ENV=prod COUNTER_PERSISTENCE=sql COUNTER_DATABASE_URL=$(DATABASE_URL) \
		$(PY) -m counter.entrypoints.api.app

cli: install ## Detect on one image: make cli IMAGE=resources/images/cat.jpg THRESHOLD=0.9
	$(PY) -m counter.entrypoints.cli detect $(or $(IMAGE),resources/images/cat.jpg) \
	    --threshold $(or $(THRESHOLD),0.5)

## --- quality -------------------------------------------------------------

test: install ## Run every test (SQLite for the database tests)
	$(PY) -m pytest

test-unit: install ## Fast tests only
	$(PY) -m pytest -m unit

test-integration: install ## Database and adapter tests against Postgres
	TEST_DATABASE_URL=$(TEST_DATABASE_URL) $(PY) -m pytest -m integration

test-e2e: install ## Full HTTP stack
	TEST_DATABASE_URL=$(TEST_DATABASE_URL) $(PY) -m pytest -m e2e

coverage: install ## Test suite with a coverage report
	TEST_DATABASE_URL=$(TEST_DATABASE_URL) $(PY) -m pytest --cov --cov-report=term-missing

lint: install ## Static checks
	$(VENV)/bin/ruff check .
	$(VENV)/bin/ruff format --check .

format: install ## Apply formatting and safe fixes
	$(VENV)/bin/ruff check --fix .
	$(VENV)/bin/ruff format .

typecheck: install ## Type check the package
	$(VENV)/bin/mypy

verify: lint typecheck test ## What CI runs

## --- database ------------------------------------------------------------

db-up: ## Start Postgres and create the test database
	$(COMPOSE) up -d postgres
	@until $(COMPOSE) exec -T postgres pg_isready -U $(DB_USER) >/dev/null 2>&1; do sleep 1; done
	@$(COMPOSE) exec -T postgres psql -U $(DB_USER) -d $(DB_NAME) \
	    -c "SELECT 'CREATE DATABASE $(DB_NAME)_test' WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = '$(DB_NAME)_test')\gexec" >/dev/null
	@echo "postgres ready on localhost:$(DB_PORT)"

db-down: ## Stop Postgres and delete its volume
	$(COMPOSE) down -v postgres

db-shell: ## psql session against the local database
	$(COMPOSE) exec postgres psql -U $(DB_USER) -d $(DB_NAME)

migrate: install ## Apply migrations to the local database and the test database
	COUNTER_DATABASE_URL=$(DATABASE_URL) $(VENV)/bin/alembic upgrade head
	COUNTER_DATABASE_URL=$(TEST_DATABASE_URL) $(VENV)/bin/alembic upgrade head

migrate-down: install ## Roll the local database back to an empty schema
	COUNTER_DATABASE_URL=$(DATABASE_URL) $(VENV)/bin/alembic downgrade base

## --- models --------------------------------------------------------------

models-download: ## Download the public RFCN model into $(MODEL_DIR) for TF Serving
	@mkdir -p tmp $(MODEL_DIR)/rfcn/1
	@test -f $(MODEL_DIR)/rfcn/1/saved_model.pb || ( \
	    curl -fsSL $(MODEL_URL) -o tmp/rfcn.tar.gz && \
	    tar -xzf tmp/rfcn.tar.gz -C tmp && \
	    mv tmp/rfcn_resnet101_coco_2018_01_28/saved_model/saved_model.pb $(MODEL_DIR)/rfcn/1/ && \
	    rm -rf tmp/rfcn_resnet101_coco_2018_01_28 tmp/rfcn.tar.gz )
	@echo "model ready in $(MODEL_DIR)/rfcn/1"

tfs-up: models-download ## Start TensorFlow Serving with the RFCN model
	$(COMPOSE) --profile tfserving up -d tfserving

tfs-down: ## Stop TensorFlow Serving
	$(COMPOSE) --profile tfserving stop tfserving

## --- containers ----------------------------------------------------------

docker-build: ## Build the application image
	$(COMPOSE) build api

docker-up: ## Run the whole stack in containers (api + postgres, migrations applied)
	$(COMPOSE) up -d --build api
	@echo "API on http://localhost:5000/docs"

docker-down: ## Stop the stack and delete volumes
	$(COMPOSE) down -v

docker-logs: ## Follow the application logs
	$(COMPOSE) logs -f api

clean: ## Remove the virtualenv, caches and downloaded models
	rm -rf $(VENV) .pytest_cache .ruff_cache .mypy_cache .coverage htmlcov tmp var
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
