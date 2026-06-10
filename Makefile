.DEFAULT_GOAL := help
PYTHON ?= python
PIP ?= $(PYTHON) -m pip
PKG := src/quantumium

.PHONY: help install test lint typecheck docs build check docker docker-test clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Install package with dev dependencies (editable)
	$(PIP) install -e ".[dev]"

test: ## Run the test suite
	$(PYTHON) -m pytest -q

lint: ## Run linter
	$(PYTHON) -m ruff check src tests

typecheck: ## Run mypy in strict mode
	$(PYTHON) -m mypy --strict $(PKG)

docs: ## Build documentation
	$(MAKE) -C docs html

build: ## Build sdist and wheel
	$(PYTHON) -m build

check: lint typecheck test ## Lint + typecheck + test (CI gate)

docker: ## Build the runtime Docker image
	docker build -t quantumium .

docker-test: ## Run the test suite inside Docker
	docker compose run --rm test

clean: ## Remove build artifacts and caches
	rm -rf build dist *.egg-info src/*.egg-info \
	       .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov \
	       docs/_build
	find . -type d -name __pycache__ -exec rm -rf {} +
