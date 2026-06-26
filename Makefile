# ─────────────────────────────────────────────────────────────────────────────
# Churn Intelligence Platform — Makefile
# Usage: make <target>
# ─────────────────────────────────────────────────────────────────────────────

.PHONY: install install-optional run test lint format clean sample-data audit help

PYTHON  := python
PIP     := pip

# ── Setup ─────────────────────────────────────────────────────────────────────

install:           ## Install all core and dev dependencies
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -r requirements-dev.txt

install-optional:  ## Install optional dependencies (Prophet forecasting)
	$(PIP) install -r requirements-optional.txt

# ── Development ───────────────────────────────────────────────────────────────

run:               ## Launch the Streamlit dashboard
	streamlit run app.py

sample-data:       ## Generate synthetic sample dataset for testing
	$(PYTHON) data/sample/generator.py

# ── Quality ───────────────────────────────────────────────────────────────────

test:              ## Run test suite with coverage report
	pytest --cov=src --cov-report=term-missing --cov-report=xml

lint:              ## Lint and type-check source code
	ruff check src/ tests/
	mypy src/

format:            ## Auto-format source code
	ruff format src/ tests/

audit:             ## Run security audit on project dependencies
	pip-audit -r requirements.txt -r requirements-dev.txt

# ── Maintenance ───────────────────────────────────────────────────────────────

clean:             ## Remove build artifacts, caches, and temp files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true
	find . -name "*.pyo" -delete 2>/dev/null; true
	rm -f coverage.xml .coverage 2>/dev/null; true
	rm -rf .pytest_cache .mypy_cache .ruff_cache 2>/dev/null; true

help:              ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
