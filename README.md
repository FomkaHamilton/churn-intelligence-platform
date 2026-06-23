# 📊 Churn Intelligence Platform

> Transform raw subscription transaction data into predictive retention intelligence — churn risk scores, CLV projections, cohort analysis, revenue forecasts, and AI-generated executive summaries.

[![CI](https://github.com/FomkaHamilton/churn-intelligence-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/FomkaHamilton/churn-intelligence-platform/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

---

## What this is

Most analytics projects stop at reporting. This platform answers:

> *Which customers are likely to churn, what future value is at risk, why are they churning, and what action should be taken?*

It moves beyond descriptive analytics into **predictive and prescriptive** territory — built to production-quality standards across the full stack: data engineering, machine learning, forecasting, and AI-assisted narrative generation.

---

## Features

| Capability | Description | Tech |
|---|---|---|
| **Data Ingestion** | CSV/XLSX upload with schema validation and quality profiling | Pandas, Pydantic |
| **Cohort Analysis** | Monthly retention matrices, MRR trends, ARPU | Pandas |
| **Churn Prediction** | Logistic Regression + Random Forest with SHAP explainability | scikit-learn, SHAP |
| **CLV Modeling** | Kaplan-Meier survival analysis for expected lifetime | lifelines |
| **Forecasting** | 12-month revenue & subscriber forecasts with confidence bands | statsmodels |
| **AI Insights** | Executive summaries and intervention recommendations | Claude / OpenAI |
| **Dashboard** | Interactive multi-page Streamlit app with Plotly charts | Streamlit, Plotly |

---

## Quick start

**Option A — Docker (recommended, one command)**

```bash
cp .env.example .env        # Add your API key if you want AI insights
docker compose up
# Open http://localhost:8501
```

**Option B — Local Python**

```bash
git clone https://github.com/FomkaHamilton/churn-intelligence-platform.git
cd churn-intelligence-platform

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
cp .env.example .env        # Add your API key (optional)

streamlit run app.py
```

**Generate sample data**

```bash
python data/sample/generator.py
# Creates data/sample/subscriptions_sample.csv (~10k realistic records)
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Streamlit Dashboard                    │
│          (app.py + pages/ + src/visualization/)          │
└──────────────────────┬──────────────────────────────────┘
                       │
         ┌─────────────▼─────────────┐
         │      Analytics Engine     │
         │  feature_engineering/     │
         │  analytics/               │
         │  modeling/                │
         │  forecasting/             │
         └─────────────┬─────────────┘
                       │
         ┌─────────────▼─────────────┐
         │       Data Layer          │
         │  ingestion/  ──► CSV/XLSX │
         │  preprocessing/           │
         │  database/ ──► SQLite     │
         └─────────────┬─────────────┘
                       │
         ┌─────────────▼─────────────┐
         │      AI Insight Layer     │
         │  insights/llm_client.py   │
         │  Claude │ OpenAI │ Template│
         └───────────────────────────┘
```

---

## Repository structure

```
churn-intelligence-platform/
├── app.py                    # Streamlit entry point
├── Makefile                  # make install | run | test | lint
├── Dockerfile                # Production container
├── docker-compose.yml

├── config/
│   └── settings.yaml         # All tunable parameters

├── src/
│   ├── config/               # Pydantic settings model
│   ├── utils/                # Logging, exceptions, types
│   ├── ingestion/            # File upload + validation
│   ├── preprocessing/        # Cleaning, schema enforcement
│   ├── feature_engineering/  # RFM, tenure, churn labels
│   ├── analytics/            # Cohort, KPIs, retention
│   ├── modeling/             # Churn model, CLV, segmentation
│   ├── forecasting/          # Time-series forecasting
│   ├── insights/             # LLM insight generation
│   ├── visualization/        # Plotly chart builders
│   └── database/             # SQLite session persistence

├── tests/
│   ├── unit/                 # Per-module unit tests
│   ├── integration/          # End-to-end pipeline tests
│   └── fixtures/             # Synthetic data factories

├── data/
│   ├── raw/                  # Uploaded files (gitignored)
│   ├── processed/            # Cleaned data (gitignored)
│   ├── sessions/             # SQLite DB (gitignored)
│   └── sample/               # Synthetic demo data (committed)

└── .github/workflows/ci.yml  # Lint → Type-check → Test → Audit
```

---

## Expected input format

Minimum required columns:

| Column | Type | Description |
|---|---|---|
| `customer_id` | string | Unique customer identifier |
| `transaction_date` | date | Date of the transaction |
| `transaction_amount` | float | Revenue value (must be ≥ 0) |

Optional but enriching:

| Column | Description |
|---|---|
| `subscription_id` | Links to a specific subscription |
| `subscription_status` | `active` / `cancelled` / etc. |
| `country` | For geographic segmentation |
| `product` | For product-level analysis |

The platform also accepts common column name variants automatically (`revenue`, `amount`, `order_date`, `user_id`, etc.).

---

## Configuration

All tunable values live in `config/settings.yaml`. Key settings:

```yaml
churn:
  window_days: 90           # Days of inactivity = churned (UI-configurable)

modeling:
  test_size: 0.20           # Temporal holdout fraction
  random_state: 42

forecasting:
  horizon_months: 12
  backend: "statsmodels"    # or "prophet" (see requirements-optional.txt)

clv:
  use_survival_analysis: true
```

Override any value via environment variable (e.g. `CHURN_WINDOW_DAYS=60`).

---

## Development

```bash
make install        # Install all dependencies
make test           # Run tests with coverage
make lint           # ruff + mypy
make sample-data    # Generate synthetic test dataset
make audit          # pip-audit security scan
```

**Coverage target: 80%+**

---

## Roadmap

| Phase | Status | Description |
|---|---|---|
| 1 — Foundation | ✅ **Complete** | Repo, config, logging, CI/CD |
| 2 — Data Layer | 🔄 In progress | Upload engine, validation, data quality |
| 3 — Analytics | ⏳ Planned | Cohort analysis, KPIs, retention metrics |
| 4 — ML Layer | ⏳ Planned | Churn model, CLV, segmentation |
| 5 — Forecasting | ⏳ Planned | Revenue + subscriber forecasts |
| 6 — AI Insights | ⏳ Planned | Executive summaries, recommendations |
| 7 — Dashboard | ⏳ Planned | Full interactive Streamlit experience |
| 8 — Hardening | ⏳ Planned | 80%+ test coverage, performance profiling |
| 9 — Portfolio polish | ⏳ Planned | Architecture diagrams, screenshots, docs |

**Future enhancements:** BG/NBD probabilistic CLV · real-time data connectors · multi-tenant support · email alerting · A/B test analysis module

---

## License

MIT — see [LICENSE](LICENSE).
