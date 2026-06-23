# 📊 Churn Intelligence Platform

> Transform raw subscription transaction data into predictive retention intelligence — churn risk scores, CLV projections, cohort analysis, revenue forecasts, and AI-generated executive summaries.

[![CI](https://github.com/FomkaHamilton/churn-intelligence-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/FomkaHamilton/churn-intelligence-platform/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

---

## What this is

Most analytics tools stop at reporting — they tell you what happened. This platform goes further, answering questions that drive business action:

> *"Which customers are about to leave, how much future revenue is at risk, why are they leaving, and what should we do about it?"*

It is a full-stack analytics and machine learning platform built as a public portfolio project. Every line of code in this repository follows production engineering standards — typed, tested, linted, and containerised — to demonstrate what a Senior Analytics Engineer builds when they sit down to solve a real retention problem.

---

## For non-technical readers — what was built and why

This section explains each phase of the project in plain language, so anyone reviewing this portfolio can understand what problem was solved and what skill it demonstrates.

---

### Phase 1 — The Foundation
**What it is:** Setting up the "building" before the furniture goes in.

Before writing a single line of analysis code, a professional engineer establishes the infrastructure that makes a project maintainable, reproducible, and safe to share publicly. This phase created:

- A structured folder layout so every module has a logical home
- A configuration system that reads settings from a file and environment variables — no hard-coded values anywhere
- A structured logging system that records what the application is doing without ever logging personal customer data
- A CI/CD pipeline (automated checks that run on every code change via GitHub Actions) covering code style, type safety, and security scanning
- A Dockerfile so the application runs identically on any machine or cloud environment

**Why it matters:** A project without this foundation becomes brittle, hard to hand off, and dangerous to run in production. Building it first is the professional approach.

---

### Phase 2 — The Data Layer
**What it is:** Teaching the platform to safely accept and understand raw data files.

A business uploads a CSV or Excel file of transaction records. This phase built everything that happens between "file dropped" and "data ready for analysis":

- **File validation** — rejects unsupported formats, enforces a size limit, and detects potential security risks in uploaded files (CSV formula injection)
- **Schema normalisation** — intelligently maps common column name variants (`revenue`, `amount`, `order_date`, `user_id`) to the internal standard so users don't need to rename their files
- **Date format detection** — automatically detects whether dates are American (MM/DD/YYYY), European (DD/MM/YYYY), or ISO format, and asks the user to confirm when it's ambiguous
- **Data quality checks** — identifies null values, future dates, negative amounts, and duplicate records, each flagged with a severity level (error, warning, or info)
- **Sample data generator** — creates a realistic 10,000-customer synthetic dataset for demo purposes, with a built-in churn model that targets realistic AUC scores (0.78–0.85)

**Why it matters:** Garbage in, garbage out. A machine learning model trained on bad data produces wrong predictions. This layer ensures analysts can trust what flows into the system.

---

### Phase 3 — The Analytics Layer
**What it is:** Computing the core business health metrics from transaction history.

Once data is validated, this phase calculates the key indicators a subscription business uses to measure its health:

- **RFM Features** — For each customer: Recency (how long since they last paid), Frequency (how many times they've paid), and Monetary value (how much they've spent in total). These three numbers are the foundation of almost every customer scoring model in the industry.
- **Churn Labels** — Determines which customers have "churned" (stopped paying) based purely on gaps in their transaction history. Crucially, the platform *never* uses a "subscription_status" column as a shortcut — doing so would make the model trivially accurate in testing but useless in production, a common mistake called *label leakage*. A `LABEL_LEAKAGE_COLUMNS` blocklist and automated test enforce this permanently.
- **Cohort Retention Matrix** — Groups customers by the month they first subscribed, then tracks what percentage of each group was still paying in month 1, month 2, month 3, and so on. This is the standard way subscription businesses measure product stickiness.
- **KPI Time Series** — Month-by-month trends for MRR (Monthly Recurring Revenue), active subscriber count, ARPU (Average Revenue Per User), and monthly churn rate.
- **Analytics Dashboard Page** — All of the above rendered as interactive Plotly charts in the Streamlit app, with a colour-coded cohort heatmap.

**Why it matters:** These metrics are the vocabulary of subscription businesses. Every VP of Product, CFO, and Growth team lead speaks in these terms. An analyst who can build this pipeline from raw transactions is immediately valuable.

---

### Phase 4 — The Machine Learning Layer
**What it is:** Training a model that predicts which specific customers are likely to leave.

This phase moves from "describing what happened" to "predicting what will happen":

- **Temporal Train/Test Split** — A custom `TimeSeriesChurnSplit` class splits the customer population by when they first joined: older customers train the model, newer customers test it. This mirrors real-world deployment — you train on historical data and score new customers. A standard random split would be incorrect here because newer customers could accidentally inform the model's training, a form of data leakage.
- **Churn Prediction Model** — An ensemble of two algorithms: Logistic Regression (interpretable, well-calibrated probabilities) and Random Forest (captures non-linear patterns). Both use `class_weight='balanced'` to handle the typical 10–25% churn rate without over-predicting the majority class. The final score is the average of both models.
- **SHAP Explainability** — For every prediction, SHAP (SHapley Additive exPlanations) values explain *why* the model scored a customer as high risk. This is critical for business adoption — stakeholders need to trust and understand predictions, not just receive a black-box score.
- **Kaplan-Meier CLV** — Customer Lifetime Value is estimated using survival analysis (the same statistical technique used in medical research to model "time to event"). This produces a realistic expected remaining lifetime for each customer, which is multiplied by their spending rate to get a dollar CLV estimate.
- **Customer Segmentation** — Every customer is assigned to one of five segments (New, Healthy, At Risk, High Value, Churned) based on their churn probability and spending, with a recency-based fallback when no model has been trained yet.
- **Predictions Dashboard Page** — A "Train Model" button triggers the full pipeline, then displays the ROC curve, SHAP feature importance chart, a table of the 20 highest-risk customers, a segment distribution donut chart, and CLV summary cards.

**Why it matters:** This is the core deliverable. Predicting churn with a well-engineered, explainable model — and presenting it clearly — is the difference between an analyst role and a data scientist role.

---

### Phase 5 — The Forecasting Layer *(in progress)*
**What it is:** Projecting future revenue and subscriber counts 12 months ahead.

Knowing who churned is useful. Knowing how much revenue the business will generate next quarter is what the CFO needs for planning. This phase will build a time-series forecasting system that:

- Projects monthly revenue and subscriber counts 12 months forward
- Produces confidence intervals (the forecast band the business should plan within)
- Uses Holt-Winters exponential smoothing (statsmodels) as the default — reliable, fast, no extra dependencies
- Optionally uses Prophet (Meta's forecasting library) as a swap-in alternative for seasonal data

---

### Phases 6–9 — What's coming

| Phase | Plain-English Summary |
|---|---|
| **6 — AI Insights** | Feed the model results into Claude or GPT and get a written executive briefing: "Revenue at risk is $X. The top driver is Y. Recommended action: Z." Works with or without an API key (template fallback). |
| **7 — Full Dashboard** | Polish all eight pages into a cohesive product experience with navigation state management and export capabilities. |
| **8 — Hardening** | Push test coverage above 80%, add integration tests that run the full pipeline end-to-end, and profile performance on large datasets. |
| **9 — Portfolio Polish** | Architecture diagrams, annotated screenshots, a project walkthrough doc, and a live demo link. |

---

## Technical features

| Capability | Description | Technology |
|---|---|---|
| Data Ingestion | CSV/XLSX upload with schema validation and quality profiling | Pandas, Pydantic |
| Cohort Analysis | Monthly retention matrices, MRR trends, ARPU time series | Pandas, Plotly |
| Churn Prediction | LR + RF ensemble with SHAP explainability | scikit-learn, SHAP |
| CLV Modeling | Kaplan-Meier survival analysis for expected lifetime value | lifelines |
| Forecasting | 12-month revenue + subscriber forecasts with confidence bands | statsmodels / Prophet |
| AI Insights | Executive summaries and intervention recommendations | Anthropic / OpenAI / Template |
| Dashboard | Interactive multi-page Streamlit app | Streamlit, Plotly |

---

## Quick start

**Option A — Docker (recommended)**

```bash
cp .env.example .env        # add your API key if you want AI insights (optional)
docker compose up
# open http://localhost:8501
```

**Option B — Local Python**

```bash
git clone https://github.com/FomkaHamilton/churn-intelligence-platform.git
cd churn-intelligence-platform

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt
cp .env.example .env           # optional: add ANTHROPIC_API_KEY or OPENAI_API_KEY

streamlit run app.py
```

**Generate sample data**

```bash
python data/sample/generator.py
# creates data/sample/subscriptions_sample.csv (~10 k realistic records)
```

The app also has a **Load sample dataset** button on the Upload page so you can explore without any files.

---

## Input format

Three columns are required:

| Column | Type | Example |
|---|---|---|
| `customer_id` | string | `CUST-00123` |
| `transaction_date` | date | `2024-03-15` |
| `transaction_amount` | decimal | `49.99` |

Optional enrichment columns:

| Column | What it unlocks |
|---|---|
| `subscription_id` | Subscription-level analysis |
| `country` | Geographic segmentation |
| `product` | Product-tier breakdown |

Common column name variants are remapped automatically (`revenue`, `amount`, `user_id`, `order_date`, etc.).

---

## Configuration

All tunable values live in `config/settings.yaml`:

```yaml
churn:
  window_days: 90           # days of inactivity that defines "churned" (UI-adjustable)

modeling:
  test_size: 0.20           # temporal holdout fraction for model evaluation

forecasting:
  horizon_months: 12
  backend: "statsmodels"    # swap to "prophet" for seasonal data (see requirements-optional.txt)

clv:
  use_survival_analysis: true
```

Any value can be overridden via environment variable (e.g. `CHURN_WINDOW_DAYS=60`).

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Streamlit Dashboard                    │
│          (app.py + src/visualization/)                   │
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
         │  Claude │ OpenAI │ Template│
         └───────────────────────────┘
```

---

## Repository structure

```
churn-intelligence-platform/
├── app.py                    # Streamlit entry point (all pages)
├── Makefile                  # make install | run | test | lint | audit
├── Dockerfile                # multi-stage production container
├── docker-compose.yml
│
├── config/
│   └── settings.yaml         # all tunable parameters
│
├── src/
│   ├── config/               # Pydantic settings, shared constants
│   ├── utils/                # structured logging, exception hierarchy, types
│   ├── ingestion/            # file upload + injection detection
│   ├── preprocessing/        # cleaning, schema enforcement, date parsing
│   ├── feature_engineering/  # RFM features, churn label construction
│   ├── analytics/            # cohort retention, KPI calculations
│   ├── modeling/             # churn model, CLV, segmentation, SHAP
│   ├── forecasting/          # time-series forecasting backends
│   ├── insights/             # LLM insight generation (Phase 6)
│   ├── visualization/        # Plotly chart builders (Streamlit-free)
│   └── database/             # SQLite session persistence
│
├── tests/
│   ├── unit/                 # 159 unit tests across all modules
│   ├── integration/          # end-to-end pipeline tests (Phase 8)
│   └── fixtures/             # synthetic data factories
│
└── data/
    ├── raw/                  # uploaded files (gitignored)
    ├── processed/            # cleaned outputs (gitignored)
    └── sample/               # synthetic demo data (committed)
```

---

## Development

```bash
make install        # install all dependencies
make test           # run tests with coverage report
make lint           # ruff check + mypy type check
make format         # auto-format with ruff
make sample-data    # generate synthetic demo dataset
make audit          # pip-audit security scan
```

**Test coverage target: 80%+**  Current: 159 tests, all passing.

---

## Build progress

| Phase | Status | What was built |
|---|---|---|
| 1 — Foundation | ✅ Complete | Repo scaffold, Pydantic config, structured logging, GitHub Actions CI, Docker |
| 2 — Data Layer | ✅ Complete | Upload engine, schema normalisation, date parser, quality checker, sample generator |
| 3 — Analytics Layer | ✅ Complete | RFM features, churn labels (leakage-protected), cohort retention matrix, KPI time series |
| 4 — ML Layer | ✅ Complete | Temporal train/test split, LR+RF ensemble, SHAP, Kaplan-Meier CLV, segmentation |
| 5 — Forecasting | 🔄 In progress | Holt-Winters + Prophet forecasters, 12-month revenue/subscriber projections |
| 6 — AI Insights | ⏳ Planned | Claude/OpenAI/Template adapter, executive summaries, intervention recommendations |
| 7 — Dashboard | ⏳ Planned | Full multi-page polish, state management, export capabilities |
| 8 — Hardening | ⏳ Planned | 80%+ coverage, integration tests, performance profiling |
| 9 — Portfolio Polish | ⏳ Planned | Architecture diagrams, annotated screenshots, live demo |

---

## Design decisions worth noting

**Why no `subscription_status` as a model feature?**
Using a "status" column to predict churn is circular — the column already encodes the answer. The model would score 100% accuracy in testing and perform at random in production. A `LABEL_LEAKAGE_COLUMNS` blocklist enforces this at the code level and is tested in CI.

**Why a temporal split instead of random?**
Random train/test splits are incorrect for time-series data. If a customer joins in March and we train on their March data but test on their January data, we're predicting the past. The `TimeSeriesChurnSplit` class ensures the model is always evaluated on customers who were *newer* than everyone in the training set.

**Why class_weight='balanced' instead of SMOTE?**
SMOTE (synthetic oversampling) generates artificial data points which can introduce subtle distribution shifts. Sklearn's `class_weight='balanced'` adjusts the loss function directly — mathematically equivalent for most models, with no synthetic data artefacts.

**Why no Prophet by default?**
Prophet requires C++ build tools which can fail silently on Windows. Making it optional (in `requirements-optional.txt`) means the platform works out of the box on every OS while still supporting Prophet via a config flag for users who want seasonal modelling.

---

## License

MIT — see [LICENSE](LICENSE).

---

*Built by [FomkaHamilton](https://github.com/FomkaHamilton) as a public portfolio project demonstrating production-grade analytics engineering.*
