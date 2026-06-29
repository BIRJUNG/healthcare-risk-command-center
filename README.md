# Healthcare Risk Adjustment Analytics

Portfolio project for healthcare data science and healthcare analytics roles.

## Project summary

This project simulates a payer-side healthcare analytics workflow for risk adjustment, suspect member prioritization, and plan-level financial interpretation. It converts synthetic member demographics, utilization, chronic-condition signals, RAF scores, care gaps, member months, and actuarial value into an interactive Streamlit command center for documentation, coding, and provider outreach teams.

The dataset is fully synthetic and HIPAA-safe. It is designed to demonstrate domain fluency without using real PHI, claims, or proprietary payer data.

## Business problem

Health plans need to identify members whose documented risk may not fully reflect their clinical complexity. Reviewing every member is expensive, so analytics teams need a ranked queue that helps operations focus chart review and care-gap outreach on the highest-opportunity members and provider groups.

## What changed from the original notebook

- Rebuilt the notebook into a professional end-to-end case study.
- Removed duplicate data generation, scratch plots, unrelated cells, and unsupported performance claims.
- Added a reusable Python pipeline in `src/risk_adjustment_pipeline.py`.
- Expanded the synthetic data model to include utilization, chronic conditions, HCC-like indicators, care gaps, claims PMPM, plan actuarial value, and member months.
- Added model comparison, ROC and precision-recall evaluation, top-decile lift, feature importance, provider action tables, priority tiers, and plan risk-transfer logic.
- Added a deployable Streamlit app with segment filters, review-capacity simulation, provider strategy views, financial impact modeling, model explainability, scenario scoring, and CSV downloads.
- Added portfolio-ready documentation and resume/LinkedIn snippets.

## Key results

| Metric | Result |
|---|---:|
| Synthetic members | 5,000 |
| Suspect member rate | 17.9% |
| Best model | Logistic Regression |
| ROC AUC | 0.792 |
| Average precision | 0.526 |
| Top-decile lift | 3.49x |
| Top-decile capture | 34.9% |
| Critical-tier suspect rate | 74.7% |
| Critical-tier average claims PMPM | $2,068 |

## Business impact narrative

The model turns member-level data into a prioritized worklist. The highest 3% of scored members are labeled `Critical` and show a 74.7% suspect rate, compared with a 17.9% population baseline. This creates a clearer operating strategy for healthcare teams: review the most likely documentation opportunities first, identify provider groups with concentrated risk, and track expected opportunity by plan and member segment.

## Streamlit app

The app is designed as a modern dark glass healthcare operations cockpit. It gives users the ability to:

- Start from quick workflow presets for executive overview, chart review, provider outreach, or care-gap cleanup.
- Filter by region, plan, provider group, age range, priority tier, and suspect probability.
- Adjust monthly chart-review capacity and per-review cost.
- See guided next-best-action recommendations, net review value, queue lift, review precision, and suspect capture.
- Download the current worklist for risk adjustment operations.
- Inspect provider group concentration and plan-level transfer exposure.
- Explain model drivers and score a single member scenario interactively.

## Repository structure

```text
.
|-- data/
|   `-- synthetic_healthcare_risk_members.csv
|-- notebooks/
|   `-- Healthcare_Risk_Adjustment_Analytics.ipynb
|-- outputs/
|   |-- feature_importance.csv
|   |-- model_metrics.csv
|   |-- plan_risk_transfer_summary.csv
|   |-- priority_tier_summary.csv
|   |-- provider_action_table.csv
|   |-- scored_member_priority_queue.csv
|   |-- top_25_member_opportunities.csv
|   `-- figures/
|-- src/
|   `-- risk_adjustment_pipeline.py
|-- .streamlit/
|   `-- config.toml
|-- PROFILE_PROJECT_SNIPPETS.md
|-- README.md
|-- requirements-dev.txt
|-- streamlit_app.py
`-- requirements.txt
```

## Methods used

- Synthetic healthcare data generation
- Exploratory data analysis
- Logistic Regression and Random Forest classification
- ROC AUC and precision-recall evaluation
- Top-decile lift and ranked-list capture
- Feature importance analysis
- Statistical testing with t-test and chi-square
- OLS-based RAF adjustment check
- Provider group action planning
- Simplified plan-level risk transfer modeling

## How to run

Run the interactive app:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Run the notebook:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
jupyter notebook notebooks/Healthcare_Risk_Adjustment_Analytics.ipynb
```

The notebook writes refreshed artifacts into `data/` and `outputs/`.

## Streamlit Cloud deployment

Use these settings when creating the Streamlit Cloud app:

- Repository: this GitHub repository
- Branch: `main`
- Main file path: `streamlit_app.py`
- Python version: `3.11`
- Dependency file: `requirements.txt`

## Profile positioning

Use this project to support applications for:

- Healthcare Data Analyst
- Healthcare Data Scientist
- Risk Adjustment Analyst
- Clinical Data Analyst
- Payer Analytics Analyst
- Population Health Analyst
- Healthcare Business Intelligence Analyst

The strongest positioning is: healthcare domain analytics plus practical machine learning plus business-facing prioritization.

## Limitation

This is a synthetic portfolio project, not a production risk-adjustment payment model. The transfer logic is illustrative and should not be interpreted as a CMS, ACA, or payer-certified regulatory calculation.
