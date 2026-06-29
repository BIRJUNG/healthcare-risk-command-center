# Healthcare Risk Adjustment Analytics Case Study

## Role signal

This project is designed to support applications for healthcare data analyst, healthcare data scientist, payer analytics, risk adjustment analyst, and population health analytics roles.

## Problem

Healthcare payers need to identify members whose documented risk may not fully represent clinical complexity. Manual review capacity is limited, so analytics should prioritize members and provider groups where review is most likely to create operational value.

## Approach

1. Generated a HIPAA-safe synthetic dataset of 5,000 healthcare members.
2. Modeled healthcare-relevant attributes including RAF score, utilization, chronic burden, care gaps, member months, actuarial value, provider group, and plan.
3. Trained Logistic Regression and Random Forest models to predict suspect documentation opportunities.
4. Evaluated the workflow using ROC AUC, average precision, top-decile lift, and top-decile capture.
5. Converted predictions into priority tiers, provider action tables, top-member worklists, and a plan-level risk-transfer summary.
6. Built a Streamlit decision-support app for segment filtering, review-capacity planning, queue downloads, provider strategy, model explainability, and single-member scenario scoring.

## Results

| Result | Value |
|---|---:|
| Best model | Logistic Regression |
| ROC AUC | 0.792 |
| Average precision | 0.526 |
| Top-decile lift | 3.49x |
| Top-decile capture | 34.9% |
| Critical-tier suspect rate | 74.7% |
| Population suspect rate | 17.9% |

## Deliverables

- Executed notebook: `notebooks/Healthcare_Risk_Adjustment_Analytics.ipynb`
- Synthetic dataset: `data/synthetic_healthcare_risk_members.csv`
- Model metrics: `outputs/model_metrics.csv`
- Provider action table: `outputs/provider_action_table.csv`
- Priority tier summary: `outputs/priority_tier_summary.csv`
- Top member review queue: `outputs/top_25_member_opportunities.csv`
- Risk transfer summary: `outputs/plan_risk_transfer_summary.csv`
- Visuals: `outputs/figures/`
- Streamlit app: `streamlit_app.py`

## Healthcare interpretation

The model is useful because it produces a ranked worklist. The critical tier contains 150 members and has a 74.7% suspect rate, more than four times the baseline rate. This gives risk adjustment and provider operations teams a clear place to start before expanding to lower-priority tiers.

The Streamlit app turns that analysis into a workflow tool. A user can change the review capacity, filter to a market or provider group, estimate review value, inspect model drivers, and export the exact member queue for follow-up.

## Interview talking point

The key decision was to evaluate the model as a ranked healthcare operations queue, not just as a classifier. Top-decile lift and capture are more aligned with limited chart-review capacity than accuracy alone.
