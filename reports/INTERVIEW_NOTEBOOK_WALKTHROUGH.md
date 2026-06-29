# 🗣️ Interview Notebook Walkthrough

## Project Name

**Healthcare Risk Adjustment Analytics Command Center**

## 30-Second Pitch

I built a healthcare analytics project that simulates how a payer could prioritize members for risk adjustment documentation review. The project uses a HIPAA-safe synthetic dataset, trains machine learning models to predict suspect documentation opportunities, ranks members into review tiers, estimates potential value, and deploys the workflow through a Streamlit decision-support app.

The main strength is that it does not stop at model accuracy. It turns model scores into an operational review queue, provider action table, plan-level financial summary, and an interactive dashboard that a healthcare analytics team could actually use.

---

## 1. Business Problem

Healthcare payers have limited chart-review and provider-outreach capacity. They cannot manually review every member.

The business questions are:

- Which members should be reviewed first?
- Which providers have the highest concentration of documentation opportunities?
- How much value could a focused review sprint create?
- Which plan segments may have risk-transfer exposure?
- Which model features are driving the risk signals?

This project solves the problem by building a ranked review workflow for risk adjustment and payer analytics teams.

---

## 2. Why This Project Matters For Healthcare Analytics

Risk adjustment is important because health plans need accurate documentation of member complexity. If clinical conditions are missing, incomplete, or not documented correctly, the member's risk score may not reflect their actual disease burden.

A strong healthcare analyst or data scientist should be able to connect:

- clinical signals
- utilization patterns
- financial exposure
- provider operations
- model explainability
- stakeholder-ready reporting

This project demonstrates that full chain.

---

## 3. Data Strategy

The dataset is fully synthetic and HIPAA-safe.

I created synthetic member records instead of using real patient data because healthcare data often contains PHI and cannot be shared publicly. The synthetic design still follows realistic healthcare patterns, so the project can demonstrate domain thinking without privacy risk.

### Member-Level Fields

| Category | Example Fields | Why It Matters |
|---|---|---|
| Demographics | Age, gender, region | Risk patterns vary across member populations |
| Plan design | Plan ID, actuarial value, member months | Needed for payer and transfer-style analysis |
| Utilization | Visits, ER visits, inpatient admits, RX count | Higher utilization can signal unmanaged complexity |
| Clinical burden | Diabetes, COPD, CKD, CHF, depression | Chronic conditions drive risk and documentation needs |
| Documentation signals | Prior HCC flag, new diagnosis flag, care gaps | These are core review-queue indicators |
| Financial signals | RAF score, paid claims PMPM | Converts clinical risk into business interpretation |
| Target | SuspectFlag | Represents a member likely needing documentation review |

---

## 4. Notebook Workflow

The notebook is structured like a real analytics project:

1. **Project setup**
2. **Synthetic data generation**
3. **Data dictionary and quality checks**
4. **Exploratory data analysis**
5. **Predictive modeling**
6. **Model evaluation**
7. **Priority queue creation**
8. **Statistical validation**
9. **Plan-level risk-transfer illustration**
10. **Final output generation**

This sequence is intentional. It shows that the project moves from business context to reproducible analytics to deployable product.

---

## 5. Code Architecture

The reusable logic lives in:

```text
src/risk_adjustment_pipeline.py
```

The notebook imports reusable functions instead of keeping every function hidden inside notebook cells. This makes the project easier to maintain and easier to deploy in Streamlit.

### Main Functions

| Function | Purpose |
|---|---|
| `generate_synthetic_members()` | Creates the healthcare member dataset |
| `train_models()` | Trains Logistic Regression and Random Forest models |
| `score_members()` | Scores all members and creates risk ranks |
| `summarize_priority_queue()` | Aggregates members by priority tier |
| `provider_action_table()` | Shows provider-level action opportunities |
| `plan_transfer_table()` | Creates plan-level financial interpretation |
| `classification_summary()` | Builds confusion matrix and classification report |
| `feature_importance()` | Extracts top model drivers |

This structure shows software engineering discipline, not just notebook experimentation.

---

## 6. Data Science Decisions

### Why Logistic Regression?

Logistic Regression is useful in healthcare because it is interpretable, stable, and easy to explain to stakeholders. In this project, it also performed best by ROC AUC.

For risk adjustment work, interpretability matters because operations, compliance, and clinical stakeholders need to understand why a member is being prioritized.

### Why Random Forest?

Random Forest was added as a nonlinear comparison model. It can capture interactions between utilization, chronic burden, and plan characteristics.

Comparing both models shows that I did not assume one algorithm was automatically best. I evaluated multiple options and selected based on performance and business usefulness.

### Why Class Weight Balancing?

The suspect documentation flag is not evenly distributed. Most members are not suspect. Class weighting helps the model pay more attention to the minority class instead of optimizing only for the majority class.

---

## 7. Evaluation Metrics

I used healthcare-relevant ranking metrics, not only basic accuracy.

| Metric | Meaning | Why It Matters |
|---|---|---|
| ROC AUC | Overall ability to separate suspect and non-suspect members | Good general discrimination measure |
| Average precision | Precision-recall performance for imbalanced data | Better than accuracy when positives are less common |
| Precision | Of flagged members, how many were true suspects | Important for review efficiency |
| Recall | Of actual suspects, how many were found | Important for missed opportunity reduction |
| Top-decile lift | How much better the top 10 percent is than baseline | Matches limited review capacity |
| Top-decile capture | Percent of suspects captured in the top 10 percent | Shows value of the ranked queue |

### Best Result

The best model was **Logistic Regression**.

Key results:

- ROC AUC: **0.792**
- Average precision: **0.526**
- Top-decile lift: **3.49x**
- Top-decile capture: **34.9%**
- Critical-tier suspect rate: **74.7%**

The most important takeaway is that the model creates a highly concentrated review queue. The top-priority group has a much higher suspect rate than the baseline population.

---

## 8. Operational Output

The model output becomes a workflow.

### Priority Tiers

Members are sorted by predicted suspect probability and placed into:

- Monitor
- Medium
- High
- Critical

This helps operations teams decide where to focus first.

### Provider Action Table

The provider action table summarizes:

- member count
- suspect rate
- average RAF score
- average claims PMPM
- critical member count
- estimated opportunity PMPM

This is useful because provider outreach often happens at the provider group level, not only at the individual member level.

### Top Member Worklist

The project exports a ranked worklist that could be used by:

- risk adjustment teams
- chart review vendors
- provider relations teams
- care-gap outreach teams

This is where the project becomes more than a model. It becomes a usable analytics product.

---

## 9. Financial Interpretation

The project includes two financial views:

### Estimated Opportunity PMPM

For each member, estimated opportunity is calculated using:

- suspect probability
- RAF score
- actuarial value
- an assumed PMPM factor

This is not meant to be a production payment model. It is an illustrative business-value layer that helps translate model results into operational prioritization.

### Plan Risk-Transfer Summary

The notebook also creates a simplified plan-level table using:

- plan risk score
- actuarial value
- member months
- state average risk index
- transfer PMPM direction

This demonstrates awareness of payer risk adjustment concepts and how member-level risk can roll up into plan-level exposure.

---

## 10. Streamlit App Explanation

The Streamlit app turns the analysis into a user-facing decision tool.

### What Users Can Do

- Filter by market, plan, provider group, age, priority tier, and probability
- Choose workflow presets
- Simulate monthly review capacity
- See net review value after cost assumptions
- Download prioritized member queues
- Compare provider groups
- Inspect feature importance
- Score a single member scenario

### Why This Matters

A notebook is useful for analysis, but a dashboard is useful for decision-making.

The app shows that I can take a data science workflow and convert it into something a stakeholder could use without reading Python code.

---

## 11. How To Explain The Notebook In An Interview

### Step 1: Start With The Business Context

"The project is about helping a healthcare payer prioritize members for risk adjustment documentation review. Since review capacity is limited, the model ranks members by likelihood of being a suspect documentation opportunity."

### Step 2: Explain The Data

"I used synthetic member-level payer data with demographics, utilization, chronic conditions, RAF score, claims PMPM, plan information, and provider group. The data is HIPAA-safe but designed to reflect realistic relationships."

### Step 3: Explain The Model

"I trained Logistic Regression and Random Forest models. Logistic Regression performed best and was also easier to explain, which matters in healthcare operations."

### Step 4: Explain The Metrics

"I focused on ROC AUC and average precision, but the most business-relevant metrics were top-decile lift and capture because teams can only review a limited number of members."

### Step 5: Explain The Output

"The model scores members, creates priority tiers, exports a worklist, summarizes provider group opportunities, and feeds a Streamlit app for interactive review planning."

### Step 6: Explain The Impact

"The project shows how analytics can reduce manual review waste by focusing attention on members and providers with the strongest risk signals."

---

## 12. Strong Interview Answer

**Question:** Tell me about this project.

**Answer:**

I built an end-to-end healthcare risk adjustment analytics project focused on payer operations. The business problem was that health plans cannot manually review every member, so they need a data-driven way to prioritize members who may have missing or incomplete documentation.

I generated a synthetic, HIPAA-safe member dataset with demographics, utilization, chronic conditions, RAF score, claims PMPM, plan attributes, provider groups, and a suspect documentation target. Then I built a reusable Python pipeline to train Logistic Regression and Random Forest models, evaluate them with healthcare-relevant metrics, and score all members.

The best model was Logistic Regression with a ROC AUC of about 0.79. More importantly, the top decile had about 3.5x lift, which means the highest-ranked members were much more likely to be true documentation opportunities than the baseline population. I then converted model scores into priority tiers, provider action tables, and downloadable worklists.

Finally, I deployed the workflow as a Streamlit app with filters, capacity simulation, financial opportunity estimates, provider summaries, model explainability, and individual member scoring. This project demonstrates the full workflow from healthcare business problem to machine learning model to operational decision-support tool.

---

## 13. Likely Interview Questions And Answers

### Why did you use synthetic data?

Healthcare data often contains PHI, claims details, and proprietary plan information. Synthetic data lets me demonstrate the workflow publicly while avoiding privacy concerns. I designed the synthetic relationships to mirror realistic healthcare patterns.

### Why not use accuracy as the main metric?

Accuracy can be misleading when the positive class is smaller. A model could appear accurate by mostly predicting the majority class. For this project, ranking quality matters more, so I used ROC AUC, average precision, top-decile lift, and top-decile capture.

### What is top-decile lift?

Top-decile lift compares the suspect rate in the highest-scored 10 percent of members to the baseline suspect rate. If the lift is 3.49x, the top decile is about 3.49 times more concentrated with suspect members than the full population.

### Why is interpretability important?

Healthcare stakeholders need to trust the reason a member is prioritized. Logistic Regression coefficients and feature importance help explain the model drivers, which is useful for clinical, operations, and compliance conversations.

### How would this change with real data?

With real data, I would add claims history, diagnosis codes, procedure codes, encounter completeness, provider documentation patterns, pharmacy adherence, prior-year HCCs, chart review outcomes, and temporal validation. I would also monitor drift and validate fairness across segments.

### What would you improve next?

I would add SHAP explainability, model monitoring, threshold tuning by review capacity, historical validation, chart-review feedback loops, and role-based app views for executives, analysts, and provider operations teams.

---

## 14. Data Science Skills Demonstrated

- Healthcare domain framing
- Synthetic data generation
- Exploratory data analysis
- Feature engineering
- Classification modeling
- Imbalanced classification evaluation
- Model comparison
- Ranking and prioritization
- Operational analytics
- Provider-level aggregation
- Financial interpretation
- Streamlit deployment
- Dashboard UI and UX
- Professional documentation
- Interview-ready storytelling

---

## 15. Project Ownership Story

I treated the project like a real stakeholder problem. I started with the operational question, created a safe dataset, built reusable pipeline code, evaluated the model with business-relevant metrics, generated stakeholder-ready outputs, and deployed an interactive app.

The strongest decision was focusing on review prioritization instead of only model accuracy. That made the project more realistic for healthcare analytics roles because the real value is not just predicting a flag. The value is helping a team decide what action to take next.

---

## 16. 2-Minute Version

This project is a healthcare payer analytics workflow for risk adjustment review prioritization. I built it because health plans have limited review capacity and need to identify members who are most likely to have documentation opportunities.

I created a synthetic HIPAA-safe dataset with 5,000 members. The data includes age, gender, region, provider group, plan, member months, utilization, chronic conditions, RAF score, claims PMPM, care gaps, and documentation flags.

I trained Logistic Regression and Random Forest models. Logistic Regression performed best with a ROC AUC around 0.79 and a top-decile lift around 3.5x. I chose ranking metrics because the business problem is not just classification. It is deciding which members should be reviewed first.

After scoring members, I created priority tiers, provider action tables, top-member worklists, and a plan-level risk-transfer illustration. Then I built a Streamlit app so users can filter segments, adjust review capacity, estimate opportunity, download queues, inspect provider groups, and score individual member scenarios.

The project demonstrates healthcare domain knowledge, machine learning, explainability, operational analytics, and deployment.

---

## 17. 5-Minute Deep Dive

If asked for a deeper explanation, use this structure:

1. **Problem:** Health plans need to prioritize limited documentation review resources.
2. **Data:** Synthetic member-level payer dataset, designed to be HIPAA-safe.
3. **Features:** Demographics, utilization, chronic conditions, RAF, claims, plan, provider group, care gaps.
4. **Target:** Suspect documentation opportunity.
5. **Models:** Logistic Regression and Random Forest.
6. **Evaluation:** ROC AUC, average precision, precision, recall, top-decile lift, top-decile capture.
7. **Selection:** Logistic Regression won and was interpretable.
8. **Operationalization:** Scores became tiers, provider summaries, worklists, and app views.
9. **Deployment:** Streamlit app for decision support and scenario testing.
10. **Next Steps:** SHAP, real claims data, temporal validation, monitoring, production feedback loop.

---

## 18. Resume Bullet Options

- Built an end-to-end healthcare risk adjustment analytics app using Python, scikit-learn, and Streamlit to prioritize 5,000 synthetic payer members for documentation review.
- Developed Logistic Regression and Random Forest models, achieving ROC AUC of 0.792 and top-decile lift of 3.49x for suspect documentation prioritization.
- Converted model scores into provider action tables, priority tiers, financial opportunity estimates, and downloadable review worklists.
- Deployed an interactive Streamlit command center with filters, capacity planning, model explainability, and single-member scenario scoring.

---

## 19. Final Interview Message

The most important thing to communicate is this:

**This is not only a machine learning notebook. It is a healthcare decision-support workflow.**

That distinction makes the project stronger for healthcare data science and analyst roles because it shows that I can connect analytics to real business action.
