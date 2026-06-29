"""Reusable pipeline for the healthcare risk adjustment portfolio project.

The project uses synthetic member-level data so it can demonstrate healthcare
analytics methods without exposing PHI or relying on proprietary claims data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_FEATURES = [
    "Age",
    "Visits",
    "ER_Visits",
    "Inpatient_Admits",
    "RX_Count",
    "ChronicCount",
    "Prior_HCC_Flag",
    "New_Diagnosis_Flag",
    "CareGapCount",
    "RAF_Score",
    "PaidClaimsPMPM",
    "MemberMonths",
]

CATEGORICAL_FEATURES = [
    "Region",
    "Gender",
    "PlanID",
    "ProviderGroup",
]

MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "SuspectFlag"


@dataclass(frozen=True)
class ModelRun:
    """Container for trained artifacts and test split outputs."""

    models: Dict[str, Pipeline]
    metrics: pd.DataFrame
    x_test: pd.DataFrame
    y_test: pd.Series
    test_index: pd.Index


def _sigmoid(value: np.ndarray | pd.Series | float) -> np.ndarray | pd.Series | float:
    return 1 / (1 + np.exp(-value))


def _one_hot_encoder() -> OneHotEncoder:
    """Return a dense one-hot encoder across scikit-learn versions."""

    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def generate_synthetic_members(n_members: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Generate a HIPAA-safe synthetic healthcare risk-adjustment dataset."""

    rng = np.random.default_rng(seed)

    region = rng.choice(
        ["West", "Midwest", "South", "Northeast"],
        size=n_members,
        p=[0.31, 0.22, 0.29, 0.18],
    )
    provider_group = rng.choice(
        [
            "Alpha Medical Group",
            "BayCare Network",
            "Cedar Health",
            "Delta IPA",
            "Evergreen Clinic",
            "Frontier Health",
            "Golden Valley",
            "Harbor Physicians",
        ],
        size=n_members,
        p=[0.14, 0.11, 0.12, 0.13, 0.12, 0.13, 0.13, 0.12],
    )
    gender = rng.choice(["F", "M"], size=n_members, p=[0.52, 0.48])
    age = np.clip(rng.normal(54, 16, size=n_members).round().astype(int), 18, 90)
    plan_id = rng.choice(
        ["PlanA_Silver", "PlanB_Gold", "PlanC_Bronze"],
        size=n_members,
        p=[0.45, 0.34, 0.21],
    )
    member_months = rng.choice([3, 6, 9, 12], size=n_members, p=[0.12, 0.22, 0.25, 0.41])
    av_lookup = {"PlanA_Silver": 0.70, "PlanB_Gold": 0.80, "PlanC_Bronze": 0.60}

    age_risk = (age - 50) / 10
    diabetes = rng.binomial(1, _sigmoid(-1.4 + 0.28 * age_risk), size=n_members)
    copd = rng.binomial(1, _sigmoid(-2.3 + 0.24 * age_risk), size=n_members)
    ckd = rng.binomial(1, _sigmoid(-2.6 + 0.30 * age_risk + 0.45 * diabetes), size=n_members)
    chf = rng.binomial(1, _sigmoid(-3.0 + 0.34 * age_risk + 0.40 * ckd), size=n_members)
    depression = rng.binomial(1, _sigmoid(-2.0 + 0.08 * age_risk), size=n_members)
    chronic_count = diabetes + copd + ckd + chf + depression

    visits = np.clip(
        rng.poisson(2.8 + 1.35 * chronic_count + 0.025 * np.maximum(age - 50, 0)),
        0,
        32,
    )
    er_visits = np.clip(
        rng.poisson(0.12 + 0.28 * chronic_count + 0.16 * (visits >= 8)),
        0,
        9,
    )
    inpatient_admits = np.clip(
        rng.poisson(0.04 + 0.08 * chronic_count + 0.24 * chf + 0.18 * ckd + 0.10 * er_visits),
        0,
        6,
    )
    rx_count = np.clip(
        rng.poisson(2.3 + 1.8 * chronic_count + 1.0 * diabetes + 0.7 * chf),
        0,
        35,
    )
    care_gaps = np.clip(
        rng.poisson(0.35 + 0.35 * chronic_count + 0.18 * (visits <= 2)),
        0,
        9,
    )

    prior_hcc_probability = _sigmoid(
        -1.8 + 0.58 * chronic_count + 0.18 * inpatient_admits - 0.15 * care_gaps
    )
    prior_hcc_flag = rng.binomial(1, prior_hcc_probability, size=n_members)
    new_dx_flag = rng.binomial(
        1,
        _sigmoid(-2.2 + 0.10 * visits + 0.07 * rx_count + 0.28 * chronic_count + 0.18 * care_gaps),
        size=n_members,
    )

    provider_effect = pd.Series(provider_group).map(
        {
            "Alpha Medical Group": -0.08,
            "BayCare Network": 0.10,
            "Cedar Health": -0.02,
            "Delta IPA": 0.04,
            "Evergreen Clinic": -0.06,
            "Frontier Health": 0.13,
            "Golden Valley": 0.03,
            "Harbor Physicians": 0.09,
        }
    ).to_numpy()
    region_effect = pd.Series(region).map(
        {"West": -0.03, "Midwest": 0.02, "South": 0.08, "Northeast": -0.04}
    ).to_numpy()

    suspect_probability = _sigmoid(
        -3.10
        + 0.43 * chronic_count
        + 0.07 * visits
        + 0.06 * rx_count
        + 0.37 * er_visits
        + 0.54 * inpatient_admits
        + 0.44 * care_gaps
        + 0.58 * new_dx_flag
        - 0.47 * prior_hcc_flag
        + provider_effect
        + region_effect
    )
    suspect_flag = rng.binomial(1, suspect_probability, size=n_members)

    raf = (
        0.26
        + 0.0075 * np.maximum(age - 45, 0)
        + 0.17 * chronic_count
        + 0.18 * diabetes
        + 0.22 * copd
        + 0.34 * ckd
        + 0.43 * chf
        + 0.15 * depression
        + 0.24 * prior_hcc_flag
        + rng.normal(0, 0.18, size=n_members)
    )
    raf_score = np.round(np.clip(raf, 0.12, 4.25), 3)
    paid_claims_pmpm = np.round(
        np.clip(
            280
            + 3.8 * age
            + 130 * chronic_count
            + 245 * er_visits
            + 820 * inpatient_admits
            + 18 * rx_count
            + rng.normal(0, 110, size=n_members),
            80,
            None,
        ),
        2,
    )

    df = pd.DataFrame(
        {
            "MemberID": np.arange(100000, 100000 + n_members),
            "Region": region,
            "ProviderGroup": provider_group,
            "Age": age,
            "Gender": gender,
            "Visits": visits,
            "ER_Visits": er_visits,
            "Inpatient_Admits": inpatient_admits,
            "RX_Count": rx_count,
            "Diabetes": diabetes,
            "COPD": copd,
            "CKD": ckd,
            "CHF": chf,
            "Depression": depression,
            "ChronicCount": chronic_count,
            "Prior_HCC_Flag": prior_hcc_flag,
            "New_Diagnosis_Flag": new_dx_flag,
            "CareGapCount": care_gaps,
            "RAF_Score": raf_score,
            "PaidClaimsPMPM": paid_claims_pmpm,
            "SuspectFlag": suspect_flag,
            "PlanID": plan_id,
            "MemberMonths": member_months,
            "AV": pd.Series(plan_id).map(av_lookup).to_numpy(),
        }
    )
    df["AgeBand"] = pd.cut(
        df["Age"],
        bins=[17, 34, 49, 64, 74, 90],
        labels=["18-34", "35-49", "50-64", "65-74", "75+"],
    ).astype(str)
    df["OpportunitySegment"] = np.select(
        [
            (df["SuspectFlag"] == 1) & (df["RAF_Score"] >= df["RAF_Score"].quantile(0.75)),
            (df["SuspectFlag"] == 1),
            df["CareGapCount"] >= 3,
        ],
        ["High RAF suspect", "Documentation suspect", "Care gap follow-up"],
        default="Monitor",
    )
    return df


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", _one_hot_encoder(), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def train_models(df: pd.DataFrame, seed: int = 42) -> ModelRun:
    """Train baseline and nonlinear models and return comparable metrics."""

    x = df[MODEL_FEATURES]
    y = df[TARGET]
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.30,
        random_state=seed,
        stratify=y,
    )

    models: Dict[str, Pipeline] = {
        "Logistic Regression": Pipeline(
            steps=[
                ("preprocess", build_preprocessor()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        solver="lbfgs",
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("preprocess", build_preprocessor()),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=450,
                        max_depth=9,
                        min_samples_leaf=25,
                        class_weight="balanced",
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }

    metrics = []
    for name, model in models.items():
        model.fit(x_train, y_train)
        probability = model.predict_proba(x_test)[:, 1]
        predicted = (probability >= 0.50).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, predicted, average="binary", zero_division=0
        )
        metrics.append(
            {
                "Model": name,
                "ROC_AUC": roc_auc_score(y_test, probability),
                "Average_Precision": average_precision_score(y_test, probability),
                "Precision_at_0_50": precision,
                "Recall_at_0_50": recall,
                "F1_at_0_50": f1,
                "Top_Decile_Lift": top_decile_lift(y_test, probability),
                "Top_Decile_Capture": top_decile_capture(y_test, probability),
            }
        )

    return ModelRun(
        models=models,
        metrics=pd.DataFrame(metrics).sort_values("ROC_AUC", ascending=False).reset_index(drop=True),
        x_test=x_test,
        y_test=y_test,
        test_index=x_test.index,
    )


def top_decile_capture(y_true: Iterable[int], probability: Iterable[float]) -> float:
    score_frame = pd.DataFrame({"actual": list(y_true), "probability": list(probability)})
    cutoff = max(int(np.ceil(len(score_frame) * 0.10)), 1)
    top = score_frame.sort_values("probability", ascending=False).head(cutoff)
    total_positives = score_frame["actual"].sum()
    return 0.0 if total_positives == 0 else top["actual"].sum() / total_positives


def top_decile_lift(y_true: Iterable[int], probability: Iterable[float]) -> float:
    score_frame = pd.DataFrame({"actual": list(y_true), "probability": list(probability)})
    cutoff = max(int(np.ceil(len(score_frame) * 0.10)), 1)
    top_rate = score_frame.sort_values("probability", ascending=False).head(cutoff)["actual"].mean()
    baseline = score_frame["actual"].mean()
    return 0.0 if baseline == 0 else top_rate / baseline


def score_members(df: pd.DataFrame, model: Pipeline) -> pd.DataFrame:
    scored = df.copy()
    scored["SuspectProbability"] = model.predict_proba(scored[MODEL_FEATURES])[:, 1]
    scored["RiskRank"] = scored["SuspectProbability"].rank(method="first", ascending=False).astype(int)
    scored["PriorityTier"] = pd.qcut(
        scored["SuspectProbability"].rank(method="first"),
        q=[0, 0.70, 0.90, 0.97, 1.0],
        labels=["Monitor", "Medium", "High", "Critical"],
    )
    scored["EstimatedOpportunityPMPM"] = np.round(
        scored["SuspectProbability"]
        * scored["RAF_Score"]
        * scored["AV"]
        * 125,
        2,
    )
    return scored.sort_values("SuspectProbability", ascending=False).reset_index(drop=True)


def summarize_priority_queue(scored: pd.DataFrame) -> pd.DataFrame:
    return (
        scored.groupby("PriorityTier", observed=True)
        .agg(
            Members=("MemberID", "count"),
            SuspectRate=("SuspectFlag", "mean"),
            AvgProbability=("SuspectProbability", "mean"),
            AvgRAF=("RAF_Score", "mean"),
            AvgClaimsPMPM=("PaidClaimsPMPM", "mean"),
            EstOpportunityPMPM=("EstimatedOpportunityPMPM", "mean"),
        )
        .reset_index()
    )


def provider_action_table(scored: pd.DataFrame) -> pd.DataFrame:
    return (
        scored.groupby("ProviderGroup")
        .agg(
            Members=("MemberID", "count"),
            SuspectRate=("SuspectFlag", "mean"),
            AvgRAF=("RAF_Score", "mean"),
            AvgClaimsPMPM=("PaidClaimsPMPM", "mean"),
            CriticalMembers=("PriorityTier", lambda value: (value == "Critical").sum()),
            EstOpportunityPMPM=("EstimatedOpportunityPMPM", "mean"),
        )
        .sort_values(["CriticalMembers", "SuspectRate"], ascending=False)
        .reset_index()
    )


def plan_transfer_table(df: pd.DataFrame, premium_pmpm: float = 450) -> pd.DataFrame:
    """Create an illustrative ACA-style risk-transfer table by plan."""

    state_alrs = np.average(df["RAF_Score"] * df["AV"], weights=df["MemberMonths"])
    rows = []
    for plan, group in df.groupby("PlanID"):
        weighted_risk = np.average(group["RAF_Score"], weights=group["MemberMonths"])
        plrs = np.average(group["RAF_Score"] * group["AV"], weights=group["MemberMonths"])
        transfer_pmpm = (plrs - state_alrs) * premium_pmpm
        rows.append(
            {
                "PlanID": plan,
                "WeightedRisk": weighted_risk,
                "PLRS": plrs,
                "TotalMemberMonths": group["MemberMonths"].sum(),
                "TransferPMPM": transfer_pmpm,
                "TransferDirection": "Receivable" if transfer_pmpm > 0 else "Payable",
            }
        )
    return pd.DataFrame(rows).sort_values("TransferPMPM", ascending=False).reset_index(drop=True)


def classification_summary(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> Tuple[pd.DataFrame, np.ndarray]:
    probability = model.predict_proba(x_test)[:, 1]
    predicted = (probability >= 0.50).astype(int)
    report = classification_report(y_test, predicted, output_dict=True, zero_division=0)
    report_frame = pd.DataFrame(report).transpose()
    return report_frame, confusion_matrix(y_test, predicted)


def feature_importance(model: Pipeline, top_n: int = 15) -> pd.DataFrame:
    """Extract coefficients or feature importances from a fitted pipeline."""

    feature_names = model.named_steps["preprocess"].get_feature_names_out()
    estimator = model.named_steps["model"]
    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
        importance_name = "Importance"
    else:
        values = np.abs(estimator.coef_[0])
        importance_name = "AbsoluteCoefficient"
    return (
        pd.DataFrame({"Feature": feature_names, importance_name: values})
        .sort_values(importance_name, ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
