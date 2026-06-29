from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.risk_adjustment_pipeline import (
    MODEL_FEATURES,
    TARGET,
    classification_summary,
    feature_importance,
    generate_synthetic_members,
    plan_transfer_table,
    provider_action_table,
    score_members,
    summarize_priority_queue,
    train_models,
)


st.set_page_config(
    page_title="Healthcare Risk Command Center",
    layout="wide",
    initial_sidebar_state="expanded",
)


ACCENT_BLUE = "#7dd3fc"
ACCENT_GREEN = "#34d399"
ACCENT_GOLD = "#fbbf24"
ACCENT_PINK = "#f472b6"
ACCENT_RED = "#fb7185"
TEXT_PRIMARY = "#f8fafc"
TIER_COLORS = {
    "Monitor": "#64748b",
    "Medium": ACCENT_BLUE,
    "High": ACCENT_GOLD,
    "Critical": ACCENT_RED,
}


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --panel: rgba(17, 19, 29, 0.72);
            --line: rgba(255, 255, 255, 0.13);
            --text: #f8fafc;
            --muted: #aab4c4;
            --blue: #7dd3fc;
            --green: #34d399;
            --gold: #fbbf24;
            --pink: #f472b6;
            --red: #fb7185;
        }

        html, body, [class*="css"] {
            font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            color: var(--text);
        }

        .stApp {
            background:
                radial-gradient(circle at 14% 12%, rgba(125, 211, 252, 0.18), transparent 28%),
                radial-gradient(circle at 82% 4%, rgba(52, 211, 153, 0.13), transparent 30%),
                radial-gradient(circle at 70% 80%, rgba(244, 114, 182, 0.10), transparent 34%),
                linear-gradient(135deg, #08090f 0%, #101119 48%, #0b1512 100%);
        }

        .block-container {
            max-width: 1480px;
            padding-top: 1.25rem;
            padding-bottom: 2rem;
        }

        [data-testid="stSidebar"] {
            background: rgba(10, 12, 20, 0.82);
            border-right: 1px solid rgba(255, 255, 255, 0.10);
            backdrop-filter: blur(20px);
        }

        [data-testid="stSidebar"] * {
            color: var(--text);
        }

        [data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.045);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 1rem 1rem 0.85rem;
            box-shadow: 0 18px 48px rgba(0, 0, 0, 0.32);
        }

        [data-testid="stMetricLabel"] p {
            color: var(--muted);
            font-size: 0.82rem;
            letter-spacing: 0;
        }

        .hero-shell {
            border: 1px solid var(--line);
            background:
                linear-gradient(145deg, rgba(255, 255, 255, 0.10), rgba(255, 255, 255, 0.035)),
                rgba(12, 14, 24, 0.76);
            border-radius: 8px;
            padding: 1.35rem 1.45rem;
            box-shadow: 0 28px 80px rgba(0, 0, 0, 0.38), inset 0 1px 0 rgba(255, 255, 255, 0.18);
            backdrop-filter: blur(22px);
            margin-bottom: 1rem;
        }

        .hero-topline {
            color: var(--green);
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.35rem;
        }

        .hero-title {
            font-size: clamp(2rem, 4vw, 4.2rem);
            line-height: 1.02;
            font-weight: 780;
            letter-spacing: 0;
            margin: 0;
        }

        .hero-copy {
            color: var(--muted);
            font-size: 1rem;
            line-height: 1.58;
            max-width: 900px;
            margin-top: 0.8rem;
        }

        .status-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.75rem;
            margin-top: 1rem;
        }

        .status-card, .glass-card {
            border: 1px solid var(--line);
            background: var(--panel);
            border-radius: 8px;
            padding: 1rem;
            box-shadow: 0 18px 52px rgba(0, 0, 0, 0.30), inset 0 1px 0 rgba(255, 255, 255, 0.12);
            backdrop-filter: blur(20px);
            min-height: 100%;
        }

        .status-label {
            color: var(--muted);
            font-size: 0.76rem;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            font-weight: 700;
        }

        .status-value {
            color: var(--text);
            font-size: 1.65rem;
            line-height: 1.1;
            font-weight: 760;
            margin-top: 0.35rem;
        }

        .status-caption {
            color: var(--muted);
            font-size: 0.82rem;
            margin-top: 0.45rem;
            line-height: 1.35;
        }

        .pill-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin-top: 1rem;
        }

        .pill {
            border: 1px solid rgba(255, 255, 255, 0.12);
            background: rgba(255, 255, 255, 0.055);
            border-radius: 999px;
            color: var(--text);
            padding: 0.35rem 0.65rem;
            font-size: 0.82rem;
        }

        .section-title {
            color: var(--text);
            font-size: 1.08rem;
            font-weight: 740;
            margin: 0 0 0.5rem 0;
        }

        .small-copy {
            color: var(--muted);
            font-size: 0.9rem;
            line-height: 1.5;
        }

        div[data-testid="stDataFrame"] {
            border: 1px solid var(--line);
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 16px 44px rgba(0, 0, 0, 0.24);
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.35rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.10);
        }

        .stTabs [data-baseweb="tab"] {
            background: rgba(255, 255, 255, 0.045);
            border: 1px solid rgba(255, 255, 255, 0.10);
            border-radius: 8px 8px 0 0;
            padding: 0.72rem 0.95rem;
        }

        .stTabs [aria-selected="true"] {
            background: rgba(125, 211, 252, 0.16);
            border-color: rgba(125, 211, 252, 0.38);
        }

        .stButton > button, .stDownloadButton > button, button[kind="primary"] {
            border: 1px solid rgba(125, 211, 252, 0.38);
            background: linear-gradient(135deg, rgba(125, 211, 252, 0.22), rgba(52, 211, 153, 0.16));
            color: var(--text);
            border-radius: 8px;
            box-shadow: 0 12px 28px rgba(0, 0, 0, 0.28);
        }

        hr {
            border-color: rgba(255, 255, 255, 0.10);
        }

        @media (max-width: 900px) {
            .status-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
            .hero-title {
                font-size: 2.2rem;
            }
        }

        @media (max-width: 560px) {
            .status-grid {
                grid-template-columns: 1fr;
            }
            .hero-shell {
                padding: 1rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def plot_layout(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": TEXT_PRIMARY},
        height=height,
        margin={"l": 20, "r": 20, "t": 54, "b": 32},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
        },
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.08)", zerolinecolor="rgba(255,255,255,0.16)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.08)", zerolinecolor="rgba(255,255,255,0.16)")
    return fig


def status_card(label: str, value: str, caption: str, accent: str) -> str:
    return f"""
    <div class="status-card" style="border-top: 2px solid {accent};">
        <div class="status-label">{label}</div>
        <div class="status-value">{value}</div>
        <div class="status-caption">{caption}</div>
    </div>
    """


def section_card(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="glass-card">
            <div class="section-title">{title}</div>
            <div class="small-copy">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_data(n_members: int, seed: int) -> pd.DataFrame:
    return generate_synthetic_members(n_members=n_members, seed=seed)


@st.cache_resource(show_spinner="Training healthcare risk models...")
def load_models(n_members: int, seed: int):
    data = generate_synthetic_members(n_members=n_members, seed=seed)
    return train_models(data, seed=seed)


@st.cache_data(show_spinner=False)
def convert_df(data: pd.DataFrame) -> bytes:
    return data.to_csv(index=False).encode("utf-8")


def filter_scored_data(
    scored: pd.DataFrame,
    regions: list[str],
    plans: list[str],
    providers: list[str],
    tiers: list[str],
    age_range: tuple[int, int],
    probability_threshold: float,
) -> pd.DataFrame:
    filtered = scored.copy()
    filtered["PriorityTier"] = filtered["PriorityTier"].astype(str)
    return filtered[
        filtered["Region"].isin(regions)
        & filtered["PlanID"].isin(plans)
        & filtered["ProviderGroup"].isin(providers)
        & filtered["PriorityTier"].isin(tiers)
        & filtered["Age"].between(age_range[0], age_range[1])
        & (filtered["SuspectProbability"] >= probability_threshold)
    ].copy()


def review_simulation(filtered: pd.DataFrame, capacity_pct: int, review_cost: float) -> dict[str, float]:
    if filtered.empty:
        return {
            "review_count": 0,
            "gross_opportunity": 0.0,
            "review_cost": 0.0,
            "net_value": 0.0,
            "capture_rate": 0.0,
            "lift": 0.0,
            "review_precision": 0.0,
        }

    review_count = max(1, int(np.ceil(len(filtered) * capacity_pct / 100)))
    selected = filtered.sort_values("SuspectProbability", ascending=False).head(review_count)
    gross_opportunity = float((selected["EstimatedOpportunityPMPM"] * selected["MemberMonths"]).sum())
    total_cost = float(review_count * review_cost)
    positives = float(filtered[TARGET].sum())
    baseline = float(filtered[TARGET].mean()) if len(filtered) else 0.0
    review_precision = float(selected[TARGET].mean()) if len(selected) else 0.0
    capture_rate = float(selected[TARGET].sum() / positives) if positives else 0.0
    lift = float(review_precision / baseline) if baseline else 0.0
    return {
        "review_count": review_count,
        "gross_opportunity": gross_opportunity,
        "review_cost": total_cost,
        "net_value": gross_opportunity - total_cost,
        "capture_rate": capture_rate,
        "lift": lift,
        "review_precision": review_precision,
    }


def build_priority_chart(priority: pd.DataFrame) -> go.Figure:
    priority = priority.copy()
    priority["PriorityTier"] = priority["PriorityTier"].astype(str)
    fig = px.bar(
        priority,
        x="PriorityTier",
        y="SuspectRate",
        color="PriorityTier",
        color_discrete_map=TIER_COLORS,
        text=priority["SuspectRate"].map(lambda value: f"{value:.1%}"),
        title="Suspect rate by review tier",
        category_orders={"PriorityTier": ["Monitor", "Medium", "High", "Critical"]},
    )
    fig.update_traces(textposition="outside", marker_line_color="rgba(255,255,255,0.2)", marker_line_width=1)
    fig.update_layout(showlegend=False)
    fig.update_yaxes(tickformat=".0%")
    return plot_layout(fig, 390)


def build_scatter(scored: pd.DataFrame) -> go.Figure:
    sample = scored.sample(min(len(scored), 1800), random_state=7) if len(scored) > 1800 else scored
    fig = px.scatter(
        sample,
        x="RAF_Score",
        y="SuspectProbability",
        color="PriorityTier",
        size="PaidClaimsPMPM",
        hover_data=["MemberID", "ProviderGroup", "PlanID", "ChronicCount", "CareGapCount"],
        color_discrete_map=TIER_COLORS,
        title="Risk score vs model probability",
    )
    fig.update_traces(marker={"opacity": 0.72, "line": {"width": 0.5, "color": "rgba(255,255,255,0.24)"}})
    fig.update_yaxes(tickformat=".0%")
    return plot_layout(fig, 430)


def build_provider_chart(provider: pd.DataFrame) -> go.Figure:
    provider = provider.sort_values("CriticalMembers", ascending=True)
    fig = px.bar(
        provider,
        x="CriticalMembers",
        y="ProviderGroup",
        color="SuspectRate",
        color_continuous_scale=["#334155", ACCENT_BLUE, ACCENT_GOLD, ACCENT_RED],
        title="Provider groups with concentrated critical review volume",
        hover_data=["Members", "AvgRAF", "AvgClaimsPMPM", "EstOpportunityPMPM"],
    )
    fig.update_layout(coloraxis_colorbar={"title": "Suspect rate"})
    return plot_layout(fig, 470)


def build_transfer_chart(transfer: pd.DataFrame) -> go.Figure:
    colors = [ACCENT_GREEN if value > 0 else ACCENT_RED for value in transfer["TransferPMPM"]]
    fig = go.Figure(
        data=[
            go.Bar(
                x=transfer["PlanID"],
                y=transfer["TransferPMPM"],
                marker={"color": colors, "line": {"color": "rgba(255,255,255,0.22)", "width": 1}},
                text=[f"${value:,.0f}" for value in transfer["TransferPMPM"]],
                textposition="outside",
                hovertemplate="%{x}<br>Transfer PMPM: $%{y:,.2f}<extra></extra>",
            )
        ]
    )
    fig.update_layout(title="Illustrative plan transfer PMPM")
    fig.add_hline(y=0, line_color="rgba(255,255,255,0.45)", line_width=1)
    return plot_layout(fig, 410)


def build_feature_chart(importance: pd.DataFrame) -> go.Figure:
    value_col = [col for col in importance.columns if col != "Feature"][0]
    display_data = importance.sort_values(value_col, ascending=True)
    fig = px.bar(
        display_data,
        x=value_col,
        y="Feature",
        orientation="h",
        title="Top model drivers",
        color=value_col,
        color_continuous_scale=["#334155", ACCENT_BLUE, ACCENT_GREEN],
    )
    fig.update_layout(coloraxis_showscale=False)
    return plot_layout(fig, 500)


def build_capacity_curve(filtered: pd.DataFrame, review_cost: float) -> go.Figure:
    rows = []
    for capacity in range(1, 26):
        sim = review_simulation(filtered, capacity, review_cost)
        rows.append({"CapacityPct": capacity, **sim})
    curve = pd.DataFrame(rows)
    fig = px.line(
        curve,
        x="CapacityPct",
        y=["gross_opportunity", "net_value", "review_cost"],
        markers=True,
        title="Review capacity value curve",
        labels={"value": "Dollars", "CapacityPct": "Review capacity (%)", "variable": "Metric"},
        color_discrete_sequence=[ACCENT_GREEN, ACCENT_BLUE, ACCENT_RED],
    )
    return plot_layout(fig, 420)


def scenario_tier(probability: float) -> str:
    if probability >= 0.85:
        return "Critical"
    if probability >= 0.70:
        return "High"
    if probability >= 0.50:
        return "Medium"
    return "Monitor"


def scenario_action(probability: float, care_gaps: int, prior_hcc: int, new_dx: int) -> str:
    if probability >= 0.85 and new_dx:
        return "Immediate chart review and provider documentation follow-up"
    if probability >= 0.70 and care_gaps >= 2:
        return "Prioritize care-gap closure and diagnosis validation"
    if probability >= 0.50 and not prior_hcc:
        return "Queue for documentation review after high-tier members"
    return "Monitor through next monthly refresh"


apply_theme()

with st.sidebar:
    st.markdown("### Simulation")
    n_members = st.slider("Synthetic members", min_value=2000, max_value=10000, value=5000, step=1000)
    seed = st.number_input("Simulation seed", min_value=1, max_value=9999, value=42, step=1)

    st.markdown("---")
    st.markdown("### Operations")
    capacity_pct = st.slider("Monthly review capacity", min_value=1, max_value=25, value=10, step=1, format="%d%%")
    review_cost = st.slider("Chart review cost", min_value=25, max_value=300, value=85, step=5, format="$%d")
    probability_threshold = st.slider(
        "Minimum suspect probability",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.05,
        format="%.2f",
    )


df = load_data(n_members, seed)
model_run = load_models(n_members, seed)
best_model_name = str(model_run.metrics.iloc[0]["Model"])
best_model = model_run.models[best_model_name]
scored = score_members(df, best_model)
scored["PriorityTier"] = scored["PriorityTier"].astype(str)

with st.sidebar:
    st.markdown("---")
    st.markdown("### Segment Filters")
    region_options = sorted(scored["Region"].unique())
    plan_options = sorted(scored["PlanID"].unique())
    provider_options = sorted(scored["ProviderGroup"].unique())
    tier_options = ["Critical", "High", "Medium", "Monitor"]

    regions = st.multiselect("Regions", region_options, default=region_options)
    plans = st.multiselect("Plans", plan_options, default=plan_options)
    providers = st.multiselect("Provider groups", provider_options, default=provider_options)
    tiers = st.multiselect("Priority tiers", tier_options, default=tier_options)
    age_range = st.slider(
        "Age range",
        min_value=int(scored["Age"].min()),
        max_value=int(scored["Age"].max()),
        value=(int(scored["Age"].min()), int(scored["Age"].max())),
    )

if not regions or not plans or not providers or not tiers:
    st.warning("At least one region, plan, provider group, and priority tier must be selected.")
    st.stop()

filtered = filter_scored_data(
    scored=scored,
    regions=regions,
    plans=plans,
    providers=providers,
    tiers=tiers,
    age_range=age_range,
    probability_threshold=probability_threshold,
)

if filtered.empty:
    st.warning("No members match the current segment and probability filters.")
    st.stop()

priority_summary = summarize_priority_queue(filtered)
provider_summary = provider_action_table(filtered)
transfer_summary = plan_transfer_table(filtered)
review_stats = review_simulation(filtered, capacity_pct, float(review_cost))
model_metrics = model_run.metrics.copy()
best_metrics = model_metrics.iloc[0]

st.markdown(
    f"""
    <div class="hero-shell">
        <div class="hero-topline">Healthcare payer analytics command center</div>
        <h1 class="hero-title">Risk adjustment review queue</h1>
        <div class="hero-copy">
            Prioritize suspect members, provider groups, and plan-level financial exposure using
            a synthetic HIPAA-safe payer dataset and an explainable machine learning workflow.
        </div>
        <div class="pill-row">
            <div class="pill">Best model: {best_model_name}</div>
            <div class="pill">ROC AUC: {best_metrics['ROC_AUC']:.3f}</div>
            <div class="pill">Top-decile lift: {best_metrics['Top_Decile_Lift']:.2f}x</div>
            <div class="pill">Filtered members: {len(filtered):,}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="status-grid">
        {status_card("Members in scope", f"{len(filtered):,}", "Filtered review population", ACCENT_BLUE)}
        {status_card("Suspect rate", f"{filtered[TARGET].mean():.1%}", "Baseline inside current filters", ACCENT_GREEN)}
        {status_card("Review volume", f"{review_stats['review_count']:,.0f}", f"{capacity_pct}% monthly capacity", ACCENT_GOLD)}
        {status_card("Net review value", f"${review_stats['net_value']:,.0f}", "Gross opportunity less review cost", ACCENT_PINK)}
    </div>
    """,
    unsafe_allow_html=True,
)

tabs = st.tabs(
    [
        "Command Center",
        "Review Queue",
        "Provider Strategy",
        "Financial Simulator",
        "Model Explainability",
        "Member Scenario",
    ]
)

with tabs[0]:
    left, right = st.columns([1.12, 0.88], gap="large")
    with left:
        st.plotly_chart(build_priority_chart(priority_summary), width="stretch")
    with right:
        section_card(
            "Operational focus",
            (
                f"The current filters produce a {filtered[TARGET].mean():.1%} suspect rate. "
                f"Reviewing the top {capacity_pct}% creates {review_stats['lift']:.2f}x lift "
                f"and captures {review_stats['capture_rate']:.1%} of known suspect members in scope."
            ),
        )
        st.markdown("")
        c1, c2 = st.columns(2)
        c1.metric("Gross opportunity", f"${review_stats['gross_opportunity']:,.0f}")
        c2.metric("Review cost", f"${review_stats['review_cost']:,.0f}")
        c3, c4 = st.columns(2)
        c3.metric("Review precision", f"{review_stats['review_precision']:.1%}")
        c4.metric("Queue lift", f"{review_stats['lift']:.2f}x")

    st.plotly_chart(build_scatter(filtered), width="stretch")

with tabs[1]:
    review_count = int(review_stats["review_count"])
    review_queue = filtered.sort_values("SuspectProbability", ascending=False).head(review_count)
    display_cols = [
        "RiskRank",
        "MemberID",
        "PriorityTier",
        "SuspectProbabilityPct",
        "ProviderGroup",
        "Region",
        "PlanID",
        "Age",
        "ChronicCount",
        "CareGapCount",
        "RAF_Score",
        "PaidClaimsPMPM",
        "EstimatedOpportunityPMPM",
    ]
    review_queue = review_queue.assign(SuspectProbabilityPct=review_queue["SuspectProbability"] * 100)
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Queue members", f"{len(review_queue):,}")
    q2.metric("Queue suspect rate", f"{review_queue[TARGET].mean():.1%}")
    q3.metric("Avg probability", f"{review_queue['SuspectProbability'].mean():.1%}")
    q4.metric("Avg RAF", f"{review_queue['RAF_Score'].mean():.3f}")

    st.dataframe(
        review_queue[display_cols],
        width="stretch",
        hide_index=True,
        column_config={
            "SuspectProbabilityPct": st.column_config.ProgressColumn(
                "Suspect Probability",
                format="%.1f%%",
                min_value=0,
                max_value=100,
            ),
            "PaidClaimsPMPM": st.column_config.NumberColumn("Claims PMPM", format="$%.0f"),
            "EstimatedOpportunityPMPM": st.column_config.NumberColumn("Opportunity PMPM", format="$%.2f"),
        },
    )

    st.download_button(
        "Download current review queue",
        data=convert_df(review_queue[display_cols]),
        file_name="risk_adjustment_review_queue.csv",
        mime="text/csv",
    )

with tabs[2]:
    p_left, p_right = st.columns([1.2, 0.8], gap="large")
    with p_left:
        st.plotly_chart(build_provider_chart(provider_summary), width="stretch")
    with p_right:
        top_provider = provider_summary.iloc[0]
        section_card(
            "Provider spotlight",
            (
                f"{top_provider['ProviderGroup']} has {int(top_provider['CriticalMembers']):,} critical members, "
                f"a {top_provider['SuspectRate']:.1%} suspect rate, and average claims PMPM of "
                f"${top_provider['AvgClaimsPMPM']:,.0f}."
            ),
        )
        selected_provider = st.selectbox("Provider deep dive", provider_summary["ProviderGroup"].tolist())
        provider_members = filtered[filtered["ProviderGroup"] == selected_provider].sort_values(
            "SuspectProbability", ascending=False
        )
        st.metric("Provider members", f"{len(provider_members):,}")
        st.metric("Provider suspect rate", f"{provider_members[TARGET].mean():.1%}")

    st.dataframe(provider_summary.round(3), width="stretch", hide_index=True)

with tabs[3]:
    f_left, f_right = st.columns([1, 1], gap="large")
    with f_left:
        st.plotly_chart(build_transfer_chart(transfer_summary), width="stretch")
    with f_right:
        st.plotly_chart(build_capacity_curve(filtered, float(review_cost)), width="stretch")

    st.dataframe(transfer_summary.round(3), width="stretch", hide_index=True)
    section_card(
        "Finance interpretation",
        (
            "Transfer results are illustrative. The app uses risk scores, actuarial value, member months, "
            "and a premium assumption to show how risk concentration can create plan-level receivable "
            "or payable exposure."
        ),
    )

with tabs[4]:
    m1, m2 = st.columns([0.9, 1.1], gap="large")
    with m1:
        metrics_display = model_metrics.copy()
        for col in metrics_display.columns:
            if col != "Model":
                metrics_display[col] = metrics_display[col].round(3)
        st.dataframe(metrics_display, width="stretch", hide_index=True)

        report, confusion = classification_summary(best_model, model_run.x_test, model_run.y_test)
        st.markdown("#### Classification report")
        st.dataframe(report.round(3), width="stretch")

        fig_confusion = px.imshow(
            confusion,
            text_auto=True,
            color_continuous_scale=["#172033", ACCENT_BLUE, ACCENT_GREEN],
            labels={"x": "Predicted", "y": "Actual", "color": "Members"},
            title="Confusion matrix at 0.50 threshold",
        )
        st.plotly_chart(plot_layout(fig_confusion, 350), width="stretch")

    with m2:
        st.plotly_chart(build_feature_chart(feature_importance(best_model, top_n=15)), width="stretch")
        section_card(
            "Model governance note",
            (
                "This prototype uses synthetic data and should be validated on real claims, encounter, "
                "coding, and quality data before production use. In a real payer setting, monitoring "
                "should include calibration, subgroup performance, drift, and review outcome feedback."
            ),
        )

with tabs[5]:
    st.markdown("#### Score a single member scenario")
    with st.form("member_scenario"):
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.slider("Age", 18, 90, 67)
            gender = st.selectbox("Gender", ["F", "M"])
            region = st.selectbox("Region", sorted(scored["Region"].unique()), index=2)
            plan = st.selectbox("Plan", sorted(scored["PlanID"].unique()), index=1)
        with c2:
            provider = st.selectbox("Provider group", sorted(scored["ProviderGroup"].unique()), index=1)
            visits = st.slider("Visits", 0, 32, 9)
            er_visits = st.slider("ER visits", 0, 9, 1)
            admits = st.slider("Inpatient admits", 0, 6, 1)
        with c3:
            rx_count = st.slider("RX count", 0, 35, 8)
            chronic_count = st.slider("Chronic count", 0, 5, 3)
            prior_hcc = st.selectbox("Prior HCC documented", [0, 1], index=0)
            new_dx = st.selectbox("New diagnosis signal", [0, 1], index=1)
            care_gaps = st.slider("Care gaps", 0, 9, 3)
            raf = st.slider("RAF score", 0.12, 4.25, 1.25, step=0.01)
            claims = st.slider("Claims PMPM", 80, 5000, 1450, step=25)
            member_months = st.selectbox("Member months", [3, 6, 9, 12], index=3)

        st.form_submit_button("Score member")

    av_lookup = {"PlanA_Silver": 0.70, "PlanB_Gold": 0.80, "PlanC_Bronze": 0.60}
    scenario = pd.DataFrame(
        [
            {
                "Age": age,
                "Visits": visits,
                "ER_Visits": er_visits,
                "Inpatient_Admits": admits,
                "RX_Count": rx_count,
                "ChronicCount": chronic_count,
                "Prior_HCC_Flag": prior_hcc,
                "New_Diagnosis_Flag": new_dx,
                "CareGapCount": care_gaps,
                "RAF_Score": raf,
                "PaidClaimsPMPM": claims,
                "MemberMonths": member_months,
                "Region": region,
                "Gender": gender,
                "PlanID": plan,
                "ProviderGroup": provider,
            }
        ],
        columns=MODEL_FEATURES,
    )

    probability = float(best_model.predict_proba(scenario)[0, 1])
    tier = scenario_tier(probability)
    action = scenario_action(probability, care_gaps, prior_hcc, new_dx)
    opportunity = probability * raf * av_lookup[plan] * 125

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Suspect probability", f"{probability:.1%}")
    s2.metric("Recommended tier", tier)
    s3.metric("Opportunity PMPM", f"${opportunity:,.2f}")
    s4.metric("Expected member value", f"${opportunity * member_months:,.0f}")

    section_card("Recommended action", action)

    st.dataframe(
        scenario.assign(
            SuspectProbability=probability,
            PriorityTier=tier,
            EstimatedOpportunityPMPM=opportunity,
        ),
        width="stretch",
        hide_index=True,
    )
