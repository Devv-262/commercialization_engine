"""
streamlit_app.py — Presentation layer / dashboard.

Shows portfolio overview, concept details with SHAP charts, 
portfolio-level visuals, and auto-generated executive summary.
"""

import json
import os
import sys

# Add project root to path so 'src' can be imported regardless of where streamlit is run from
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import shap
import matplotlib.pyplot as plt

from src.config import (
    DB_PATH,
    OUTCOME_COLORS,
    OUTCOME_MVP_BUILD,
    OUTCOME_CUSTOMER_PILOT,
    OUTCOME_REUSABLE_ASSET,
    OUTCOME_INCUBATE,
    OUTCOME_ARCHIVE,
)
from src.data_pipeline.clean_validate import load_and_clean_all
from src.data_pipeline.load_to_db import run_pipeline
from src.features.feature_engineering import build_feature_table
from src.models.clustering import fit_clusters
from src.models.readiness_model import (
    get_feature_importance_df,
    predict_readiness_scores,
    train_readiness_model,
)
from src.models.confidence import compute_confidence_scores
from src.insight.shap_explainer import compute_shap_values, get_top_features_per_concept
from src.insight.decision_engine import run_decision_engine
from src.insight.narrative_generator import generate_all_narratives, generate_executive_summary

st.set_page_config(
    page_title="Commercialization Decision Engine",
    layout="wide",
)


@st.cache_data
def run_full_backend() -> dict:
    """Run the entire data and ML pipeline and cache the results in memory."""
    # Ensure data exists (in a real app, this wouldn't overwrite on every reload)
    run_pipeline(reset=True)
    
    data = load_and_clean_all(DB_PATH)
    features = build_feature_table(
        data["concepts"], data["demo_signals"], data["usage_signals"], data["commercial_signals"]
    )
    clustered = fit_clusters(features)
    
    model, x_matrix, x_df = train_readiness_model(clustered)
    readiness_scores = predict_readiness_scores(model, x_matrix, x_df.index.tolist())
    
    confidence_scores = compute_confidence_scores(
        data["demo_signals"],
        data["usage_signals"],
        data["commercial_signals"],
        clustered,
        x_df.index.tolist(),
    )
    
    shap_vals = compute_shap_values(model, x_matrix)
    top_features = get_top_features_per_concept(shap_vals, x_df.index.tolist())
    global_importance = get_feature_importance_df(model)
    
    outcome_df = run_decision_engine(
        clustered,
        readiness_scores,
        confidence_scores,
        data["usage_signals"],
        data["demo_signals"],
        data["commercial_signals"],
    )
    
    narratives = generate_all_narratives(outcome_df, top_features, data["concepts"])
    exec_summary = generate_executive_summary(outcome_df, data["concepts"])
    
    # Save the executive summary to file as requested
    os.makedirs(os.path.dirname("reports/executive_summary.md"), exist_ok=True)
    with open("reports/executive_summary.md", "w") as f:
        f.write(exec_summary)
        
    return {
        "concepts": data["concepts"],
        "features": clustered,
        "outcomes": outcome_df,
        "narratives": narratives,
        "shap_values": shap_vals,
        "top_features": top_features,
        "global_importance": global_importance,
        "exec_summary": exec_summary,
        "x_df": x_df,
    }


def render_portfolio_overview(state: dict):
    st.header("Portfolio Overview")
    
    outcomes = state["outcomes"].copy()
    concepts = state["concepts"].set_index("concept_id")
    
    # Join name for display
    display_df = outcomes.join(concepts[["concept_name"]], on="concept_id")
    display_df = display_df.sort_values("readiness_score", ascending=False)
    
    # Top-level KPIs
    total_concepts = len(display_df)
    actionable = len(display_df[display_df["recommended_outcome"].isin([OUTCOME_MVP_BUILD, OUTCOME_CUSTOMER_PILOT])])
    mean_readiness = display_df["readiness_score"].mean()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Concepts Evaluated", total_concepts)
    col2.metric("Actionable (Build or Pilot)", actionable)
    col3.metric("Average Readiness Score", f"{mean_readiness:.1f} / 100")
    
    st.markdown("---")
    st.subheader("Ranked Concept Outcomes")
    
    # Prepare a clean dataframe for the data grid
    grid_df = display_df[[
        "concept_name", "recommended_outcome", "readiness_score", "confidence_score", "delivery_complexity"
    ]].copy()
    
    st.dataframe(
        grid_df,
        column_config={
            "concept_name": st.column_config.TextColumn("Concept Name", width="medium"),
            "recommended_outcome": st.column_config.TextColumn("Recommended Outcome", width="medium"),
            "readiness_score": st.column_config.ProgressColumn(
                "Readiness Score",
                help="Readiness for commercialization (0-100)",
                format="%.1f",
                min_value=0,
                max_value=100,
            ),
            "confidence_score": st.column_config.NumberColumn(
                "Confidence",
                help="Statistical confidence (0-100)",
                format="%.1f",
            ),
            "delivery_complexity": st.column_config.NumberColumn(
                "Complexity",
                help="Delivery Complexity (0-1)",
                format="%.2f",
            ),
        },
        hide_index=True,
        use_container_width=True,
        height=600,
    )


def render_concept_detail(state: dict):
    st.header("Concept Detail & Explainability")
    
    concepts = state["concepts"].set_index("concept_id")
    outcomes = state["outcomes"].set_index("concept_id")
    
    concept_id = st.selectbox(
        "Select a concept to analyze:",
        options=outcomes.index.tolist(),
        format_func=lambda x: f"{concepts.loc[x, 'concept_name']} ({x})"
    )
    
    row = outcomes.loc[concept_id]
    name = concepts.loc[concept_id, "concept_name"]
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Readiness Score", f"{row['readiness_score']:.1f}/100")
    col2.metric("Confidence Score", f"{row['confidence_score']:.1f}/100")
    
    color = OUTCOME_COLORS.get(row['recommended_outcome'], "#7f8c8d")
    col3.markdown(
        f"Outcome: <br/><div style='background-color: {color}; color: white; padding: 8px; "
        f"border-radius: 4px; text-align: center; font-weight: bold; font-size: 1.2em'>"
        f"{row['recommended_outcome']}</div>",
        unsafe_allow_html=True
    )
    
    st.subheader("AI Narrative")
    st.info(state["narratives"][concept_id])
    
    # Add Technicalities Section
    features = state["features"].set_index("concept_id")
    cluster_id = int(features.loc[concept_id, "cluster_id"]) if "cluster_id" in features.columns else "N/A"
    
    with st.expander("How was this generated? (ML & AI Technicalities)"):
        st.markdown(f"""
        **1. Unsupervised Learning (KMeans Clustering):**  
        This concept was assigned to **Cluster {cluster_id}**. The engine grouped this concept with others exhibiting similar demand behavior to calculate its segment repeatability.
        
        **2. Supervised Learning (Gradient Boosting Regressor):**  
        The Readiness Score of **{row['readiness_score']:.1f}** was predicted by a Gradient Boosting tree. It analyzed non-linear interactions between 8 engineered features (e.g., if engagement is high but decision-maker presence is 0, the score drops non-linearly).
        
        **3. AI Explainability (SHAP):**  
        The narrative above and the waterfall chart below are generated dynamically using `shap.TreeExplainer`. This prevents LLM hallucination by mathematically proving exactly which features pushed the model's score up or down.
        """)
        
    st.subheader("SHAP Evidence Waterfall")
    st.markdown("Features pushing the readiness score up (red) vs down (blue).")
    
    # SHAP plotting
    idx = state["x_df"].index.get_loc(concept_id)
    shap_vals = state["shap_values"]
    
    fig, ax = plt.subplots(figsize=(10, 4))
    shap.waterfall_plot(shap.Explanation(
        values=shap_vals.values[idx],
        base_values=shap_vals.base_values[idx],
        data=shap_vals.data[idx],
        feature_names=shap_vals.feature_names
    ), show=False)
    st.pyplot(fig)
    plt.close()


def render_portfolio_visuals(state: dict):
    st.header("Portfolio Visuals")
    
    outcomes = state["outcomes"]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Outcome Distribution")
        dist = outcomes["recommended_outcome"].value_counts().reset_index()
        dist.columns = ["Outcome", "Count"]
        
        fig1 = px.bar(
            dist, x="Outcome", y="Count", 
            color="Outcome",
            color_discrete_map=OUTCOME_COLORS,
            title="Concepts by Recommended Outcome",
            template="plotly_dark"
        )
        st.plotly_chart(fig1, use_container_width=True)
        
    with col2:
        st.subheader("Global Feature Importance (SHAP)")
        imp = state["global_importance"].head(6)
        fig2 = px.bar(
            imp, x="importance", y="feature",
            orientation="h",
            title="Most Influential Signals Across Portfolio",
            template="plotly_dark"
        )
        fig2.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig2, use_container_width=True)
        
    st.subheader("Readiness vs. Confidence Landscape")
    
    # Scatter plot
    display_df = outcomes.join(state["concepts"].set_index("concept_id")[["concept_name"]], on="concept_id")
    fig3 = px.scatter(
        display_df,
        x="confidence_score",
        y="readiness_score",
        color="recommended_outcome",
        color_discrete_map=OUTCOME_COLORS,
        hover_name="concept_name",
        size_max=15,
        title="Portfolio Matrix (Upper Right = MVP Build)",
        template="plotly_dark"
    )
    # Draw quadrants
    fig3.add_hline(y=40, line_dash="dash", line_color="gray")
    fig3.add_vline(x=60, line_dash="dash", line_color="gray")
    st.plotly_chart(fig3, use_container_width=True)


def render_executive_summary(state: dict):
    st.header("Executive Summary")
    st.markdown("Stakeholder-ready synthesis generated deterministically from model outputs.")
    
    st.markdown("---")
    st.markdown(state["exec_summary"])
    st.markdown("---")
    
    with open("reports/executive_summary.md", "rb") as file:
        st.download_button(
            label="Download Markdown Report",
            data=file,
            file_name="executive_summary.md",
            mime="text/markdown",
        )


def main():
    with st.spinner("Initializing pipeline and running models..."):
        state = run_full_backend()
        
    # --- Sidebar Navigation ---
    with st.sidebar:
        st.markdown("<h2 style='color:#F8FAFC; margin-bottom: 0px;'>Decision Engine</h2>", unsafe_allow_html=True)
        st.markdown("<p style='color:#94A3B8; margin-top: 0px; font-size: 0.9em;'>AI Commercialization Portfolio</p>", unsafe_allow_html=True)
        st.markdown("---")
        
        selection = st.radio(
            "Navigation",
            ["Portfolio Overview", "Concept Detail", "Portfolio Visuals", "Executive Summary"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown("<small style='color:#64748B;'>Engine version 1.0.0<br/>Data updated just now</small>", unsafe_allow_html=True)

    # --- Top Navbar (simulated) ---
    st.markdown("<h1 style='color:#F8FAFC;'>Commercialization Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    # --- Route to specific view ---
    if selection == "Portfolio Overview":
        render_portfolio_overview(state)
    elif selection == "Concept Detail":
        render_concept_detail(state)
    elif selection == "Portfolio Visuals":
        render_portfolio_visuals(state)
    elif selection == "Executive Summary":
        render_executive_summary(state)


if __name__ == "__main__":
    main()
