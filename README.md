# AI/ML Commercialization Decision Engine

A working Python prototype that analyzes mock customer demo, feedback, and usage data for multiple early-stage AI product concepts, ranks them, and recommends one of five commercial outcomes for each, with explainable evidence.

## Architecture

The core flow is split into two distinct layers to prevent logic entanglement:

```text
Customer signals -> ML pattern detection -> AI explanation -> Commercial decision
```

### 1. ML Layer (`src/models/`)
- **KMeans clustering**: Segments concepts to find multi-market demand patterns (the primary signal for a "Reusable Asset").
- **GradientBoostingRegressor**: Trained on a bootstrapped synthetic label (to simulate lack of historical ground-truth) to capture non-linear interactions between demand, value, and feasibility.
- **Statistical Confidence**: Not an ML model; scores trust based on sample size, variance, and cluster tightness.

### 2. AI & Insight Layer (`src/insight/`)
- **SHAP Explainability**: Uses `TreeExplainer` on the GBR model to deterministically extract top positive/negative feature contributions per concept.
- **Narrative Generation**: Deterministic, template-based natural language generation (no external LLM APIs). Converts SHAP values into business-readable paragraphs.
- **Decision Engine**: A decision matrix mapping scores into 5 distinct outcomes.

## The 5 Commercial Outcomes

| Outcome | Trigger Condition |
|---|---|
| **Archive** | Low readiness, OR high delivery complexity relative to value, OR negative-trending engagement. Evaluated first. |
| **MVP Build** | High readiness, high confidence, low-to-moderate delivery complexity. |
| **Reusable Asset** | High readiness, high cluster_repeatability (demand spans multiple segments). |
| **Customer Pilot** | High readiness + high complexity (too big for MVP), OR moderate readiness + strong named-customer urgency/budget. |
| **Incubate** | Moderate readiness + low confidence (default fallback). |

## Running the Prototype

1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

2. Generate data and run the ML pipeline:
   ```bash
   python -m src.data_pipeline.load_to_db
   ```

3. View the dashboard:
   ```bash
   streamlit run app/streamlit_app.py
   ```
