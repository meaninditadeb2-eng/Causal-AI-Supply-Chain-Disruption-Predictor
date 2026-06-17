"""
app.py — Main entry point for Causal AI Supply Chain Dashboard
============================================================
Run with:  streamlit run app.py
"""

import streamlit as st

st.set_page_config(
    page_title="Causal AI — Supply Chain",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1B3A6B 0%, #2E75B6 100%);
}
[data-testid="stSidebar"] .stRadio label span,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] small,
[data-testid="stSidebar"] div { color: #E8EEF8 !important; }

/* ── Metric cards ── */
div[data-testid="metric-container"] {
    background: #FFFFFF;
    border: 1px solid #D5DBE8;
    border-radius: 10px;
    padding: 14px 18px;
    border-left: 5px solid #2E75B6;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #1B3A6B, #2E75B6);
    color: white !important;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 0.5rem 1.8rem;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #2E75B6, #1D9E75) !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}

/* ── Cards ── */
.card {
    background: white;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    border: 1px solid #E0E4ED;
    border-left: 5px solid #2E75B6;
    margin-bottom: 0.9rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.card-teal  { border-left-color: #1D9E75 !important; }
.card-red   { border-left-color: #C0392B !important; }
.card-amber { border-left-color: #E67E22 !important; }

/* ── Risk colours ── */
.risk-high   { color: #C0392B; font-weight: 800; }
.risk-medium { color: #E67E22; font-weight: 800; }
.risk-low    { color: #1D9E75; font-weight: 800; }

/* ── Info / warning boxes ── */
.info-box {
    background: #EAF3FE;
    border-left: 4px solid #2E75B6;
    border-radius: 6px;
    padding: 0.8rem 1rem;
    margin: 0.6rem 0;
    font-size: 0.88rem;
    color: #1B3A6B;
}
.warn-box {
    background: #FFF8E6;
    border-left: 4px solid #E67E22;
    border-radius: 6px;
    padding: 0.8rem 1rem;
    margin: 0.6rem 0;
    font-size: 0.88rem;
    color: #7D4E00;
}
.success-box {
    background: #EAF7F2;
    border-left: 4px solid #1D9E75;
    border-radius: 6px;
    padding: 0.8rem 1rem;
    margin: 0.6rem 0;
    font-size: 0.88rem;
    color: #0F4C35;
}

/* ── Section headers ── */
.section-title {
    font-size: 1.55rem;
    font-weight: 700;
    color: #1B3A6B;
    padding-bottom: 0.4rem;
    border-bottom: 3px solid #2E75B6;
    margin-bottom: 1.3rem;
}

/* ── Table styling ── */
.dataframe { font-size: 0.85rem !important; }
.dataframe thead th {
    background: #1B3A6B !important;
    color: white !important;
}

/* ── Hide Streamlit branding ── */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔗 Causal AI")
    st.markdown("### Supply Chain Disruption")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        options=[
            "🏠  Overview",
            "🎯  Disruption Predictor",
            "🔄  Counterfactual Simulator",
            "🕸️  Causal Graph",
            "📊  Feature Importance",
            "📈  Model Results",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("""
**Dataset**
- DataCo: 180,519 orders
- Disruption rate: 54.83%
- 8 causal features

**Best Model**
- XGBoost (causal features)
- ROC-AUC: 0.9997
- F1 Score: 0.9939

**Causal Discovery**
- Precision: 1.000
- Recall: 0.692
- SHD: 4
""")
    st.markdown("---")
    st.markdown(
        "<small>Built with Streamlit · Plotly · scikit-learn · XGBoost</small>",
        unsafe_allow_html=True,
    )

# ── Page routing ───────────────────────────────────────────────────────────
if "Overview" in page:
    from pages import p1_overview as pg
elif "Predictor" in page:
    from pages import p2_predictor as pg
elif "Counterfactual" in page:
    from pages import p3_counterfactual as pg
elif "Causal Graph" in page:
    from pages import p4_causal_graph as pg
elif "Feature Importance" in page:
    from pages import p5_feature_importance as pg
elif "Model Results" in page:
    from pages import p6_model_results as pg

pg.show()
