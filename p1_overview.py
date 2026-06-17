"""Page 1 — Project Overview"""
import streamlit as st
import plotly.graph_objects as go
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import load_models, generate_training_data, SHIPPING_MODES


def show():
    st.markdown('<p class="section-title">🏠 Project Overview</p>', unsafe_allow_html=True)

    # ── Hero metrics ───────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("📦 Real Orders",     "180,519",  "DataCo dataset")
    c2.metric("🎯 Best AUC",        "0.9997",   "XGBoost causal")
    c3.metric("⚡ Disruption Rate", "54.83%",   "Near-balanced")
    c4.metric("🕸️ Causal Precision","1.000",    "Zero false edges")
    c5.metric("💡 Risk Reducible",  "~45%",     "Combined intervention")

    st.markdown("---")

    # ── What this system does ──────────────────────────────────────────────
    col1, col2 = st.columns([1.1, 1])

    with col1:
        st.markdown("### What This System Does")
        for icon, title, desc in [
            ("🔮", "Predict",
             "Real-time disruption probability for any order using 8 causal features."),
            ("🔄", "Simulate",
             "Counterfactual what-if analysis — reduce transport delays by 30% and see the impact instantly."),
            ("🕸️", "Explain Causally",
             "Interactive causal graph showing which variables truly cause disruptions — not just correlate."),
            ("📊", "Explain Features",
             "Fast feature importance charts showing what drives predictions globally across all orders."),
        ]:
            st.markdown(f"""
            <div class="card">
            <strong>{icon} {title}</strong><br>
            <span style="color:#555; font-size:0.92rem">{desc}</span>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown("### Key Research Findings")
        findings = [
            ("🔴", "Triple Convergence on TransportDelay",
             "RF importance, XGBoost importance, and DoWhy ATE all rank it #1 independently."),
            ("🟡", "First Class shipping = 95.3% disruption",
             "Causal reasoning: it's chosen during stress — symptom, not cause."),
            ("🟢", "Combined interventions cut risk by 45%",
             "Transport + Order + Stock risk addressed together: 54.8% → ~30%."),
            ("🔵", "Causal discovery precision = 1.000",
             "Every discovered causal edge was genuine — zero false claims."),
            ("⚪", "Geography is NOT a causal driver",
             "All 5 markets show ~55% disruption — the problem is operational."),
        ]
        for emoji, title, detail in findings:
            st.markdown(f"""
            <div class="card" style="padding:0.8rem 1.1rem; margin-bottom:0.6rem">
            <strong>{emoji} {title}</strong><br>
            <span style="color:#666; font-size:0.85rem">{detail}</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Architecture pipeline ──────────────────────────────────────────────
    st.markdown("### End-to-End Pipeline")
    stages  = ["Raw Data", "Feature Eng.", "Causal Discovery", "SCM (DoWhy)",
               "ML Models", "Feature Imp.", "Counterfactuals", "Dashboard"]
    colors  = ["#1B3A6B","#2E75B6","#2E75B6","#1D7A6B",
               "#1D7A6B","#8E44AD","#8E44AD","#C0392B"]

    fig = go.Figure()
    for i, (stage, color) in enumerate(zip(stages, colors)):
        fig.add_trace(go.Scatter(
            x=[i], y=[0], mode="markers+text",
            marker=dict(size=50, color=color, symbol="circle",
                        line=dict(color="white", width=2)),
            text=[stage], textposition="top center",
            textfont=dict(size=9.5, color="#333"),
            showlegend=False, hoverinfo="skip",
        ))
        if i < len(stages) - 1:
            fig.add_annotation(
                x=i+0.52, y=0, ax=i+0.12, ay=0,
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=2, arrowwidth=2, arrowcolor="#2E75B6",
            )
    fig.update_layout(
        height=165, margin=dict(l=10, r=10, t=55, b=5),
        xaxis=dict(visible=False, range=[-0.5, 7.5]),
        yaxis=dict(visible=False, range=[-0.6, 0.9]),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Quick data preview ─────────────────────────────────────────────────
    st.markdown("### Training Data Snapshot")
    data = load_models()
    df   = data["df"]

    col1, col2 = st.columns(2)
    with col1:
        counts = df["Disruption"].value_counts()
        fig = go.Figure(go.Bar(
            x=["On Time", "Disrupted"], y=counts.values,
            marker_color=["#1D9E75", "#C0392B"],
            text=[f"{v:,}<br>({v/len(df):.1%})" for v in counts.values],
            textposition="outside",
        ))
        fig.update_layout(
            title="Target Distribution", height=300,
            margin=dict(l=10, r=10, t=40, b=10),
            plot_bgcolor="white", paper_bgcolor="white",
            yaxis=dict(gridcolor="#F0F0F0"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        ship_r = df.groupby("ShippingRisk")["Disruption"].mean()
        ship_r.index = [SHIPPING_MODES.get(int(i), str(i)) for i in ship_r.index]
        fig = go.Figure(go.Bar(
            x=ship_r.values, y=ship_r.index, orientation="h",
            marker_color=["#1D9E75" if v < 0.45 else "#E67E22" if v < 0.65
                          else "#C0392B" for v in ship_r.values],
            text=[f"{v:.1%}" for v in ship_r.values],
            textposition="outside",
        ))
        fig.update_layout(
            title="Disruption Rate by Shipping Mode", height=300,
            margin=dict(l=10, r=60, t=40, b=10),
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(tickformat=".0%", gridcolor="#F0F0F0"),
        )
        st.plotly_chart(fig, use_container_width=True)
