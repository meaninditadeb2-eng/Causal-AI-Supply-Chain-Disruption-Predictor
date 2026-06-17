"""Page 3 — Counterfactual Simulator"""
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import load_models, CAUSAL_FEATURES, predict_proba


def apply_intervention(df_base, intervention: dict):
    df = df_base.copy()
    for col, fn in intervention.items():
        if col in df.columns:
            df[col] = fn(df[col]) if callable(fn) else fn
    return df


def show():
    st.markdown('<p class="section-title">🔄 Counterfactual Simulator</p>',
                unsafe_allow_html=True)
    st.markdown(
        "Simulate the effect of business interventions across **all orders**. "
        "Adjust the levers to see the predicted disruption rate change in real time."
    )

    data     = load_models()
    rf       = data["rf"]
    df       = data["df"]
    X_base   = df[CAUSAL_FEATURES].fillna(0)
    baseline = rf.predict_proba(X_base)[:, 1].mean()

    # ── Intervention controls ──────────────────────────────────────────────
    st.markdown("### ⚙️  Define Your Intervention")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Logistics**")
        transport_pct = st.slider(
            "Reduce TransportDelay by", 0, 80, 0, 5, format="%d%%",
            help="Improve carrier SLAs, better route planning",
        )
        upgrade_shipping = st.checkbox(
            "Upgrade all shipments by 1 tier",
            help="Standard → Second Class, First Class → Same Day",
        )

    with col2:
        st.markdown("**Order Quality**")
        remove_order_risk = st.checkbox(
            "Eliminate all OrderRisk events",
            help="Better fraud detection, stronger order validation",
        )
        demand_pct = st.slider(
            "Reduce DemandSpike by", 0, 50, 0, 5, format="%d%%",
            help="Better demand forecasting, pre-positioning inventory",
        )

    with col3:
        st.markdown("**Inventory**")
        remove_stock_risk = st.checkbox(
            "Eliminate all StockRisk events",
            help="Safety stock policies, better supplier planning",
        )
        inv_pct = st.slider(
            "Reduce InventoryPressure by", 0, 50, 0, 5, format="%d%%",
            help="Improve supply-demand alignment",
        )

    # ── Apply intervention ─────────────────────────────────────────────────
    interv = {}
    if transport_pct > 0:
        interv["TransportDelay"] = lambda x: x * (1 - transport_pct / 100)
    if upgrade_shipping:
        interv["ShippingRisk"] = lambda x: (x - 1).clip(lower=0)
    if remove_order_risk:
        interv["OrderRisk"] = 0
    if demand_pct > 0:
        interv["DemandSpike"] = lambda x: x * (1 - demand_pct / 100)
    if remove_stock_risk:
        interv["StockRisk"] = 0
    if inv_pct > 0:
        interv["InventoryPressure"] = lambda x: x * (1 - inv_pct / 100)

    X_cf       = apply_intervention(X_base, interv)
    cf_rate    = rf.predict_proba(X_cf)[:, 1].mean()
    reduction  = baseline - cf_rate
    rel_pct    = reduction / baseline * 100 if baseline > 0 else 0
    orders_saved = int(reduction * 180519)

    st.markdown("---")

    # ── KPI row ────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📊 Baseline Rate",       f"{baseline:.1%}")
    c2.metric("✅ After Intervention",  f"{cf_rate:.1%}",
              f"↓ {reduction:.1%}")
    c3.metric("📉 Relative Reduction",  f"{rel_pct:.1f}%")
    c4.metric("💡 Orders Saved / yr",   f"{orders_saved:,}")

    # ── Before / After gauges ──────────────────────────────────────────────
    col1, col2 = st.columns(2)
    for col, rate, title, color in [
        (col1, baseline, "Before Intervention", "#C0392B"),
        (col2, cf_rate,  "After Intervention",  "#1D9E75"),
    ]:
        with col:
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=rate * 100,
                delta={"reference": baseline * 100, "valueformat": ".1f",
                       "font": {"size": 18}},
                number={"suffix": "%", "font": {"size": 42, "color": color}},
                title={"text": title, "font": {"size": 16}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar":  {"color": color, "thickness": 0.35},
                    "steps": [
                        {"range": [0,  40], "color": "#E8F5E9"},
                        {"range": [40, 70], "color": "#FFF8E6"},
                        {"range": [70,100], "color": "#FFEBEE"},
                    ],
                },
            ))
            fig.update_layout(height=260,
                              margin=dict(l=15, r=15, t=40, b=10),
                              paper_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)

    # ── Pre-built scenario comparison ──────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📊 Pre-Built Intervention Ranking")
    st.markdown(
        "Compare 9 standard scenarios to find the highest-impact intervention strategy."
    )

    SCENARIOS = {
        "Baseline":                     {},
        "Reduce Transport –30%":        {"TransportDelay": lambda x: x * 0.70},
        "Reduce Transport –50%":        {"TransportDelay": lambda x: x * 0.50},
        "Upgrade Shipping Tier":        {"ShippingRisk":   lambda x: (x - 1).clip(lower=0)},
        "Eliminate OrderRisk":          {"OrderRisk": 0},
        "Eliminate StockRisk":          {"StockRisk": 0},
        "Reduce DemandSpike –20%":      {"DemandSpike":    lambda x: x * 0.80},
        "Reduce GeoPoliticalRisk –50%": {"GeopoliticalRisk": lambda x: x * 0.50},
        "⭐ Combined Best":              {
            "TransportDelay": lambda x: x * 0.70,
            "OrderRisk":  0,
            "StockRisk":  0,
            "ShippingRisk": lambda x: (x - 1).clip(lower=0),
        },
    }

    rows = []
    for name, sc in SCENARIOS.items():
        X_s  = apply_intervention(X_base, sc)
        rate = rf.predict_proba(X_s)[:, 1].mean()
        rows.append({
            "Scenario":      name,
            "Rate":          rate,
            "Reduction":     baseline - rate,
            "Reduction (%)": round((baseline - rate) / baseline * 100, 1),
        })

    sc_df = pd.DataFrame(rows).sort_values("Rate")
    bar_colors = [
        "#95A5A6" if r["Scenario"] == "Baseline"
        else "#C0392B" if "Combined" in r["Scenario"]
        else "#1D9E75" if r["Reduction (%)"] > 5
        else "#E67E22"
        for _, r in sc_df.iterrows()
    ]

    fig = go.Figure(go.Bar(
        x=sc_df["Rate"],
        y=sc_df["Scenario"],
        orientation="h",
        marker_color=bar_colors,
        text=[
            f"{r:.1%}  (baseline)" if s == "Baseline"
            else f"{r:.1%}  (↓ {rd:.1f}%)"
            for s, r, rd in zip(sc_df["Scenario"], sc_df["Rate"], sc_df["Reduction (%)"])
        ],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Rate: %{x:.1%}<extra></extra>",
    ))
    fig.add_vline(x=baseline, line_width=2, line_dash="dash", line_color="#C0392B",
                  annotation_text="Baseline", annotation_position="top right")
    fig.update_layout(
        height=420, margin=dict(l=10, r=180, t=30, b=10),
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(tickformat=".0%", range=[0, baseline * 1.3],
                   title="Predicted Disruption Rate", gridcolor="#F0F0F0"),
        yaxis=dict(tickfont=dict(size=11)),
        title="Disruption Rate by Intervention Strategy",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="info-box">
    <strong>How this works:</strong> Each intervention is applied to every order simultaneously.
    The trained Random Forest then re-predicts disruption probability for the modified dataset.
    The difference is the estimated causal impact of that intervention — grounded in the
    model's learned feature-outcome relationships.
    </div>
    """, unsafe_allow_html=True)
