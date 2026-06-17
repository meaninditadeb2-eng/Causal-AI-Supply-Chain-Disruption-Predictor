"""Page 2 — Real-time Disruption Predictor (no SHAP)"""
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import (load_models, CAUSAL_FEATURES, FEATURE_LABELS, FEATURE_DESC,
                   SHIPPING_MODES, MARKETS, predict_proba, risk_badge,
                   feature_contributions)


def show():
    st.markdown('<p class="section-title">🎯 Real-Time Disruption Predictor</p>',
                unsafe_allow_html=True)
    st.markdown(
        "Set the order parameters below. The model scores every change **instantly** "
        "and shows which factors are driving the risk."
    )

    data = load_models()
    rf   = data["rf"]

    # ── Input sliders ──────────────────────────────────────────────────────
    st.markdown("### 📋 Order Parameters")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Logistics**")
        transport_delay = st.slider(
            "Transport Delay (days over schedule)",
            min_value=-4.0, max_value=8.0, value=1.0, step=0.5,
            help=FEATURE_DESC["TransportDelay"],
        )
        shipping_mode = st.selectbox(
            "Shipping Mode",
            options=list(SHIPPING_MODES.keys()),
            format_func=lambda x: SHIPPING_MODES[x],
            index=1,
            help=FEATURE_DESC["ShippingRisk"],
        )
        market = st.selectbox(
            "Market Region",
            options=list(MARKETS.keys()),
            index=0,
            help=FEATURE_DESC["GeopoliticalRisk"],
        )
        demand_spike = st.slider(
            "Demand Spike  (0 = normal → 1 = extreme surge)",
            min_value=0.0, max_value=1.0, value=0.4, step=0.05,
            help=FEATURE_DESC["DemandSpike"],
        )

    with col2:
        st.markdown("**Order & Inventory**")
        inventory_pressure = st.slider(
            "Inventory Pressure  (0 = well-stocked → 1 = depleted)",
            min_value=0.0, max_value=1.0, value=0.4, step=0.05,
            help=FEATURE_DESC["InventoryPressure"],
        )
        order_risk = st.selectbox(
            "Order Status",
            options=[0, 1],
            format_func=lambda x: "Normal  ✅" if x == 0
                                  else "At Risk ⚠️  (canceled / fraud / review)",
            help=FEATURE_DESC["OrderRisk"],
        )
        stock_risk = st.selectbox(
            "Product Availability",
            options=[0, 1],
            format_func=lambda x: "In Stock  ✅" if x == 0 else "Out of Stock  ⚠️",
            help=FEATURE_DESC["StockRisk"],
        )
        price_pressure = st.slider(
            "Discount Rate  (0 = no discount → 0.5 = 50 % off)",
            min_value=0.0, max_value=0.5, value=0.05, step=0.01,
            help=FEATURE_DESC["PricePressure"],
        )

    # ── Build input dict ───────────────────────────────────────────────────
    input_vals = {
        "TransportDelay":    transport_delay,
        "ShippingRisk":      float(shipping_mode),
        "GeopoliticalRisk":  MARKETS[market],
        "DemandSpike":       demand_spike,
        "InventoryPressure": inventory_pressure,
        "OrderRisk":         float(order_risk),
        "StockRisk":         float(stock_risk),
        "PricePressure":     price_pressure,
    }

    st.markdown("---")

    # ── Prediction output ──────────────────────────────────────────────────
    prob               = predict_proba(rf, input_vals)
    label, css, color  = risk_badge(prob)

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.markdown("### Prediction")
        st.markdown(f"""
        <div class="card" style="text-align:center; padding:2rem 1rem;">
            <div style="font-size:3.2rem; font-weight:900; color:{color};">{prob:.1%}</div>
            <div style="font-size:1.2rem; font-weight:700; color:{color};
                        margin-top:0.5rem;">{label}</div>
            <div style="font-size:0.82rem; color:#888; margin-top:0.7rem;">
                Disruption probability<br>Random Forest · Causal Features
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("### Risk Gauge")
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%", "font": {"size": 38, "color": color}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar":  {"color": color, "thickness": 0.32},
                "steps": [
                    {"range": [0,  40], "color": "#E8F5E9"},
                    {"range": [40, 70], "color": "#FFF8E6"},
                    {"range": [70,100], "color": "#FFEBEE"},
                ],
                "threshold": {
                    "line": {"color": color, "width": 4},
                    "thickness": 0.78, "value": prob * 100,
                },
            },
        ))
        fig.update_layout(height=220, margin=dict(l=15, r=15, t=15, b=10),
                          paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    with col3:
        st.markdown("### Action")
        if prob >= 0.70:
            st.markdown("""
            <div class="warn-box">
            <strong>⚡ Immediate action required</strong><br><br>
            • Escalate to logistics manager<br>
            • Check alternative routes<br>
            • Notify customer proactively<br>
            • Verify order status integrity<br>
            • Consider expedited upgrade
            </div>
            """, unsafe_allow_html=True)
        elif prob >= 0.40:
            st.markdown("""
            <div class="info-box">
            <strong>👁️ Monitor closely</strong><br><br>
            • Flag for daily status check<br>
            • Review transport delay root cause<br>
            • Confirm inventory availability<br>
            • Consider proactive communication
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="success-box">
            <strong>✅ Low risk — standard processing</strong><br><br>
            • No immediate action needed<br>
            • Continue normal monitoring<br>
            • Good candidate for standard shipping
            </div>
            """, unsafe_allow_html=True)

    # ── Feature contributions (fast, no SHAP) ─────────────────────────────
    st.markdown("---")
    st.markdown("### 🔍 What Is Driving This Prediction?")
    st.markdown(
        "<div class='info-box'>Each bar shows how much this feature's current value "
        "pushes the risk score up (🔴) or down (🟢), compared to the average order. "
        "Calculated from feature importance × normalised deviation from mean — "
        "instant, no SHAP required.</div>",
        unsafe_allow_html=True,
    )

    contrib_df = feature_contributions(rf, input_vals)
    bar_colors = ["#C0392B" if v > 0 else "#1D9E75" for v in contrib_df["Contribution"]]

    fig = go.Figure(go.Bar(
        x=contrib_df["Contribution"],
        y=contrib_df["Feature"],
        orientation="h",
        marker_color=bar_colors,
        text=[f"{v:+.4f}  (val = {r})"
              for v, r in zip(contrib_df["Contribution"], contrib_df["Value"])],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Contribution: %{x:+.4f}<extra></extra>",
    ))
    fig.add_vline(x=0, line_width=1.5, line_color="#333")
    fig.update_layout(
        height=360, margin=dict(l=10, r=160, t=30, b=10),
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(title="Contribution (red = increases risk, green = reduces risk)",
                   gridcolor="#F0F0F0"),
        yaxis=dict(tickfont=dict(size=12)),
        title="Feature Contributions — This Order vs Average Order",
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Quick scenario comparison ──────────────────────────────────────────
    st.markdown("---")
    st.markdown("### ⚡ Quick Scenario Comparison")
    st.markdown("See how your current order compares to three reference scenarios.")

    scenarios = {
        "Best Case (all low)": {
            "TransportDelay": -1.0, "ShippingRisk": 0.0,
            "GeopoliticalRisk": 0.0, "DemandSpike": 0.1,
            "InventoryPressure": 0.1, "OrderRisk": 0.0,
            "StockRisk": 0.0, "PricePressure": 0.0,
        },
        "Your Order": input_vals,
        "Average Order": {
            "TransportDelay": 0.0, "ShippingRisk": 1.5,
            "GeopoliticalRisk": 0.5, "DemandSpike": 0.4,
            "InventoryPressure": 0.4, "OrderRisk": 0.0,
            "StockRisk": 0.0, "PricePressure": 0.05,
        },
        "Worst Case (all high)": {
            "TransportDelay": 6.0, "ShippingRisk": 3.0,
            "GeopoliticalRisk": 1.0, "DemandSpike": 0.9,
            "InventoryPressure": 0.9, "OrderRisk": 1.0,
            "StockRisk": 1.0, "PricePressure": 0.4,
        },
    }

    probs  = {k: predict_proba(rf, v) for k, v in scenarios.items()}
    colors_sc = ["#1D9E75" if p < 0.4 else "#E67E22" if p < 0.7
                 else "#C0392B" for p in probs.values()]
    highlight = ["#2E75B6" if k == "Your Order" else c
                 for k, c in zip(probs.keys(), colors_sc)]

    fig = go.Figure(go.Bar(
        x=list(probs.keys()),
        y=list(probs.values()),
        marker_color=highlight,
        text=[f"{v:.1%}" for v in probs.values()],
        textposition="outside",
    ))
    fig.update_layout(
        height=300, margin=dict(l=10, r=10, t=30, b=10),
        plot_bgcolor="white", paper_bgcolor="white",
        yaxis=dict(tickformat=".0%", range=[0, 1.1], gridcolor="#F0F0F0",
                   title="Disruption Probability"),
        title="Your Order vs Reference Scenarios",
    )
    st.plotly_chart(fig, use_container_width=True)
