"""Page 4 — Causal Graph Explorer"""
import streamlit as st
import plotly.graph_objects as go
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import SCM_EFFECTS


def show():
    st.markdown('<p class="section-title">🕸️ Causal Graph Explorer</p>',
                unsafe_allow_html=True)
    st.markdown(
        "The Structural Causal Model (SCM) that defines causal relationships "
        "in this project. Hover over any edge for details."
    )

    # ── Node layout ────────────────────────────────────────────────────────
    NODES = {
        "WeatherRisk":      (-3.5,  2.2, "#E8A87C", "Root Cause"),
        "GeopoliticalRisk": (-3.5,  0.7, "#E8A87C", "Root Cause"),
        "DemandSpike":      (-3.5, -0.7, "#E8A87C", "Root Cause"),
        "CyberIncident":    (-3.5, -2.2, "#E8A87C", "Root Cause"),
        "SupplierDelay":    ( 0.0,  2.0, "#5DADE2", "Mediator"),
        "TransportDelay":   ( 0.0,  0.7, "#5DADE2", "Mediator"),
        "InventoryLevel":   ( 0.0, -0.7, "#5DADE2", "Mediator"),
        "PortCongestion":   ( 0.0, -2.0, "#5DADE2", "Mediator"),
        "ProductionRate":   ( 0.0, -3.1, "#5DADE2", "Mediator"),
        "Disruption":       ( 3.5,  0.0, "#E74C3C", "Outcome"),
    }

    EDGES = [
        ("WeatherRisk",      "SupplierDelay",  "+0.60", False),
        ("WeatherRisk",      "TransportDelay", "+0.70", False),
        ("GeopoliticalRisk", "SupplierDelay",  "+0.50", False),
        ("GeopoliticalRisk", "TransportDelay", "+0.30", False),
        ("DemandSpike",      "InventoryLevel", "−1.00", True),
        ("DemandSpike",      "PortCongestion", "+0.40", False),
        ("TransportDelay",   "PortCongestion", "+0.50", False),
        ("SupplierDelay",    "ProductionRate", "−1.00", True),
        ("SupplierDelay",    "Disruption",     "+2.00 ⭐", False),
        ("TransportDelay",   "Disruption",     "+1.80 ⭐", False),
        ("PortCongestion",   "Disruption",     "+1.50", False),
        ("CyberIncident",    "Disruption",     "+1.20", False),
        ("InventoryLevel",   "Disruption",     "−1.70 🛡️", True),
        ("ProductionRate",   "Disruption",     "−1.30 🛡️", True),
    ]

    fig = go.Figure()

    # Draw edges
    for src, tgt, coeff, protective in EDGES:
        x0, y0 = NODES[src][:2]
        x1, y1 = NODES[tgt][:2]
        color   = "#1D9E75" if protective else "#C0392B"
        width   = 1.5 + abs(float(coeff.replace("⭐","").replace("🛡️","")
                                        .replace("−","-").strip())) * 0.4

        fig.add_trace(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            mode="lines",
            line=dict(width=width, color=color),
            opacity=0.65,
            hoverinfo="skip",
            showlegend=False,
        ))
        # Hover dot at midpoint
        fig.add_trace(go.Scatter(
            x=[(x0 + x1) / 2], y=[(y0 + y1) / 2],
            mode="markers",
            marker=dict(size=10, color=color, opacity=0.0),
            hovertemplate=(
                f"<b>{src} → {tgt}</b><br>"
                f"Coefficient: {coeff}<br>"
                f"{'🛡️ Protective (reduces risk)' if protective else '⚠️ Increases risk'}"
                "<extra></extra>"
            ),
            showlegend=False,
        ))

    # Draw nodes
    for name, (nx, ny, color, role) in NODES.items():
        is_outcome = name == "Disruption"
        fig.add_trace(go.Scatter(
            x=[nx], y=[ny],
            mode="markers+text",
            marker=dict(
                size=52 if is_outcome else 44,
                color=color,
                line=dict(color="white", width=2.5 if is_outcome else 1.5),
            ),
            text=[name],
            textposition="top center",
            textfont=dict(size=9.5 if not is_outcome else 10, color="#222"),
            hovertemplate=(
                f"<b>{name}</b><br>Role: {role}<extra></extra>"
            ),
            showlegend=False,
        ))

    # Legend traces
    for color, label in [
        ("#E8A87C", "Root Cause (Exogenous)"),
        ("#5DADE2", "Mediator"),
        ("#E74C3C", "Outcome"),
        ("#C0392B", "Risk-increasing path"),
        ("#1D9E75", "Protective path"),
    ]:
        mode = "markers" if label in ("Root Cause (Exogenous)", "Mediator", "Outcome") else "lines"
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode=mode,
            marker=dict(size=12, color=color),
            line=dict(width=3, color=color),
            name=label, showlegend=True,
        ))

    fig.update_layout(
        height=560,
        margin=dict(l=10, r=10, t=15, b=10),
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(visible=False, range=[-4.5, 4.5]),
        yaxis=dict(visible=False, range=[-4.2, 3.2]),
        legend=dict(x=0.01, y=0.01, bgcolor="rgba(255,255,255,0.92)",
                    bordercolor="#DDD", borderwidth=1),
        hovermode="closest",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── Two info panels ────────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Direct Effects on Disruption")
        effects = [
            ("SupplierDelay",  "+2.00", "🔴 Strongest driver", "#FFEBEE"),
            ("TransportDelay", "+1.80", "🔴 Second strongest", "#FFEBEE"),
            ("InventoryLevel", "−1.70", "🟢 Protective #1",   "#E8F5E9"),
            ("PortCongestion", "+1.50", "🔴 Third driver",     "#FFEBEE"),
            ("ProductionRate", "−1.30", "🟢 Protective #2",   "#E8F5E9"),
            ("CyberIncident",  "+1.20", "🔴 Fourth driver",    "#FFEBEE"),
        ]
        for var, coef, label, bg in effects:
            st.markdown(f"""
            <div style="background:{bg}; border-radius:7px; padding:0.45rem 0.9rem;
                        margin-bottom:0.35rem; display:flex; justify-content:space-between;
                        align-items:center;">
                <span style="font-weight:600; color:#1B3A6B">{var}</span>
                <span style="font-family:monospace; font-weight:700">{coef}</span>
                <span style="font-size:0.85rem">{label}</span>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown("### Causal Discovery Validation")
        st.markdown("""
        <div class="card card-teal">
        We ran the <strong>PC algorithm</strong> on synthetic data where we know the
        true 13-edge graph. Results:
        </div>
        """, unsafe_allow_html=True)

        for metric, val, color, note in [
            ("Precision",              "1.000", "#1D9E75", "Every discovered edge is real"),
            ("Recall",                 "0.692", "#E67E22", "9 of 13 true edges found"),
            ("F1 Score",               "0.800", "#2E75B6", "Strong overall recovery"),
            ("Structural Hamming Dist","4",     "#95A5A6", "0 = perfect (lower is better)"),
            ("False Positive Edges",   "0",     "#1D9E75", "No spurious causal claims"),
        ]:
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between; align-items:center;
                        padding:0.38rem 0.6rem; border-bottom:1px solid #EEE;">
                <span style="color:#555; font-size:0.9rem">{metric}</span>
                <span style="font-weight:700; color:{color}; font-size:1.05rem">{val}</span>
                <span style="color:#888; font-size:0.8rem; text-align:right">{note}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class="success-box" style="margin-top:0.8rem">
        <strong>Precision = 1.000</strong> means the algorithm never claimed a false
        causal relationship. All its discoveries are scientifically trustworthy.
        </div>
        """, unsafe_allow_html=True)
