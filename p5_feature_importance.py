"""Page 5 — Feature Importance (replaces SHAP — fast, no heavy dependencies)"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import load_models, CAUSAL_FEATURES, FEATURE_DESC, generate_training_data


def show():
    st.markdown('<p class="section-title">📊 Feature Importance Analysis</p>',
                unsafe_allow_html=True)
    st.markdown(
        "Compare feature importance across models and explore how each feature "
        "relates to disruption risk — no heavy computation required."
    )

    data        = load_models()
    rf          = data["rf"]
    xgb         = data["xgb"]
    imp_df      = data["importance"]
    df_train    = generate_training_data()

    tab1, tab2, tab3, tab4 = st.tabs([
        "🏆 Ranked Importance",
        "⚖️  RF vs XGBoost",
        "📉 Feature Distributions",
        "🔗 Correlation with Risk",
    ])

    # ── Tab 1: Ranked importance ───────────────────────────────────────────
    with tab1:
        st.markdown("### Global Feature Importance — Averaged Across RF + XGBoost")
        st.markdown("""
        <div class="info-box">
        Importance scores show how much each feature reduces impurity across all
        decision trees. Higher = more predictive of disruption.
        Features are averaged across both RF and XGBoost for robustness.
        </div>
        """, unsafe_allow_html=True)

        sorted_df = imp_df.sort_values("Average")
        pct       = sorted_df["Average"] / sorted_df["Average"].sum() * 100
        colors    = ["#C0392B" if v > sorted_df["Average"].median()
                     else "#3498DB" for v in sorted_df["Average"]]

        fig = go.Figure(go.Bar(
            x=sorted_df["Average"],
            y=sorted_df["Feature"],
            orientation="h",
            marker_color=colors,
            text=[f"{v:.4f}  ({p:.1f}%)"
                  for v, p in zip(sorted_df["Average"], pct[::-1])],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
        ))
        fig.update_layout(
            height=380, margin=dict(l=10, r=200, t=30, b=10),
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(title="Average Feature Importance", gridcolor="#F0F0F0"),
            yaxis=dict(tickfont=dict(size=12)),
            title="Feature Importance — Average of RF + XGBoost",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Importance table
        st.markdown("### Feature Importance Table")
        display_df = imp_df.copy()
        display_df["Description"] = [FEATURE_DESC[f] for f in display_df["Feature"]]
        display_df["RF (%)"]      = (display_df["RF"]      / display_df["RF"].sum() * 100).round(1)
        display_df["XGBoost (%)"] = (display_df["XGBoost"] / display_df["XGBoost"].sum() * 100).round(1)
        st.dataframe(
            display_df[["Feature","RF (%)","XGBoost (%)","Description"]]
            .reset_index(drop=True),
            use_container_width=True,
        )

        st.markdown("""
        <div class="success-box">
        <strong>Key finding:</strong> TransportDelay accounts for the vast majority of
        predictive power in both models. This aligns with DoWhy causal effect estimates —
        both statistical importance and causal ATE independently confirm it as the
        dominant driver of supply chain disruption.
        </div>
        """, unsafe_allow_html=True)

    # ── Tab 2: RF vs XGBoost comparison ───────────────────────────────────
    with tab2:
        st.markdown("### RF vs XGBoost — Feature Importance Comparison")
        st.markdown(
            "When two different algorithms rank features the same way, "
            "that convergence is evidence the signal is real, not model-specific."
        )

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Random Forest",
            x=imp_df["Feature"], y=imp_df["RF"],
            marker_color="#2E75B6", opacity=0.85,
            text=[f"{v:.4f}" for v in imp_df["RF"]],
            textposition="outside",
        ))
        fig.add_trace(go.Bar(
            name="XGBoost",
            x=imp_df["Feature"], y=imp_df["XGBoost"],
            marker_color="#1D9E75", opacity=0.85,
            text=[f"{v:.4f}" for v in imp_df["XGBoost"]],
            textposition="outside",
        ))
        fig.update_layout(
            barmode="group", height=420,
            margin=dict(l=10, r=10, t=40, b=60),
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(tickangle=-30, gridcolor="#F0F0F0"),
            yaxis=dict(title="Feature Importance", gridcolor="#F0F0F0"),
            legend=dict(x=0.65, y=0.98),
            title="RF vs XGBoost — Side-by-Side Feature Importance",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Rank comparison
        rf_rank  = imp_df.sort_values("RF",      ascending=False)["Feature"].tolist()
        xgb_rank = imp_df.sort_values("XGBoost", ascending=False)["Feature"].tolist()

        st.markdown("### Rank Comparison")
        rank_df = pd.DataFrame({
            "Feature":       CAUSAL_FEATURES,
            "RF Rank":       [rf_rank.index(f) + 1  for f in CAUSAL_FEATURES],
            "XGBoost Rank":  [xgb_rank.index(f) + 1 for f in CAUSAL_FEATURES],
        })
        rank_df["Rank Difference"] = abs(rank_df["RF Rank"] - rank_df["XGBoost Rank"])
        rank_df["Agreement"]       = rank_df["Rank Difference"].apply(
            lambda x: "✅ Strong" if x <= 1 else "⚠️ Moderate" if x <= 2 else "❌ Differs"
        )
        rank_df = rank_df.sort_values("RF Rank")
        st.dataframe(rank_df.reset_index(drop=True), use_container_width=True)

    # ── Tab 3: Feature distributions ──────────────────────────────────────
    with tab3:
        st.markdown("### Feature Distributions — Disrupted vs On-Time Orders")

        feat = st.selectbox(
            "Select feature to explore",
            CAUSAL_FEATURES,
            format_func=lambda x: f"{x} — {FEATURE_DESC[x]}",
        )

        disrupted     = df_train[df_train["Disruption"] == 1][feat]
        not_disrupted = df_train[df_train["Disruption"] == 0][feat]

        if feat in ("OrderRisk", "StockRisk"):
            # Binary feature — bar chart
            d0 = df_train[df_train[feat] == 0]["Disruption"].mean()
            d1 = df_train[df_train[feat] == 1]["Disruption"].mean()
            fig = go.Figure(go.Bar(
                x=[f"{feat}=0", f"{feat}=1"],
                y=[d0, d1],
                marker_color=["#1D9E75", "#C0392B"],
                text=[f"{d0:.1%}", f"{d1:.1%}"],
                textposition="outside",
            ))
            fig.update_layout(
                title=f"Disruption Rate by {feat} Value",
                height=340, margin=dict(l=10, r=10, t=40, b=10),
                plot_bgcolor="white", paper_bgcolor="white",
                yaxis=dict(tickformat=".0%", gridcolor="#F0F0F0"),
            )
        else:
            # Continuous feature — overlapping histograms
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=not_disrupted, name="On Time", opacity=0.65,
                marker_color="#1D9E75", nbinsx=40,
                histnorm="probability density",
            ))
            fig.add_trace(go.Histogram(
                x=disrupted, name="Disrupted", opacity=0.65,
                marker_color="#C0392B", nbinsx=40,
                histnorm="probability density",
            ))
            fig.update_layout(
                barmode="overlay",
                title=f"{feat} Distribution — Disrupted vs On-Time",
                height=380, margin=dict(l=10, r=10, t=40, b=10),
                plot_bgcolor="white", paper_bgcolor="white",
                xaxis=dict(title=feat, gridcolor="#F0F0F0"),
                yaxis=dict(title="Density", gridcolor="#F0F0F0"),
                legend=dict(x=0.7, y=0.98),
            )

        st.plotly_chart(fig, use_container_width=True)

        # Summary stats
        col1, col2, col3 = st.columns(3)
        col1.metric(f"Mean (Disrupted)",  f"{disrupted.mean():.3f}")
        col2.metric(f"Mean (On-Time)",    f"{not_disrupted.mean():.3f}")
        col3.metric("Difference",         f"{(disrupted.mean() - not_disrupted.mean()):+.3f}")

    # ── Tab 4: Correlation with disruption ─────────────────────────────────
    with tab4:
        st.markdown("### Feature Correlation with Disruption")
        st.markdown("""
        <div class="warn-box">
        <strong>Remember:</strong> Correlation ≠ Causation. These correlations motivated
        our causal analysis — the actual causal effects were confirmed by DoWhy (Colab_03).
        </div>
        """, unsafe_allow_html=True)

        corr = df_train[CAUSAL_FEATURES + ["Disruption"]].corr()["Disruption"].drop("Disruption")
        corr_df = corr.reset_index()
        corr_df.columns = ["Feature", "Correlation"]
        corr_df = corr_df.sort_values("Correlation")

        colors = ["#C0392B" if v > 0 else "#1D9E75" for v in corr_df["Correlation"]]
        fig = go.Figure(go.Bar(
            x=corr_df["Correlation"],
            y=corr_df["Feature"],
            orientation="h",
            marker_color=colors,
            text=[f"{v:+.3f}" for v in corr_df["Correlation"]],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Correlation: %{x:+.3f}<extra></extra>",
        ))
        fig.add_vline(x=0, line_width=1.5, line_color="#333")
        fig.update_layout(
            height=380, margin=dict(l=10, r=80, t=30, b=10),
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(title="Pearson Correlation with Disruption",
                       gridcolor="#F0F0F0"),
            yaxis=dict(tickfont=dict(size=12)),
            title="Correlation with Disruption — Red = risk factor, Green = protective",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Full correlation matrix heatmap
        st.markdown("### Full Correlation Matrix")
        corr_full = df_train[CAUSAL_FEATURES + ["Disruption"]].corr()
        fig = go.Figure(go.Heatmap(
            z=corr_full.values,
            x=corr_full.columns,
            y=corr_full.columns,
            colorscale="RdBu_r",
            zmid=0, zmin=-1, zmax=1,
            text=[[f"{v:.2f}" for v in row] for row in corr_full.values],
            texttemplate="%{text}",
            textfont=dict(size=9),
            colorbar=dict(thickness=15),
        ))
        fig.update_layout(
            height=420, margin=dict(l=10, r=10, t=30, b=10),
            xaxis=dict(tickangle=-30),
            title="Correlation Matrix — All Causal Features",
        )
        st.plotly_chart(fig, use_container_width=True)
