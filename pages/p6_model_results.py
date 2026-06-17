"""Page 6 — Model Results & Evaluation"""
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_curve, roc_auc_score, confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score,
)
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import load_models, CAUSAL_FEATURES


def show():
    st.markdown('<p class="section-title">📈 Model Evaluation Results</p>',
                unsafe_allow_html=True)

    data = load_models()
    rf   = data["rf"]
    xgb  = data["xgb"]
    lr   = data["lr"]
    X_te = data["X_te"]
    y_te = data["y_te"]

    models = {
        "Logistic Regression": lr,
        "Random Forest":       rf,
        "XGBoost":             xgb,
    }

    # ── Compute metrics + ROC ──────────────────────────────────────────────
    rows     = []
    roc_data = {}
    for name, model in models.items():
        yp    = model.predict(X_te)
        proba = model.predict_proba(X_te)[:, 1]
        fpr, tpr, _ = roc_curve(y_te, proba)
        auc   = roc_auc_score(y_te, proba)
        roc_data[name] = (fpr, tpr, auc)
        rows.append({
            "Model":     name,
            "Accuracy":  round(accuracy_score(y_te, yp),   4),
            "Precision": round(precision_score(y_te, yp),  4),
            "Recall":    round(recall_score(y_te, yp),     4),
            "F1":        round(f1_score(y_te, yp),         4),
            "ROC-AUC":   round(auc,                        4),
        })

    results_df = pd.DataFrame(rows).set_index("Model")

    # ── Summary metrics ───────────────────────────────────────────────────
    best = results_df["ROC-AUC"].idxmax()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🎯 Best AUC",       f"{results_df.loc[best,'ROC-AUC']:.4f}", best)
    c2.metric("📊 Best F1",        f"{results_df.loc[best,'F1']:.4f}")
    c3.metric("✅ Best Accuracy",  f"{results_df.loc[best,'Accuracy']:.4f}")
    c4.metric("📦 Test Set Size",  f"{len(y_te):,} orders")

    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Comparison",
        "📈 ROC Curves",
        "🔲 Confusion Matrix",
        "🔍 Error Analysis",
    ])

    # ── Tab 1: Bar comparison ──────────────────────────────────────────────
    with tab1:
        st.markdown("### Model Performance Comparison")

        metric_colors = {
            "Accuracy":  "#1D9E75",
            "Precision": "#2E75B6",
            "Recall":    "#8E44AD",
            "F1":        "#E67E22",
            "ROC-AUC":   "#C0392B",
        }
        fig = go.Figure()
        for m, c in metric_colors.items():
            fig.add_trace(go.Bar(
                name=m,
                x=list(results_df.index),
                y=results_df[m].values,
                marker_color=c, opacity=0.85,
                text=[f"{v:.4f}" for v in results_df[m].values],
                textposition="outside",
                textfont=dict(size=9),
            ))
        fig.update_layout(
            barmode="group", height=430,
            margin=dict(l=10, r=10, t=40, b=20),
            plot_bgcolor="white", paper_bgcolor="white",
            yaxis=dict(range=[0.45, 1.08], gridcolor="#F0F0F0",
                       title="Score"),
            legend=dict(orientation="h", x=0, y=1.12),
            title="All Models — All Metrics (Causal Features Only)",
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Metrics table — Plotly table, no external dependencies ──────
        st.markdown("### Detailed Metrics Table")

        # Build colour per cell using plotly table instead
        cell_colors = []
        for col in results_df.columns:
            vals   = results_df[col].values
            colors = []
            for v in vals:
                ratio = (v - 0.5) / 0.5          # 0.5 → red, 1.0 → green
                ratio = max(0.0, min(1.0, ratio))
                r = int(255 * (1 - ratio))
                g = int(200 * ratio + 55)
                colors.append(f"rgba({r},{g},80,0.25)")
            cell_colors.append(colors)

        fig_tbl = go.Figure(go.Table(
            header=dict(
                values=["<b>Model</b>"] + [f"<b>{c}</b>" for c in results_df.columns],
                fill_color="#1B3A6B",
                font=dict(color="white", size=12),
                align="center",
                height=32,
            ),
            cells=dict(
                values=[list(results_df.index)] +
                       [[f"{v:.4f}" for v in results_df[c]] for c in results_df.columns],
                fill_color=[["white"] * len(results_df)] + cell_colors,
                font=dict(color="#1B3A6B", size=12),
                align="center",
                height=30,
            ),
        ))
        fig_tbl.update_layout(
            height=160,
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig_tbl, use_container_width=True)

        st.markdown("""
        <div class="warn-box">
        <strong>Important — Data Leakage Warning:</strong>
        Models trained on <em>all</em> raw DataCo features achieve AUC = 1.000
        because <code>Days for shipping (real)</code> directly encodes the target variable.
        The results above use <strong>only the 8 causal features</strong> — these represent
        true generalisation performance without leakage. This is a critical distinction
        that most practitioners miss.
        </div>
        """, unsafe_allow_html=True)

    # ── Tab 2: ROC curves ──────────────────────────────────────────────────
    with tab2:
        st.markdown("### ROC Curves — All Models")
        fig = go.Figure()
        roc_colors = ["#3498DB", "#1D9E75", "#C0392B"]

        for (name, (fpr, tpr, auc)), color in zip(roc_data.items(), roc_colors):
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                name=f"{name}  (AUC = {auc:.4f})",
                line=dict(width=2.5, color=color),
                hovertemplate=(
                    f"<b>{name}</b><br>"
                    "FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>"
                ),
            ))

        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            line=dict(dash="dash", color="#AAA", width=1.5),
            name="Random Classifier  (AUC = 0.500)",
            showlegend=True,
        ))
        fig.update_layout(
            height=480,
            margin=dict(l=10, r=10, t=40, b=10),
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(title="False Positive Rate", gridcolor="#F0F0F0",
                       range=[-0.01, 1.01]),
            yaxis=dict(title="True Positive Rate",  gridcolor="#F0F0F0",
                       range=[-0.01, 1.01]),
            legend=dict(x=0.42, y=0.12, bgcolor="rgba(255,255,255,0.92)",
                        bordercolor="#DDD", borderwidth=1),
            title="ROC Curves — Causal Features Only",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        <div class="info-box">
        <strong>How to read ROC curves:</strong> The closer a curve hugs the top-left
        corner, the better the model. AUC = Area Under the Curve — a score of 1.0 is
        perfect, 0.5 is random guessing.
        </div>
        """, unsafe_allow_html=True)

    # ── Tab 3: Confusion matrix ────────────────────────────────────────────
    with tab3:
        st.markdown("### Confusion Matrix")
        chosen = st.selectbox("Select Model", list(models.keys()), index=1)
        model  = models[chosen]
        yp     = model.predict(X_te)
        cm     = confusion_matrix(y_te, yp)
        total  = cm.sum()

        fig = go.Figure(go.Heatmap(
            z=cm,
            x=["Predicted: On Time", "Predicted: Disrupted"],
            y=["Actual: On Time", "Actual: Disrupted"],
            colorscale=[[0, "#EAF3FE"], [1, "#1B3A6B"]],
            text=[
                [f"TN: {cm[0,0]:,}", f"FP: {cm[0,1]:,}"],
                [f"FN: {cm[1,0]:,}", f"TP: {cm[1,1]:,}"],
            ],
            texttemplate="<b>%{text}</b><br>%{z:,}",
            textfont=dict(size=14),
            showscale=False,
        ))
        fig.update_layout(
            height=380,
            margin=dict(l=10, r=10, t=50, b=10),
            title=(
                f"{chosen} — Confusion Matrix<br>"
                f"<sub>Accuracy: {(cm[0,0]+cm[1,1])/total:.3%}  |  "
                f"Test orders: {total:,}</sub>"
            ),
        )
        st.plotly_chart(fig, use_container_width=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("True Negatives",  f"{cm[0,0]:,}", "Correctly predicted on-time")
        c2.metric("False Positives", f"{cm[0,1]:,}", "False disruption alerts")
        c3.metric("False Negatives", f"{cm[1,0]:,}", "Missed disruptions ⚠️")
        c4.metric("True Positives",  f"{cm[1,1]:,}", "Correctly caught disruptions")

        st.markdown("""
        <div class="info-box">
        <strong>False Negatives</strong> (missed disruptions) are typically the most
        costly error — a disruption you didn't predict means no proactive action was taken.
        <strong>False Positives</strong> cause unnecessary escalation but are less costly.
        </div>
        """, unsafe_allow_html=True)

    # ── Tab 4: Error Analysis ──────────────────────────────────────────────
    with tab4:
        st.markdown("### Prediction Error Analysis")
        st.markdown(
            "Examine where the model makes mistakes and what those orders look like."
        )

        chosen2 = st.selectbox("Select Model for Error Analysis",
                               list(models.keys()), index=1, key="err_model")
        model2  = models[chosen2]
        proba2  = model2.predict_proba(X_te)[:, 1]
        yp2     = model2.predict(X_te)
        y_arr   = y_te.values

        err_df = X_te.copy()
        err_df["True Label"]   = y_arr
        err_df["Predicted"]    = yp2
        err_df["Probability"]  = proba2.round(3)
        err_df["Error Type"]   = "Correct"
        err_df.loc[(err_df["True Label"]==1)&(err_df["Predicted"]==0),
                   "Error Type"] = "False Negative"
        err_df.loc[(err_df["True Label"]==0)&(err_df["Predicted"]==1),
                   "Error Type"] = "False Positive"

        fn = err_df[err_df["Error Type"] == "False Negative"]
        fp = err_df[err_df["Error Type"] == "False Positive"]

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### False Negatives — Missed Disruptions")
            st.markdown(f"*{len(fn):,} orders predicted as on-time but actually disrupted*")
            if len(fn) > 0:
                st.dataframe(fn[CAUSAL_FEATURES + ["Probability"]].head(10),
                             use_container_width=True)
                st.markdown("**Average feature values (False Negatives):**")
                st.dataframe(fn[CAUSAL_FEATURES].mean().round(3).to_frame("Mean"),
                             use_container_width=True)

        with col2:
            st.markdown("#### False Positives — False Alarms")
            st.markdown(f"*{len(fp):,} orders predicted as disrupted but actually on-time*")
            if len(fp) > 0:
                st.dataframe(fp[CAUSAL_FEATURES + ["Probability"]].head(10),
                             use_container_width=True)
                st.markdown("**Average feature values (False Positives):**")
                st.dataframe(fp[CAUSAL_FEATURES].mean().round(3).to_frame("Mean"),
                             use_container_width=True)

        # Probability distribution of errors
        st.markdown("#### Prediction Confidence Distribution")
        fig = go.Figure()
        for etype, color in [
            ("Correct",        "#1D9E75"),
            ("False Negative", "#C0392B"),
            ("False Positive", "#E67E22"),
        ]:
            subset = err_df[err_df["Error Type"] == etype]["Probability"]
            if len(subset) > 0:
                fig.add_trace(go.Histogram(
                    x=subset,
                    name=f"{etype} ({len(subset):,})",
                    opacity=0.65,
                    marker_color=color,
                    nbinsx=30,
                    histnorm="probability density",
                ))
        fig.update_layout(
            barmode="overlay", height=320,
            margin=dict(l=10, r=10, t=30, b=10),
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(title="Predicted Probability", gridcolor="#F0F0F0"),
            yaxis=dict(title="Density", gridcolor="#F0F0F0"),
            legend=dict(x=0.55, y=0.98),
            title="Prediction Confidence — Correct vs Error Cases",
        )
        st.plotly_chart(fig, use_container_width=True)
