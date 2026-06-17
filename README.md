# Causal AI Supply Chain Dashboard

Interactive Streamlit dashboard for supply chain disruption prediction and causal analysis.
**No SHAP dependency** — all explainability uses fast built-in importance scores.

## 🚀 Live Demo

Deploy free on Streamlit Cloud:
1. Fork this repo
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repo → select `dashboard/app.py`
4. Deploy → get a public URL

## 📄 Pages

| Page | Description |
|------|-------------|
| 🏠 Overview | Project summary, key findings, architecture pipeline |
| 🎯 Disruption Predictor | Real-time risk score + feature contributions + scenario comparison |
| 🔄 Counterfactual Simulator | Intervention sliders + before/after gauges + ranked scenarios |
| 🕸️ Causal Graph | Interactive DAG with hover tooltips + discovery validation |
| 📊 Feature Importance | RF vs XGBoost + distributions + correlation analysis |
| 📈 Model Results | ROC curves + confusion matrix + error analysis |

## ⚡ Run Locally

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/causal-supply-chain
cd causal-supply-chain/dashboard

# Install dependencies (no SHAP needed)
pip install -r requirements.txt

# Launch
streamlit run app.py
```

## 🏗️ Structure

```
dashboard/
├── app.py                  ← Entry point + navigation + global CSS
├── utils.py                ← Model training, data generation, helpers
├── requirements.txt        ← Dependencies (no shap)
├── .streamlit/
│   └── config.toml         ← Theme + server config
└── pages/
    ├── p1_overview.py      ← Overview page
    ├── p2_predictor.py     ← Disruption predictor
    ├── p3_counterfactual.py← Counterfactual simulator
    ├── p4_causal_graph.py  ← Causal graph explorer
    ├── p5_feature_importance.py ← Feature analysis (replaces SHAP)
    └── p6_model_results.py ← Model evaluation
```

## 📦 Dependencies

```
streamlit, pandas, numpy, scikit-learn, xgboost, plotly, joblib, networkx, scipy
```

No heavy dependencies. Loads in under 60 seconds on Streamlit Cloud free tier.

## 🔍 Why No SHAP?

SHAP TreeExplainer requires several seconds per prediction batch and can cause
timeouts on Streamlit Cloud's free tier. Instead, this dashboard uses:

- **Feature importance** from RF + XGBoost (instant, built-in)
- **Feature contributions** via importance × normalised deviation (instant)
- **Distribution analysis** comparing disrupted vs on-time orders
- **Correlation analysis** with full heatmap

These provide equivalent analytical value for a production dashboard.
