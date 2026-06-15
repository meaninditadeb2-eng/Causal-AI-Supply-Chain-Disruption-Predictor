# Causal AI Supply Chain Dashboard

Interactive Streamlit dashboard for supply chain disruption prediction and causal analysis.

## Pages

| Page                        | Description                                      |
| --------------------------- | ------------------------------------------------ |
| 🏠 Overview                 | Project summary, key findings, architecture      |
| 🎯 Disruption Predictor     | Real-time disruption risk prediction             |
| 🔄 Counterfactual Simulator | What-if intervention analysis                    |
| 🕸️ Causal Graph            | Interactive DAG with effect sizes                |
| 📈 Model Results            | ROC curves, confusion matrix, feature importance |

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Run on Streamlit Cloud (Free Hosting)

1. Push this folder to a GitHub repository
2. Go to Streamlit Community Cloud
3. Connect your repository and select `app.py`
4. Deploy and receive a public URL

## Project Structure

```text
dashboard/
├── app.py               ← Main entry point + navigation
├── utils.py             ← Model training, data generation, shared helpers
├── requirements.txt     ← Project dependencies
├── pages/
│   ├── overview.py      ← Project overview
│   ├── predictor.py     ← Disruption prediction
│   ├── counterfactual.py← What-if intervention simulator
│   ├── causal_graph.py  ← Interactive causal graph
│   └── model_results.py ← Model evaluation metrics
```

## Key Features

* Real-time disruption risk prediction
* Counterfactual scenario analysis
* Causal graph visualization
* Root-cause discovery
* Interactive business intelligence dashboard
* XGBoost-based predictive modeling

## Tech Stack

* Python
* Streamlit
* XGBoost
* Scikit-Learn
* Pandas
* NumPy
* Plotly
* NetworkX
* Matplotlib

```
```
