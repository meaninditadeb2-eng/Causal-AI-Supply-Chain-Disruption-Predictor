"""
utils.py
========
Shared utilities — model training, data generation, prediction helpers.
No SHAP dependency — all explainability uses fast built-in importance scores.
"""

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")

# ── Feature definitions ────────────────────────────────────────────────────
CAUSAL_FEATURES = [
    "TransportDelay",
    "ShippingRisk",
    "GeopoliticalRisk",
    "DemandSpike",
    "InventoryPressure",
    "OrderRisk",
    "StockRisk",
    "PricePressure",
]

FEATURE_LABELS = {
    "TransportDelay":    "Transport Delay (days over schedule)",
    "ShippingRisk":      "Shipping Risk (0=Same Day → 3=Standard)",
    "GeopoliticalRisk":  "Geopolitical Risk (0=USCA → 1=Africa)",
    "DemandSpike":       "Demand Spike (0=normal → 1=extreme)",
    "InventoryPressure": "Inventory Pressure (0=well-stocked → 1=depleted)",
    "OrderRisk":         "Order Risk (0=Normal / 1=Canceled/Fraud)",
    "StockRisk":         "Stock Risk (0=In Stock / 1=Out of Stock)",
    "PricePressure":     "Discount Rate Applied (0–0.5)",
}

FEATURE_DESC = {
    "TransportDelay":    "Days real − Days scheduled. Positive = late.",
    "ShippingRisk":      "Ordered by typical delay risk: Same Day=0, Standard=3.",
    "GeopoliticalRisk":  "Regional risk proxy mapped from Market column.",
    "DemandSpike":       "Normalised order quantity — sudden demand surge.",
    "InventoryPressure": "1 − normalised profit ratio. Low margin = high pressure.",
    "OrderRisk":         "1 if status is CANCELED, SUSPECTED_FRAUD, or PAYMENT_REVIEW.",
    "StockRisk":         "1 if product status = unavailable.",
    "PricePressure":     "Discount rate as demand/margin signal.",
}

SHIPPING_MODES = {0: "Same Day", 1: "First Class", 2: "Second Class", 3: "Standard Class"}
MARKETS = {"USCA": 0.00, "Europe": 0.25, "LATAM": 0.50, "Pacific Asia": 0.75, "Africa": 1.00}

# True SCM causal effects (logit coefficients from structural equations)
SCM_EFFECTS = {
    "SupplierDelay":   +2.00,
    "TransportDelay":  +1.80,
    "InventoryLevel":  -1.70,
    "PortCongestion":  +1.50,
    "ProductionRate":  -1.30,
    "CyberIncident":   +1.20,
}

# ── Training data generation ───────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def generate_training_data(n: int = 12000, seed: int = 99) -> pd.DataFrame:
    """Generate realistic DataCo-like training data."""
    rng = np.random.default_rng(seed)

    transport_delay    = rng.normal(0.0, 2.5, n)
    shipping_risk      = rng.integers(0, 4, n).astype(float)
    geopolitical_risk  = rng.uniform(0, 1, n)
    demand_spike       = rng.beta(2, 3, n)
    inventory_pressure = rng.beta(3, 2, n)
    order_risk         = rng.binomial(1, 0.12, n).astype(float)
    stock_risk         = rng.binomial(1, 0.08, n).astype(float)
    price_pressure     = rng.beta(1, 5, n)

    logit = (
        0.45 * transport_delay
      + 0.30 * shipping_risk
      + 0.25 * geopolitical_risk
      + 0.20 * demand_spike
      + 0.15 * inventory_pressure
      + 0.80 * order_risk
      + 0.40 * stock_risk
      + 0.10 * price_pressure
      - 0.50
    )
    prob       = 1 / (1 + np.exp(-logit))
    disruption = rng.binomial(1, prob, n)

    return pd.DataFrame({
        "TransportDelay":    transport_delay,
        "ShippingRisk":      shipping_risk,
        "GeopoliticalRisk":  geopolitical_risk,
        "DemandSpike":       demand_spike,
        "InventoryPressure": inventory_pressure,
        "OrderRisk":         order_risk,
        "StockRisk":         stock_risk,
        "PricePressure":     price_pressure,
        "Disruption":        disruption,
    })

# ── Train and cache all models ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models() -> dict:
    """Train RF, XGBoost, and Logistic Regression. Cache result."""
    df = generate_training_data()
    X  = df[CAUSAL_FEATURES]
    y  = df["Disruption"]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Random Forest
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)

    # XGBoost
    xgb = XGBClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=5,
        random_state=42, verbosity=0, eval_metric="logloss"
    )
    xgb.fit(X_tr, y_tr)

    # Logistic Regression
    lr = Pipeline([
        ("sc",  StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    lr.fit(X_tr, y_tr)

    def metrics(model, Xte=X_te, yte=y_te):
        yp    = model.predict(Xte)
        proba = model.predict_proba(Xte)[:, 1]
        return {
            "Accuracy":  round(accuracy_score(yte, yp),   4),
            "F1":        round(f1_score(yte, yp),         4),
            "ROC-AUC":   round(roc_auc_score(yte, proba), 4),
        }

    # Feature importance (normalised 0–1)
    rf_imp  = rf.feature_importances_
    xgb_imp = xgb.feature_importances_
    avg_imp = (rf_imp + xgb_imp) / 2

    importance_df = pd.DataFrame({
        "Feature":   CAUSAL_FEATURES,
        "RF":        rf_imp,
        "XGBoost":   xgb_imp,
        "Average":   avg_imp,
    }).sort_values("Average", ascending=False)

    return {
        "rf":           rf,
        "xgb":          xgb,
        "lr":           lr,
        "X_tr":         X_tr,
        "X_te":         X_te,
        "y_tr":         y_tr,
        "y_te":         y_te,
        "df":           df,
        "importance":   importance_df,
        "metrics": {
            "Logistic Regression": metrics(lr),
            "Random Forest":       metrics(rf),
            "XGBoost":             metrics(xgb),
        },
    }

# ── Prediction helpers ─────────────────────────────────────────────────────
def predict_proba(model, input_dict: dict) -> float:
    """Predict disruption probability for a single order."""
    row = pd.DataFrame([input_dict])[CAUSAL_FEATURES]
    return float(model.predict_proba(row)[0, 1])


def risk_badge(prob: float) -> tuple:
    """Return (label, css_class, hex_color) based on probability."""
    if prob >= 0.70:
        return "🔴  HIGH RISK",   "risk-high",   "#C0392B"
    elif prob >= 0.40:
        return "🟡  MEDIUM RISK", "risk-medium", "#E67E22"
    else:
        return "🟢  LOW RISK",    "risk-low",    "#1D9E75"


def feature_contributions(model, input_dict: dict) -> pd.DataFrame:
    """
    Fast feature contribution estimate using mean-baseline comparison.
    For each feature: contribution = importance × (value − mean) normalised.
    No SHAP required — runs instantly.
    """
    df_train  = generate_training_data()
    means     = df_train[CAUSAL_FEATURES].mean()
    stds      = df_train[CAUSAL_FEATURES].std().replace(0, 1)
    imp       = model.feature_importances_

    contribs = []
    for i, feat in enumerate(CAUSAL_FEATURES):
        val        = input_dict[feat]
        z_score    = (val - means[feat]) / stds[feat]
        contrib    = float(imp[i] * z_score)
        contribs.append({
            "Feature":      feat,
            "Value":        round(val, 3),
            "Contribution": round(contrib, 4),
            "Direction":    "↑ Increases risk" if contrib > 0 else "↓ Reduces risk",
        })

    return pd.DataFrame(contribs).sort_values("Contribution", ascending=False)
