import gradio as gr
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt

# ----------------- DATA LOADING -----------------
def load_data():
    try:
        return pd.read_csv("df_transformed.csv")
    except FileNotFoundError:
        return None

df = load_data()
if df is None:
    raise FileNotFoundError("df_transformed.csv not found in the folder.")

# ----------------- FEATURES & TARGETS -----------------
feature_columns = [
    'LME_ZS_Close', 'US_Stock_OKE_adj_close', 'US_Stock_TECK_adj_close',
    'US_Stock_TRGP_adj_close', 'US_Stock_CLF_adj_close', 'US_Stock_HAL_adj_close',
    'US_Stock_X_adj_close', 'LME_AH_Close', 'LME_CA_Close',
    'JPX_Gold_Standard_Futures_Close', 'US_Stock_OXY_adj_close',
    'US_Stock_CVE_adj_close', 'JPX_Platinum_Standard_Futures_Close',
    'US_Stock_HL_adj_close', 'LME_PB_Close', 'US_Stock_ALB_adj_close',
    'US_Stock_DVN_adj_close', 'US_Stock_OIH_adj_close'
]

target_columns = [
    'LME_AH_Close - US_Stock_CVE_adj_close',
    'US_Stock_HL_adj_close - LME_AH_Close',
    'JPX_Platinum_Standard_Futures_Close - US_Stock_CLF_adj_close',
    'US_Stock_TRGP_adj_close - LME_CA_Close',
    'JPX_Gold_Standard_Futures_Close - US_Stock_X_adj_close',
    'US_Stock_OXY_adj_close - LME_CA_Close',
    'US_Stock_DVN_adj_close - LME_ZS_Close',
    'LME_ZS_Close - US_Stock_HAL_adj_close',
    'JPX_Platinum_Standard_Futures_Close - US_Stock_TRGP_adj_close',
    'US_Stock_OXY_adj_close - LME_AH_Close',
    'LME_AH_Close - US_Stock_CLF_adj_close',
    'JPX_Platinum_Standard_Futures_Close - US_Stock_CVE_adj_close',
    'JPX_Gold_Standard_Futures_Close - US_Stock_HL_adj_close',
    'LME_PB_Close - US_Stock_ALB_adj_close',
    'LME_CA_Close - US_Stock_X_adj_close',
    'US_Stock_DVN_adj_close - LME_PB_Close',
    'LME_AH_Close - US_Stock_OKE_adj_close',
    'US_Stock_OIH_adj_close - LME_CA_Close',
    'LME_PB_Close - US_Stock_TECK_adj_close'
]

# ----------------- MODELS -----------------
models = {
    "Ridge": Ridge(alpha=1.0),
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42),
}

# ----------------- FUNCTIONS -----------------
def safe_feature_engineering(df, features, target):
    available = [f for f in features if f in df.columns and f != target]
    df_out = df[available].copy()
    if " - " in target:
        a1, a2 = target.split(" - ")
        df_out = df_out.drop(columns=[c for c in (a1, a2) if c in df_out.columns], errors="ignore")
    return df_out

def run_ensemble(X_input, target):
    try:
        X_train = safe_feature_engineering(df, feature_columns, target)
        y_train = df[target].values

        predictions, weights, performances = [], [], []

        for name, model in models.items():
            model.fit(X_train, y_train)
            r2 = r2_score(y_train, model.predict(X_train))
            pred = model.predict(X_input[X_train.columns])[0]
            predictions.append(pred)
            weights.append(max(r2, 0.1))
            performances.append({"model": name, "prediction": pred, "r2": r2})

        weights = np.array(weights) / np.sum(weights)
        ensemble_pred = np.average(predictions, weights=weights)
        avg_r2 = np.mean([p["r2"] for p in performances])
        return ensemble_pred, performances, weights, avg_r2
    except Exception as e:
        return None, [], [], 0

def trade_suggestion(pred, a1, a2, avg_r2, threshold=0.6, min_conf=0.6):
    if avg_r2 < min_conf:
        return "⚠️ Model confidence too low — no trade suggestion."
    if abs(pred) < threshold:
        return "⏸️ No strong signal — hold / wait."
    if pred > 0:
        return f"📈 Spread expected to WIDEN → Consider LONG {a1}, SHORT {a2}"
    else:
        return f"📉 Spread expected to NARROW → Consider SHORT {a1}, LONG {a2}"

# ----------------- GRADIO INTERFACE -----------------
def predict_spread(*inputs, target):
    X_input = pd.DataFrame([inputs], columns=feature_columns)
    ensemble_pred, performances, weights, avg_r2 = run_ensemble(X_input, target)
    
    if ensemble_pred is None:
        return "Error in prediction. Check inputs."
    
    a1, a2 = target.split(" - ")
    suggestion = trade_suggestion(ensemble_pred, a1, a2, avg_r2)

    # Graph
    import matplotlib
    matplotlib.use("Agg")
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(df[target].values[-50:], label="Actual Spread (last 50)", color="blue")
    ax.axhline(y=ensemble_pred, color="red", linestyle="--", label="Predicted Spread")
    ax.legend()
    ax.set_title(f"Spread Trend: {a1} vs {a2}")
    ax.set_ylabel("Spread Value")

    return suggestion, fig

inputs = [gr.Number(value=float(df[f].iloc[-1]), label=f) for f in feature_columns]
inputs.append(gr.Dropdown(choices=target_columns, label="Target Spread"))

outputs = [gr.Textbox(label="Trade Suggestion"), gr.Plot(label="Spread Visualization")]

gr.Interface(fn=predict_spread, inputs=inputs, outputs=outputs, live=True, title="Commodity Spread Predictor").launch()
