import os

# Set Streamlit's folder to a writable temp location
os.environ["STREAMLIT_HOME"] = "/tmp/.streamlit"
# Make sure the folder exists
os.makedirs("/tmp/.streamlit", exist_ok=True)

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings("ignore")

# ----------------- PAGE CONFIG -----------------
st.set_page_config(page_title="Commodity Spread Predictor", layout="wide")

# ----------------- DATA LOADING -----------------
@st.cache_data
def load_data():
    try:
        return pd.read_csv("df_transformed.csv")
    except FileNotFoundError:
        st.error("Data file not found. Please make sure 'df_transformed.csv' is in the same folder.")
        return None

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

def run_ensemble(df, X_input, target):
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
    except Exception:
        return None, [], [], 0

def trade_suggestion(pred, a1, a2, avg_r2, threshold=0.6, min_conf=0.6):
    """Turn model prediction into trade suggestion with fixed thresholds"""
    if avg_r2 < min_conf:
        return "⚠️ Model confidence too low — no trade suggestion."

    if abs(pred) < threshold:
        return "⏸️ No strong signal — hold / wait."

    if pred > 0:
        return f"📈 Spread expected to WIDEN → Consider LONG {a1}, SHORT {a2}"
    else:
        return f"📉 Spread expected to NARROW → Consider SHORT {a1}, LONG {a2}"

# ----------------- TRADING STRATEGY GUIDE -----------------
def trading_strategy_guide():
    st.sidebar.markdown("---")
    st.sidebar.subheader("📈 Trading Strategy Guide")
    
    st.sidebar.markdown("""
    **Strategy Rules:**
    
    ✅ **ENTER LONG Spread** when:
    - Predicted change > +0.6
    - Confidence R² > 0.6
    - Buy Asset1, Short Asset2
    
    ✅ **ENTER SHORT Spread** when:
    - Predicted change < -0.6  
    - Confidence R² > 0.6
    - Short Asset1, Buy Asset2
    
    ⏸️ **HOLD** when:
    - Signal strength < 0.6
    - Confidence R² < 0.6
    
    **Risk Management:**
    - Monitor OIH, OXY, CVE for confirmation
    - Use stop-loss on individual legs
    - Re-assess daily with new predictions
    """)

# ----------------- MAIN APP -----------------
def main():
    st.title("Commodity Spread Predictor")

    df = load_data()
    if df is None:
        return

    # Sidebar
    st.sidebar.header("Controls")
    target = st.sidebar.selectbox("Choose Spread", target_columns)

    # Manual input
    X_input = {}
    for f in feature_columns:
        X_input[f] = st.sidebar.number_input(f, value=float(df[f].iloc[-1]), step=1.0)
    X_input = pd.DataFrame([X_input])

    # Add trading strategy guide to sidebar
    trading_strategy_guide()

    # Run prediction
    ensemble_pred, performances, weights, avg_r2 = run_ensemble(df, X_input, target)

    if ensemble_pred is not None:
        a1, a2 = target.split(" - ")
        st.subheader("Prediction Result")
        st.write(f"Spread: **{a1} vs {a2}**")
        st.write(f"Predicted change: {ensemble_pred:.4f} | Confidence (avg R²): {avg_r2:.2f}")

        if ensemble_pred > 0:
            st.success(f"Spread expected to WIDEN ⬆️ → {a1} stronger than {a2}")
        else:
            st.error(f"Spread expected to NARROW ⬇️ → {a1} weaker than {a2}")

        # Trade Suggestion
        st.subheader("Trade Suggestion")
        suggestion = trade_suggestion(ensemble_pred, a1, a2, avg_r2)
        if "LONG" in suggestion:
            st.success(suggestion)
        elif "SHORT" in suggestion:
            st.error(suggestion)
        else:
            st.info(suggestion)

        # Show model contributions
        st.subheader("Model Contributions")
        for perf, w in zip(performances, weights):
            st.write(f"{perf['model']}: {perf['prediction']:.4f} (weight {w:.0%}, R²={perf['r2']:.2f})")

        # Show input data
        st.subheader("Input Data Used")
        st.dataframe(X_input.T)

        # ----------------- GRAPH -----------------
        st.subheader("Spread Visualization")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df[target].values[-50:], label="Actual Spread (last 50)", color="blue")
        ax.axhline(y=ensemble_pred, color="red", linestyle="--", label="Predicted Spread")
        ax.legend()
        ax.set_title(f"Spread Trend: {a1} vs {a2}")
        ax.set_ylabel("Spread Value")
        st.pyplot(fig)

    else:
        st.info("Select a spread and input values to see predictions.")

    # Risk Disclaimer
    st.markdown("---")
    st.markdown("""
    ### ⚠️ Risk Disclaimer
    *This tool provides algorithmic trading suggestions for educational purposes only. 
    Past performance does not guarantee future results. Always conduct your own research 
    and consider consulting with a qualified financial advisor before making investment decisions.*
    """)

if __name__ == "__main__":
    main()