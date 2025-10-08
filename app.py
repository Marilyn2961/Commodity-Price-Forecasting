import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import time

# ------------------ STYLING ------------------
st.set_page_config(
    page_title="Commodity Spread Predictor", 
    layout="wide"
)

st.markdown("""
<style>
    .main-header {
        font-size: 32px; font-weight: bold; color: #1f3d7a; text-align: center; margin-bottom: 10px;
    }
    .sub-header {
        font-size: 18px; color: #444; text-align: center; margin-bottom: 20px;
    }
    .section-header {
        font-size: 22px; font-weight: bold; color: #1f3d7a; margin-top: 20px; margin-bottom: 10px;
    }
    .signal-box {
        padding: 20px; border-radius: 8px; margin-top: 20px; color: white; font-weight: bold; text-align: center;
    }
    .signal-buy { background-color: #28a745; }
    .signal-sell { background-color: #dc3545; }
    .signal-hold { background-color: #ffc107; color: black; }
    .signal-wait { background-color: #6c757d; }
    .metric-card {
        background: white; 
        padding: 15px; 
        text-align: center;
        border-bottom: 3px solid #1f3d7a;
    }
    .metric-value {
        font-size: 24px; font-weight: bold; margin: 5px 0;
    }
    .metric-label {
        font-size: 14px; color: #666; margin-bottom: 5px; font-weight: 600;
    }
    .metric-subtext {
        font-size: 12px; color: #888; margin-top: 5px;
    }
    .bullish { color: #28a745; }
    .bearish { color: #dc3545; }
    .neutral { color: #6c757d; }
    .interactive-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 25px;
        border-radius: 12px;
        color: white;
        margin: 20px 0;
    }
    .strategy-card {
        background: white;
        padding: 20px;
        border-radius: 8px;
        border-left: 4px solid #1f3d7a;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .signal-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 12px;
        color: white;
        margin: 10px 0;
        text-align: center;
    }
    .strategy-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 20px;
        margin: 20px 0;
    }
    .risk-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 20px;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# ------------------ DATA LOADER ------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("df_transformed.csv")
        feature_cols = [c for c in df.columns if " - " not in c and np.issubdtype(df[c].dtype, np.number)]
        target_cols = [c for c in df.columns if " - " in c]
        
        if df.empty:
            st.error("Dataset is empty")
            return None, [], []
        if not target_cols:
            st.error("No target columns found (expected format: 'Asset1 - Asset2')")
            return None, [], []
            
        return df, feature_cols, target_cols
    except FileNotFoundError:
        st.error("Data file 'df_transformed.csv' not found. Please ensure it's in the correct directory.")
        return None, [], []
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, [], []

# ------------------ MODELING ------------------
def run_ensemble(df, X_input_df, target, feature_columns):
    try:
        X = df[feature_columns]
        y = df[target]

        models = [
            ("Linear Regression", LinearRegression()),
            ("Ridge Regression", Ridge(alpha=1.0)),
            ("Random Forest", RandomForestRegressor(n_estimators=100, random_state=42))
        ]

        predictions, performances, weights = [], [], []

        for i, (name, model) in enumerate(models):
            model.fit(X, y)
            pred = model.predict(X_input_df)[0]
            y_pred = model.predict(X)
            r2 = r2_score(y, y_pred)
            mae = mean_absolute_error(y, y_pred)

            predictions.append(pred)
            performances.append({"model": name, "prediction": pred, "r2": r2, "mae": mae})
            weights.append(max(r2, 0.001))

        weights = np.array(weights) / np.sum(weights)
        ensemble_pred = np.dot(predictions, weights)
        avg_r2 = np.mean([p["r2"] for p in performances])
        mae = np.mean([p["mae"] for p in performances])

        return ensemble_pred, performances, weights, avg_r2, mae
    
    except Exception as e:
        st.error(f"Model training failed: {e}")
        return None, [], [], 0, 0

def interpret_confidence(r2):
    if r2 >= 0.7: 
        return "High", "High reliability"
    elif r2 >= 0.4: 
        return "Medium", "Moderate reliability"
    else: 
        return "Low", "Low reliability"

# ------------------ TRADING LOGIC ------------------
def trade_suggestion_detailed(pred, a1, a2, r2, threshold, min_confidence):
    conf_level, conf_text = interpret_confidence(r2)

    if abs(pred) < threshold or r2 < min_confidence:
        return {
            "action": "WAIT / NO TRADE",
            "simple_action": "Hold Position",
            "confidence": conf_level,
            "confidence_text": conf_text,
            "reason": "Signal below threshold or low confidence",
            "sentiment": "neutral"
        }

    if pred > 0:
        return {
            "action": f"BUY {a1} / SELL {a2}",
            "simple_action": f"Long {a1}, Short {a2}",
            "confidence": conf_level,
            "confidence_text": conf_text,
            "reason": f"Spread expected to widen by {abs(pred):.3f}",
            "sentiment": "bullish"
        }
    else:
        return {
            "action": f"SELL {a1} / BUY {a2}",
            "simple_action": f"Short {a1}, Long {a2}",
            "confidence": conf_level,
            "confidence_text": conf_text,
            "reason": f"Spread expected to narrow by {abs(pred):.3f}",
            "sentiment": "bearish"
        }

# ------------------ MAIN APP ------------------
def main():
    st.markdown('<div class="main-header">Commodity Spread Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Analyze and predict price relationships between commodities and equities</div>', unsafe_allow_html=True)
    st.markdown("---")

    # Load data with loading state
    with st.spinner("Loading dataset..."):
        df, feature_columns, target_columns = load_data()
    
    if df is None:
        st.info("To use this app, please ensure you have a 'df_transformed.csv' file in the same directory.")
        return

    # -------- SIDEBAR --------
    st.sidebar.header("Analysis Setup")

    # Asset pair selection
    display_targets = [t.replace('_', ' ') for t in target_columns]
    target_dict = dict(zip(display_targets, target_columns))
    
    selected_display = st.sidebar.selectbox("Select asset pair:", display_targets)
    target = target_dict[selected_display]

    # Model parameters
    st.sidebar.subheader("Model Parameters")
    
    threshold = st.sidebar.slider(
        "Signal Threshold", 
        0.1, 1.0, 0.6, 0.1,
        help="Minimum predicted change magnitude to generate a trading signal"
    )
    
    min_confidence = st.sidebar.slider(
        "Minimum Confidence", 
        0.1, 1.0, 0.6, 0.1,
        help="Minimum R² score required for reliable predictions"
    )

    # Price inputs with reset functionality
    with st.sidebar.expander("Price Inputs", expanded=True):
        st.caption("Adjust prices for prediction (default = last available dataset value)")
        
        if 'user_inputs' not in st.session_state:
            st.session_state.user_inputs = {}
        
        if st.button("Reset All to Default", use_container_width=True):
            st.session_state.user_inputs = {}
            st.rerun()
        
        for f in feature_columns:
            current_val = float(df[f].iloc[-1])
            col1, col2 = st.columns([3, 1])
            with col1:
                user_input = st.number_input(
                    f"{f.replace('_', ' ')}",
                    value=st.session_state.user_inputs.get(f, current_val),
                    step=0.1,
                    format="%.2f",
                    key=f"input_{f}",
                    help=f"Current value: {current_val:.2f}"
                )
            with col2:
                if st.button("↺", key=f"reset_{f}", help="Reset to default"):
                    st.session_state.user_inputs[f] = current_val
                    st.rerun()
            
            st.session_state.user_inputs[f] = user_input

    X_input = st.session_state.user_inputs
    X_input_df = pd.DataFrame([X_input])

    # -------- MAIN CONTENT --------
    if st.button("Run Analysis", type="primary", use_container_width=True):
        
        with st.spinner("Training models and generating predictions..."):
            ensemble_pred, performances, weights, avg_r2, mae = run_ensemble(
                df, X_input_df, target, feature_columns
            )

        if ensemble_pred is not None:
            a1, a2 = target.split(" - ")
            display_a1, display_a2 = a1.replace('_', ' '), a2.replace('_', ' ')

            # Prediction Overview - with directional arrows
            st.markdown('<div class="section-header">Prediction Overview</div>', unsafe_allow_html=True)

            conf_level, conf_text = interpret_confidence(avg_r2)
            signal_str = "Strong" if abs(ensemble_pred) > threshold else "Weak"
            sentiment = "bullish" if ensemble_pred > 0 else "bearish" if ensemble_pred < 0 else "neutral"

            # Define arrow symbols and colors
            if ensemble_pred > 0:
                arrow_symbol = "↗️"
                arrow_color = "#28a745"
                direction_text = "Widening"
            elif ensemble_pred < 0:
                arrow_symbol = "↘️"
                arrow_color = "#dc3545"
                direction_text = "Narrowing"
            else:
                arrow_symbol = "➡️"
                arrow_color = "#6c757d"
                direction_text = "Neutral"

            strength_pct = min(abs(ensemble_pred) / threshold, 1.0)

            mcol1, mcol2, mcol3, mcol4 = st.columns(4)

            with mcol1:
                # Predicted Change with arrow
                st.markdown(f'<div class="metric-card">', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-label">Predicted Change</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value" style="color: {arrow_color}; font-size: 28px;">{arrow_symbol} {ensemble_pred:.3f}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-subtext">{direction_text} • {sentiment.capitalize()}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with mcol2:
                # Confidence Level with indicator
                confidence_color = "#28a745" if conf_level == "High" else "#ffc107" if conf_level == "Medium" else "#dc3545"
                confidence_icon = "🔴" if conf_level == "Low" else "🟡" if conf_level == "Medium" else "🟢"
                st.markdown(f'<div class="metric-card">', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-label">Confidence Level</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value" style="color: {confidence_color}; font-size: 28px;">{confidence_icon} {conf_level}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-subtext">R²: {avg_r2:.3f} — {conf_text}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with mcol3:
                # Signal Strength
                strength_color = "#28a745" if strength_pct > 0.7 else "#ffc107" if strength_pct > 0.4 else "#dc3545"
                strength_icon = "👎" if strength_pct <= 0.4 else "👌" if strength_pct <= 0.7 else "💪"
                st.markdown(f'<div class="metric-card">', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-label">Signal Strength</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value" style="color: {strength_color}; font-size: 28px;">{strength_icon} {int(strength_pct * 100)}%</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-subtext">{strength_pct:.0%} of threshold • Threshold: {threshold}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with mcol4:
                # Model Fit with indicator
                fit_color = "#28a745" if avg_r2 >= 0.7 else "#ffc107" if avg_r2 >= 0.4 else "#dc3545"
                fit_icon = "📉" if avg_r2 < 0.4 else "📈" if avg_r2 < 0.7 else "📊"
                fit_text = "Good Fit" if avg_r2 >= 0.7 else "Moderate Fit" if avg_r2 >= 0.4 else "Poor Fit"
                st.markdown(f'<div class="metric-card">', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-label">Model Fit (R²)</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value" style="color: {fit_color}; font-size: 28px;">{fit_icon} {avg_r2:.3f}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-subtext">{fit_text}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # Trading Signal - Interactive Design
            st.markdown('<div class="section-header">Trading Signal</div>', unsafe_allow_html=True)
            
            suggestion = trade_suggestion_detailed(
                ensemble_pred, display_a1, display_a2, avg_r2, threshold, min_confidence
            )

            # Main Signal Display
            st.markdown(f'''
            <div class="signal-card">
                <h2 style="margin: 0; font-size: 32px; color: white;">{suggestion['action']}</h2>
                <p style="font-size: 18px; margin: 10px 0; color: white; opacity: 0.9;">{suggestion['simple_action']}</p>
            </div>
            ''', unsafe_allow_html=True)

            # Signal Details in Columns
            col_detail1, col_detail2 = st.columns(2)
            with col_detail1:
                st.markdown("**Confidence Level**")
                st.info(f"**{suggestion['confidence']}** - {suggestion['confidence_text']}")
            
            with col_detail2:
                st.markdown("**Signal Reason**")
                st.warning(f"{suggestion['reason']}")

            # Quick Action Buttons
            if suggestion['sentiment'] in ["bullish", "bearish"]:
                st.markdown("---")
                action_col1, action_col2, action_col3 = st.columns(3)
                with action_col1:
                    if st.button(" View Trading Strategy", use_container_width=True):
                        st.session_state.active_tab = "Trading Strategy"
                with action_col2:
                    if st.button(" Analyze Price Charts", use_container_width=True):
                        st.session_state.active_tab = "Price Charts"
                with action_col3:
                    if st.button(" Download Analysis Report", use_container_width=True):
                        st.session_state.download_report = True

            # Detailed Analysis Tabs
            st.markdown('<div class="section-header">Detailed Analysis</div>', unsafe_allow_html=True)
            
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                " Trading Strategy", 
                " Price Charts", 
                " Model Performance", 
                " Input Summary",
                " Methodology"
            ])

            # ------------------ Trading Strategy (Interactive Version) ------------------
            with tab1:
                st.subheader("Trading Strategy Guide")
                
                # Current Signal Analysis - Interactive Layout
                st.markdown("###  Current Signal Analysis")
                
                col_signal1, col_signal2, col_signal3 = st.columns(3)
                
                with col_signal1:
                    st.markdown("**Spread Direction**")
                    if direction_text == "Widening":
                        st.success(f"**{direction_text}**")
                    else:
                        st.error(f"**{direction_text}**")
                    st.markdown(f"*{sentiment.capitalize()} sentiment*")
                
                with col_signal2:
                    st.markdown("**Recommended Action**")
                    if suggestion['action'] == "WAIT / NO TRADE":
                        st.warning(f"**{suggestion['action']}**")
                    else:
                        st.success(f"**{suggestion['action']}**")
                    st.markdown(f"*{suggestion['simple_action']}*")
                
                with col_signal3:
                    st.markdown("**Signal Quality**")
                    if signal_str == "Strong":
                        st.success(f"**{signal_str} Signal**")
                    else:
                        st.warning(f"**{signal_str} Signal**")
                    st.markdown(f"*{suggestion['confidence']} Confidence*")

                st.markdown("---")

                # Trading Strategies - Interactive Grid
                st.markdown("###  Trading Strategies")
                
                col_strat1, col_strat2 = st.columns(2)
                
                with col_strat1:
                    with st.expander("🟢 LONG SPREAD STRATEGY", expanded=True):
                        st.markdown("""
                        **Action**: Buy First Asset / Sell Second Asset  
                        **When**: Spread expected to widen  
                        **Expectation**: First asset outperforms second  
                        **Profit from**:
                        - First asset rises more than second
                        - First asset falls less than second
                        """)
                
                with col_strat2:
                    with st.expander("🔴 SHORT SPREAD STRATEGY", expanded=True):
                        st.markdown("""
                        **Action**: Sell First Asset / Buy Second Asset  
                        **When**: Spread expected to narrow  
                        **Expectation**: Second asset outperforms first  
                        **Profit from**:
                        - Second asset rises more than first
                        - Second asset falls less than first
                        """)

                st.markdown("---")

                # Risk Management - Interactive Sections
                st.markdown("###  Risk Management")
                
                risk_col1, risk_col2 = st.columns(2)
                
                with risk_col1:
                    with st.expander("POSITION MANAGEMENT", expanded=True):
                        st.markdown("""
                        - Allocate 1-2% of portfolio per trade
                        - Maximum 5% in correlated spreads
                        - Adjust based on volatility
                        - Set stop-loss at 1.5x predicted move
                        """)
                
                with risk_col2:
                    with st.expander("PORTFOLIO CONSIDERATIONS", expanded=True):
                        st.markdown("""
                        - Monitor asset correlations
                        - Diversify across uncorrelated spreads
                        - Consider transaction costs
                        - Review macroeconomic factors
                        - Past performance ≠ future results
                        """)

            # ------------------ Price Charts ------------------
            with tab2:
                st.subheader("Price Chart Analysis")
                
                history_window = st.radio("Chart Range", ["Last 50", "Last 100", "Full"], horizontal=True)
                
                if history_window == "Last 50":
                    historical_data = df[target].values[-50:]
                    x_range = range(len(historical_data))
                elif history_window == "Last 100":
                    historical_data = df[target].values[-100:]
                    x_range = range(len(historical_data))
                else:
                    historical_data = df[target].values
                    x_range = range(len(historical_data))

                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(x_range, historical_data, label="Historical Spread", color="#1f3d7a", linewidth=2)
                ax.axhline(y=ensemble_pred, color="#d9534f", linestyle="--", linewidth=2,
                           label=f"Predicted: {ensemble_pred:.3f}")
                
                current_val = historical_data[-1]
                ax.axhline(y=current_val, color="#28a745", linestyle=":", linewidth=1.5,
                           label=f"Current: {current_val:.3f}")
                
                ax.legend()
                ax.set_title(f"Spread Trend: {display_a1} vs {display_a2}", fontsize=14, fontweight='bold')
                ax.set_xlabel("Time Period")
                ax.set_ylabel("Spread Value")
                ax.grid(True, alpha=0.3)
                
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                
                st.pyplot(fig)

            # ------------------ Model Performance ------------------
            with tab3:
                st.subheader("Model Performance Details")
                
                perf_df = pd.DataFrame(performances)
                perf_df['Weight'] = weights
                perf_df['Weighted Contribution'] = perf_df['prediction'] * perf_df['Weight']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Ensemble R² Score", f"{avg_r2:.4f}")
                    st.metric("Mean Absolute Error", f"{mae:.4f}")
                
                with col2:
                    st.metric("Number of Models", len(performances))
                    st.metric("Ensemble Prediction", f"{ensemble_pred:.4f}")
                
                st.dataframe(
                    perf_df.style.format({
                        'prediction': '{:.4f}',
                        'r2': '{:.4f}', 
                        'mae': '{:.4f}',
                        'Weight': '{:.2%}',
                        'Weighted Contribution': '{:.4f}'
                    }).background_gradient(subset=['r2'], cmap='Blues'),
                    use_container_width=True
                )

            # ------------------ Input Summary ------------------
            with tab4:
                st.subheader("Feature Inputs Summary")
                
                inputs_df = pd.DataFrame({
                    "Feature": [f.replace('_', ' ') for f in X_input.keys()],
                    "Input Value": X_input.values(),
                    "Default Value": [float(df[f].iloc[-1]) for f in feature_columns],
                    "Difference": [X_input[f] - float(df[f].iloc[-1]) for f in feature_columns]
                })
                
                st.dataframe(
                    inputs_df.style.format({
                        'Input Value': '{:.2f}',
                        'Default Value': '{:.2f}',
                        'Difference': '{:.2f}'
                    }).applymap(
                        lambda x: 'color: red' if isinstance(x, (int, float)) and x < 0 else 'color: green', 
                        subset=['Difference']
                    ),
                    use_container_width=True
                )
                
                st.subheader("Feature Impact Analysis")
                try:
                    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
                    rf_model.fit(df[feature_columns], df[target])
                    feature_importance = pd.DataFrame({
                        'feature': [f.replace('_', ' ') for f in feature_columns],
                        'importance': rf_model.feature_importances_
                    }).sort_values('importance', ascending=True)
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    y_pos = np.arange(len(feature_importance))
                    ax.barh(y_pos, feature_importance['importance'], color='#1f3d7a', alpha=0.7)
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels(feature_importance['feature'])
                    ax.set_xlabel('Feature Importance')
                    ax.set_title('Relative Feature Importance in Prediction')
                    ax.grid(True, alpha=0.3, axis='x')
                    st.pyplot(fig)
                except Exception as e:
                    st.info("Feature importance visualization not available for current data")

            # ------------------ Methodology ------------------
            with tab5:
                st.subheader("Methodology Explanation")
                
                st.markdown("""
                ### Model Configuration
                
                **Signal Threshold = 0.6 and Minimum Confidence ≥ 0.6**
                
                We deploy only the best performing ensemble models. To ensure meaningful trading signals:
                
                - **Threshold = 0.6**: Filters out weak/noisy signals, focusing only on significant predicted moves
                - **Confidence ≥ 0.6**: Ensures model reliability before suggesting trades
                """)
                
                perf_comparison = pd.DataFrame({
                    "Model Type": ["Ensemble-Mean-Top3", "Ensemble-Weighted-Top3"],
                    "MAE": [0.0172, 0.0162],
                    "RMSE": [0.0247, 0.0228],
                    "R²": [0.6716, 0.7227],
                    "Directional Accuracy": ["78.40%", "80.28%"]
                })
                
                st.dataframe(
                    perf_comparison.style.highlight_max(axis=0, color='#d4edda'),
                    use_container_width=True
                )
                
                st.markdown("""
                ### Technical Approach
                
                **Ensemble Modeling**: Combines multiple algorithms for robust predictions:
                - Linear Regression: Captures linear relationships
                - Ridge Regression: Handles multicollinearity  
                - Random Forest: Models complex non-linear patterns
                
                **Weighted Averaging**: Models are weighted by their R² performance to create the final ensemble prediction.
                """)

            # Download Results
            st.markdown("---")
            col1, col2 = st.columns([1, 3])
            with col1:
                results_df = pd.DataFrame([{
                    "Target": target,
                    "Predicted_Change": ensemble_pred,
                    "Confidence_Level": conf_level,
                    "Signal_Strength": signal_str,
                    "R2_Score": avg_r2,
                    "MAE": mae,
                    "Trading_Signal": suggestion['action'],
                    "Timestamp": pd.Timestamp.now()
                }])
                
                st.download_button(
                    "Download Results (CSV)", 
                    results_df.to_csv(index=False), 
                    "trading_signal_results.csv",
                    help="Download detailed analysis results"
                )

    else:
        st.info("Configure your analysis in the sidebar and click 'Run Analysis' to generate trading signals")

    # -------- FOOTER --------
    st.markdown("---")
    
    with st.expander("Risk Disclaimer", expanded=False):
        st.markdown("""
        **Important Notice**: This tool provides algorithmic trading suggestions for educational and research purposes only. 
        
        - Past performance does not guarantee future results
        - Always conduct your own research and validation
        - Consider consulting with a qualified financial advisor
        - Understand all risks before trading
        - The developers are not responsible for trading decisions or losses
        """)

# ------------------ RUN APP ------------------
if __name__ == "__main__":
    main()