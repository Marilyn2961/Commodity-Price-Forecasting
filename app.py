import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Spread Prediction Dashboard", 
    page_icon="📊", 
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    """Load and prepare data"""
    try:
        df = pd.read_csv("df_transformed.csv")
        return df
    except FileNotFoundError:
        st.error("❌ Data file 'df_transformed.csv' not found. Please ensure it's in the same directory.")
        return None

# Feature and target columns (same as our modeling)
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

# OUR ACTUAL TOP 3 MODELS FROM ENSEMBLE
models = {
    "Ridge": Ridge(alpha=1.0),
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
}

# Optimized weights based on our ensemble performance
weights = np.array([0.35, 0.35, 0.30])  # Ridge, LinearRegression, RandomForest
weights = weights / weights.sum()

# Safe feature engineering (same as our modeling)
def safe_feature_engineering(df, feature_columns, target):
    """Safe feature engineering without target leakage"""
    available_features = [f for f in feature_columns if f in df.columns and f != target]
    df_fe = df[available_features].copy()
    
    # Remove target components to prevent leakage
    if ' - ' in target:
        asset1, asset2 = target.split(' - ')
        cols_to_remove = [asset1, asset2]
        df_fe = df_fe.drop(columns=[col for col in cols_to_remove if col in df_fe.columns])
    
    return df_fe

# Ensemble prediction function
def ensemble_predict(X_input, target_idx=0, df=None):
    """Make ensemble prediction using our top 3 models"""
    if df is None:
        return None
    
    target = target_columns[target_idx]
    
    try:
        # Prepare features (same as training)
        X_train = safe_feature_engineering(df, feature_columns, target)
        y_train = df[target].values
        
        # Get predictions from each model
        preds_list = []
        model_performances = []
        
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            pred = model.predict(X_input[X_train.columns])[0]
            preds_list.append(pred)
            
            # Calculate model performance (R² on training data for demo)
            train_pred = model.predict(X_train)
            r2 = 1 - np.sum((y_train - train_pred) ** 2) / np.sum((y_train - np.mean(y_train)) ** 2)
            model_performances.append({
                'model': model_name,
                'prediction': pred,
                'r2_score': max(0, r2)  # Ensure non-negative
            })
        
        # Weighted ensemble prediction
        ensemble_pred = np.average(preds_list, axis=0, weights=weights)
        
        return ensemble_pred, model_performances, preds_list
        
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, [], []

# Main app
def main():
    st.markdown('<h1 class="main-header">📊 Spread Prediction Dashboard</h1>', unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Sidebar
    st.sidebar.header("🎯 Prediction Controls")
    
    # Target selection
    selected_target_idx = st.sidebar.selectbox(
        "Select Spread to Predict", 
        range(len(target_columns)), 
        format_func=lambda x: target_columns[x]
    )
    
    selected_target = target_columns[selected_target_idx]
    
    # Input method selection
    input_method = st.sidebar.radio(
        "Input Method",
        ["Manual Input", "Use Recent Data", "Upload CSV"]
    )
    
    input_df = None
    
    if input_method == "Manual Input":
        st.sidebar.subheader("📝 Manual Feature Input")
        input_data = {}
        
        # Group features for better organization
        col1, col2 = st.sidebar.columns(2)
        
        for i, col in enumerate(feature_columns):
            with col1 if i % 2 == 0 else col2:
                input_data[col] = st.slider(
                    col, 
                    float(df[col].min()), 
                    float(df[col].max()), 
                    float(df[col].mean()),
                    key=f"slider_{col}"
                )
        
        input_df = pd.DataFrame([input_data])
        
    elif input_method == "Use Recent Data":
        st.sidebar.subheader("📈 Use Recent Market Data")
        recent_idx = st.sidebar.slider(
            "Select recent data point", 
            0, len(df)-1, len(df)-1
        )
        input_df = df[feature_columns].iloc[[recent_idx]].copy()
        st.sidebar.info(f"Using data from index {recent_idx}")
        
    else:  # Upload CSV
        st.sidebar.subheader("📤 Upload Prediction Data")
        uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])
        if uploaded_file:
            try:
                uploaded_df = pd.read_csv(uploaded_file)
                # Check if required columns exist
                missing_cols = [col for col in feature_columns if col not in uploaded_df.columns]
                if missing_cols:
                    st.sidebar.error(f"Missing columns: {missing_cols}")
                else:
                    input_df = uploaded_df[feature_columns]
                    st.sidebar.success(f"Loaded {len(uploaded_df)} rows")
            except Exception as e:
                st.sidebar.error(f"Error reading file: {e}")
    
    # Main content
    if input_df is not None:
        # Make prediction
        ensemble_pred, model_performances, individual_preds = ensemble_predict(
            input_df, selected_target_idx, df
        )
        
        if ensemble_pred is not None:
            # Prediction results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                st.metric(
                    f"🎯 Ensemble Prediction", 
                    f"{ensemble_pred:.4f}",
                    delta="Ensemble (Weighted Average)"
                )
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.write("**Model Performance (R²)**")
                for perf in model_performances:
                    st.write(f"{perf['model']}: {perf['r2_score']:.3f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.write("**Individual Predictions**")
                for perf, pred in zip(model_performances, individual_preds):
                    st.write(f"{perf['model']}: {pred:.4f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Visualization
            st.subheader("📈 Historical Performance & Prediction")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Historical data plot
                fig1, ax1 = plt.subplots(figsize=(10, 6))
                historical_data = df[selected_target].tail(50)  # Last 50 points
                ax1.plot(historical_data.values, label='Historical Spread', marker='o', alpha=0.7)
                ax1.axhline(y=ensemble_pred, color='red', linestyle='--', 
                           label=f'Predicted: {ensemble_pred:.4f}', linewidth=2)
                ax1.set_xlabel('Time Index (Recent)')
                ax1.set_ylabel('Spread Value')
                ax1.set_title(f'Historical {selected_target}')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                st.pyplot(fig1)
            
            with col2:
                # Model comparison
                fig2, ax2 = plt.subplots(figsize=(10, 6))
                models_names = [perf['model'] for perf in model_performances]
                predictions = [perf['prediction'] for perf in model_performances]
                r2_scores = [perf['r2_score'] for perf in model_performances]
                
                bars = ax2.bar(models_names, predictions, alpha=0.7, 
                              color=['blue', 'green', 'orange'])
                ax2.axhline(y=ensemble_pred, color='red', linestyle='--', 
                           label=f'Ensemble: {ensemble_pred:.4f}', linewidth=2)
                
                # Add value labels on bars
                for bar, pred in zip(bars, predictions):
                    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                            f'{pred:.4f}', ha='center', va='bottom')
                
                ax2.set_ylabel('Prediction Value')
                ax2.set_title('Model Predictions Comparison')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                st.pyplot(fig2)
            
            # Data display
            st.subheader("📊 Input Data & Historical Values")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Your Input Features:**")
                st.dataframe(input_df.style.format("{:.4f}"), use_container_width=True)
            
            with col2:
                st.write("**Recent Historical Data:**")
                recent_data = df[feature_columns + [selected_target]].tail(10)
                st.dataframe(recent_data.style.format("{:.4f}"), use_container_width=True)
            
            # Download functionality
            st.subheader("💾 Download Results")
            
            # Create download dataframe
            download_df = input_df.copy()
            download_df[f'Predicted_{selected_target}'] = ensemble_pred
            
            for perf in model_performances:
                download_df[f'{perf["model"]}_Prediction'] = perf['prediction']
                download_df[f'{perf["model"]}_R2'] = perf['r2_score']
            
            csv = download_df.to_csv(index=False)
            
            st.download_button(
                label="📥 Download Prediction Results (CSV)",
                data=csv,
                file_name=f"spread_prediction_{selected_target.replace(' ', '_')}.csv",
                mime="text/csv",
                help="Download the prediction results as a CSV file"
            )
    
    else:
        # Welcome/instructions
        st.info("👈 Please select an input method from the sidebar to get started!")
        
        # Show dataset overview
        st.subheader("📁 Dataset Overview")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Rows", len(df))
        
        with col2:
            st.metric("Features", len(feature_columns))
        
        with col3:
            st.metric("Spread Targets", len(target_columns))
        
        # Show sample of recent data
        st.subheader("📈 Recent Market Data Sample")
        st.dataframe(df[feature_columns + target_columns[:5]].tail(10).style.format("{:.4f}"))

if __name__ == "__main__":
    main()