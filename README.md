# Commodity Price Forecasting Using Multi-Market Financial Time Series Data

![Commodity Price Forecasting](images/project%20banner.png)

## Project Overview
Global commodity markets are influenced by multiple, interconnected factors such as stock indices, futures contracts, and foreign exchange rates. Forecasting commodity returns is a non-trivial task due to high volatility, lagged dependencies, and complex inter-market relationships.

This project leverages historical data from diverse sources—including the London Metal Exchange (LME), Japan Exchange Group (JPX), and U.S. stock markets—to construct predictive models for commodity returns and spreads. The approach incorporates lagged features to capture the influence of past price movements on future values, enabling data-driven insights for traders, investors, and organizations.

## Project Objectives
1. To preprocess and integrate multi-market financial time series data for commodities.  
2. To explore and analyze patterns, correlations, and volatility in commodity returns.  
3. To develop predictive models for selected commodity prices and spreads.  
4. To evaluate model accuracy and stability using appropriate performance metrics.  
5. To generate insights that can support trading strategies and risk management.  

## Methodology

### Data Source
The datasets came from a [kaggle competition](https://www.kaggle.com/competitions/mitsui-commodity-prediction-challenge/data). In this project we used the train.csv, label_pairs and train_labels.csv 

### Exploratory Data Analysis (EDA)
- Correlation heatmaps across different markets. 
- Checked for the distribution of features and targets. 
- Distribution analysis of returns and spreads to notice outliers.  
- Target trends over time.

### Data Preprocessing and Future Engineering
- Stationary check
- Spread calculations to create spread columns. These help us see how one asset is performing compared to another.
- Create lag features for our dataset while avoiding adata leakage

### Modelling
- **Baseline Models**: Ridge & Lasso Regression, Loinear Regression, Random Forest LightGBM, XGboost.  
- **Advanced Models**: Gradient Boosting, ElasticNet, SVR, MLP Regressor.  
- Implemented **walk-forward evaluation** to simulate real-time forecasting.  
![Future Importance](images/Future%20Importance.png)

### Evaluation Metrics
Models were evaluated using multiple complementary metrics:  
- **Mean Absolute Error (MAE)** – Measures average prediction error.  
- **Root Mean Squared Error (RMSE)** – Penalizes large deviations.  
- **R-squared (R²)** – Variance explained by the model.  
- **Directional Accuracy** – Proportion of correctly predicted price directions. 
- **Mean Absolute Persentage Error (MAPE)** - It measures the average percentage error between actual values and predicted values 
- **Symmetric Mean Absolute Percentage Error (SMAPE)** - A symmetric version of MAPE that avoids extreme percentages when actual values are near zero. 
![Model Perfomance Comparison](images/Model%20performance%20comparison.png)

### Why Weighted Enssemble Methods was Choosen
- It captures both linear and non-linear patterns in the data by combining Ridge, Linear Regression, and RandomForest models.  
- It consistently outperforms individual models across MAE, RMSE, R², and directional accuracy.  
- The ensemble provides more stable predictions, reducing sensitivity to outliers or market volatility.  
- Feature importance and SHAP analyses show that the ensemble effectively leverages the strongest predictors, particularly key energy sector stocks.

Conclusion: Using a weighted ensemble ensures higher predictive accuracy, robustness, and interpretability, making it the optimal choice for short-term commodity spread forecasting.  
![Actual vs Predicted](images/Actual%20vs%20Predicted.png)

## Recommendation
1. **Go with the Ensemble-Weighted-Top3 Model** - This is because it gave us the most consistent results across accuracy, error reduction, and direction of price movement.  
2. **Energy Sector Features Matter Most** - Stocks in the energy sector (like **OIH**, **HAL**, and **CVE**) turned out to be the strongest drivers of spread predictions.  
3. **Markets Don’t Move in Isolation** - Relationships between commodities, stocks, and futures added a lot of predictive power.  
  
## Limitations
1. **Market Regimes Change**  
   - The model is tuned for today’s conditions but markets evolve are caused by varying factors.  
   - Over-reliance on energy features might reduce accuracy in tech-led or defensive markets.  
3. **Validation Gaps**  
   - Backtests are based on past data true future-proofing is untested.  
   - Assumes markets remain liquid execution could fail during crises.  

## Visualizations
[Tableau Dashboard](https://public.tableau.com/app/profile/rodgers.otieno/viz/Phase5_17595776062790/Group5DashBoard?publish=yes)

## Deployed Model
- We deployed the [weighted ensemble method](https://commodity-price-forecasting.streamlit.app/) as a price forecasting app

## Conclusion
This project demonstrates the feasibility of leveraging multi-market time series data for forecasting commodity returns. By integrating lagged effects and advanced ensemble methods, the models achieved meaningful predictive accuracy, offering practical utility for risk management and trading strategy optimization.

## Future Work
1. **Strengthen Validation Testing**  
   - Use rolling “walk-forward” validation to mimic real-world trading.  
   - Test the models during big market swings to make sure they can handle stress.  
   - Include transaction costs and slippage so predictions match real-world trading outcomes.  
2. **Smarter Feature Engineering**  
   - Add more market indicators like volatility and momentum.  
   - Detect “market regimes” for example  calm vs. volatile periods to make the model adapt.  
   - Use cross-market correlations to capture how different assets influence each other.  
3. **Build a Monitoring System**  
   - Set up alerts for when performance drops, so retraining happens automatically.   

