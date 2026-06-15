Name:SUJITKUMAR.R
Company:Codec Technologies India
NCS ID: E19E86-0116588288923
Domain:Artificial Intelligence Intern 
Duration:09/06/2026 to 09/07/2026
Mentor:Dr. Anurag Shrivastava

Project overview-Build a tool that predicts future stock prices using historical data.
Helps in understanding market trends, investment risks, and financial forecasting.
🔄 Workflow
Data Collection-
Gather historical stock data (Open, Close, High, Low, Volume).
Sources: Yahoo Finance API, Kaggle datasets.

Preprocessing-
Handle missing values and outliers.
Normalize/scale data for better model performance.
Split into training and testing sets.

Feature Engineering-
Create features like moving averages, daily returns, volatility.
Use lagged values to capture time-series dependencies.

Model Building-
Basic: Linear Regression for trend prediction.
Machine Learning: Random Forest, Support Vector Regression.
Deep Learning: LSTM (Long Short-Term Memory) networks for sequential data.

Evaluation-
Metrics: Mean Squared Error (MSE), Root Mean Squared Error (RMSE), R² score.
Compare actual vs predicted prices using graphs.

Deployment-
Integrate into dashboards or apps for real-time stock monitoring.
Allow users to input a stock ticker and get predictions.

Key Highlights-
Practical Impact: Demonstrates how AI can assist in financial decision-making.
Scalable: Can expand from simple regression to advanced deep learning models.
Applications: Investment analysis, portfolio management, financial education.

Output -
Downloading data for AAPL...
Available columns: ['Open', 'High', 'Low', 'Close', 'Volume', 'lag_1', 'lag_2',
'lag_3', 'lag_4', 'lag_5', 'ma_5', 'ma_10', 'vol_5', 'target']
Training Linear Regression...
Linear Regression metrics: {'rmse': 4.86, 'mae': 3.91}, saved: models/AAPL_linear.joblib
Training Random Forest...
Random Forest metrics: {'rmse': 18.40, 'mae': 14.68}, saved: models/AAPL_rf.joblib
