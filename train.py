import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

from .data import download_data, create_features, train_test_split_timeseries
from .models import build_linear_regression, build_random_forest, save_model


def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, preds)
    return {'rmse': rmse, 'mae': mae}


def main():
    parser = argparse.ArgumentParser(description='Train stock price predictor')
    parser.add_argument('--ticker', required=True, help='Ticker symbol, e.g. AAPL')
    parser.add_argument('--period', default='5y', help='History period for yfinance')
    parser.add_argument('--interval', default='1d', help='Data interval')
    parser.add_argument('--lags', type=int, default=5, help='Number of lag features')
    parser.add_argument('--test-size', type=float, default=0.2, help='Fraction for test split')
    args = parser.parse_args()

    ticker = args.ticker
    print(f'Downloading data for {ticker}...')
    df = download_data(ticker, period=args.period, interval=args.interval)
    data = create_features(df, lags=args.lags)

    # Flatten column names if they are tuples
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [c[0] if isinstance(c, tuple) else c for c in data.columns]
    elif any(isinstance(c, tuple) for c in data.columns):
        data.columns = [c[0] if isinstance(c, tuple) else c for c in data.columns]

    print(f"Available columns: {list(data.columns)}")
    
    # Prepare features and target - filter to columns that exist
    lag_cols = [c for c in data.columns if 'lag_' in str(c)]
    ma_cols = [c for c in data.columns if any(x in str(c) for x in ['ma_5', 'ma_10'])]
    vol_cols = [c for c in data.columns if 'vol_' in str(c)]
    feature_cols = lag_cols + ma_cols + vol_cols
    
    if not feature_cols:
        print(f"Warning: No feature columns found. Available columns: {list(data.columns)}")
        feature_cols = [c for c in data.columns if c not in ['Close', 'target', 'Open', 'High', 'Low', 'Volume']]
    X = data[feature_cols].values
    y = data['target'].values

    train, test = train_test_split_timeseries(data, test_size=args.test_size)
    X_train = train[feature_cols].values
    y_train = train['target'].values
    X_test = test[feature_cols].values
    y_test = test['target'].values

    os.makedirs('models', exist_ok=True)

    # Linear Regression
    name, lr = build_linear_regression()
    print('Training Linear Regression...')
    lr.fit(X_train, y_train)
    metrics_lr = evaluate_model(lr, X_test, y_test)
    path_lr = f'models/{ticker}_{name}.joblib'
    save_model(lr, path_lr)
    print(f'Linear Regression metrics: {metrics_lr}, saved: {path_lr}')

    # Random Forest
    name, rf = build_random_forest()
    print('Training Random Forest...')
    rf.fit(X_train, y_train)
    metrics_rf = evaluate_model(rf, X_test, y_test)
    path_rf = f'models/{ticker}_{name}.joblib'
    save_model(rf, path_rf)
    print(f'Random Forest metrics: {metrics_rf}, saved: {path_rf}')

    print('Done.')


if __name__ == '__main__':
    main()
