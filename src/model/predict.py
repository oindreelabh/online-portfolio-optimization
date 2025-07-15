import numpy as np
from keras.models import load_model
import pandas as pd
import joblib
import os
from src.utils.logger import setup_logger
from src.model.online_learning import OnlineGradientDescentMomentum

logger = setup_logger(os.path.basename(__file__).replace(".py", ""))

def load_lstm_model(model_path, scaler_path):
    model = load_model(model_path)
    scaler_dict = joblib.load(scaler_path)
    scaler = scaler_dict['scaler']
    feature_cols = scaler_dict['feature_cols']
    target_col = scaler_dict['target_col']
    return model, scaler, feature_cols, target_col

def predict_next(model, scaler, feature_cols, recent_data, sequence_length=10):
    # recent_data: DataFrame with latest `sequence_length` rows
    X_features = recent_data[feature_cols].values[-sequence_length:]
    # Stack dummy target column for scaling
    dummy_target = np.zeros((sequence_length, 1))
    recent_for_scaling = np.hstack([X_features, dummy_target])
    recent_scaled = scaler.transform(recent_for_scaling)[:, :-1]
    X = recent_scaled.reshape(1, sequence_length, len(feature_cols))
    pred_scaled = model.predict(X)
    # Inverse transform to get actual value
    dummy = np.zeros((1, len(feature_cols) + 1))
    dummy[0, :-1] = recent_scaled[-1]
    dummy[0, -1] = pred_scaled
    pred_actual = scaler.inverse_transform(dummy)[0, -1]
    return pred_actual

def load_online_model(model_path, n_features):
    model = OnlineGradientDescentMomentum(n_features=n_features, model_path=model_path)
    return model

def predict_online(model, feature_cols, recent_data):
    # Getting the expected number of features from the model
    expected_features = len(model.weights)
    actual_features = len(feature_cols)

    # Handling feature mismatch
    if expected_features != actual_features:
        logger.warning(f"Feature count mismatch: model expects {expected_features} features but data has {actual_features}")

        # Option 1: If model expects more features, pad with zeros
        if expected_features > actual_features:
            X = recent_data[feature_cols].values
            padding = np.zeros((X.shape[0], expected_features - actual_features))
            X = np.hstack([X, padding])
            logger.info(f"Padded features from {actual_features} to {expected_features}")
        # Option 2: If model expects fewer features, truncate
        else:
            X = recent_data[feature_cols].values[:, :expected_features]
            logger.info(f"Truncated features from {actual_features} to {expected_features}")
    else:
        X = recent_data[feature_cols].values

    preds = model.predict(X)
    return preds[-1]  # last prediction

def suggest_rebalance(predictions, current_allocations):
    # predictions: dict {ticker: predicted_return}
    # current_allocations: dict {ticker: current_weight}

    # Sort tickers by predicted return (highest first)
    sorted_tickers = sorted(predictions, key=predictions.get, reverse=True)
    logger.info(f"Sorted tickers by predicted returns: {sorted_tickers}")

    # Assign rank-based weights (higher predicted returns get higher weights)
    n = len(sorted_tickers)
    rank_weights = {ticker: (n - i) / (n * (n + 1) / 2) for i, ticker in enumerate(sorted_tickers)}

    # Limit how much we can change from current allocation (max 20% adjustment)
    max_change_pct = 0.20
    suggested = {}

    # Calculate suggested allocations based on both predictions and current weights
    for ticker in sorted_tickers:
        current = current_allocations.get(ticker, 0)
        # Target weight based on prediction rank
        target = rank_weights[ticker]

        # Adjust weight considering current allocation (blend of current and target)
        adjustment = (target - current) * max_change_pct
        suggested[ticker] = current + adjustment

    # Normalize to ensure weights sum to 1
    total_weight = sum(suggested.values())
    suggested = {t: w/total_weight for t, w in suggested.items()}

    return suggested

def hybrid_predict_and_rebalance(recent_data, lstm_model_path, lstm_scaler_path, online_model_path, tickers, current_allocations, sequence_length=10):
    lstm_model, scaler, feature_cols, target_col = load_lstm_model(lstm_model_path, lstm_scaler_path)
    online_model = load_online_model(online_model_path, n_features=len(feature_cols))
    hybrid_preds = {}
    for ticker in tickers:
        ticker_data = recent_data[recent_data['ticker'] == ticker]
        if len(ticker_data) < sequence_length:
            logger.info(f"Not enough data for {ticker} to make predictions. Skipping.")
            continue
        lstm_pred = predict_next(lstm_model, scaler, feature_cols, ticker_data, sequence_length)
        online_pred = predict_online(online_model, feature_cols, ticker_data)
        # Combine predictions using a simple average / can try weighted later
        hybrid_pred = (lstm_pred + online_pred) / 2
        hybrid_preds[ticker] = hybrid_pred
    suggested = suggest_rebalance(hybrid_preds, current_allocations)
    return suggested

# testing
recent_data = pd.read_csv("../../data/processed/stock_prices_latest.csv")
tickers = recent_data['ticker'].unique()
current_allocations = {'AAPL': 0.7, 'TSLA': 0.3}  # example
suggested = hybrid_predict_and_rebalance(
    recent_data,
    '../../models/lstm_model.keras',
    '../../models/lstm_model_scaler.pkl',
    '../../models/ogdm_model.pkl',
    tickers,
    current_allocations,
    sequence_length=5
)
print(suggested)
