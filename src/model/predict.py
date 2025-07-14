import numpy as np
from keras.models import load_model
import pandas as pd
import joblib
from src.model.online_learning import OnlineGradientDescentMomentum

def load_lstm_model(model_path, scaler_path):
    model = load_model(model_path)
    scaler_dict = joblib.load(scaler_path)
    scaler = scaler_dict['scaler']
    feature_cols = scaler_dict['feature_cols']
    target_col = scaler_dict['target_col']
    return model, scaler, feature_cols, target_col

def predict_next(model, scaler, feature_cols, target_col, recent_data, sequence_length=10):
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
    X = recent_data[feature_cols].values
    preds = model.predict(X)
    return preds[-1]  # last prediction

def suggest_rebalance(predictions, current_allocations):
    # predictions: dict {ticker: predicted_return}
    # current_allocations: dict {ticker: current_weight}
    sorted_tickers = sorted(predictions, key=predictions.get, reverse=True)
    suggested = {t: 1.0/len(predictions) for t in sorted_tickers}  # equal weight as example
    return suggested

def hybrid_predict_and_rebalance(recent_data, lstm_model_path, lstm_scaler_path, online_model_path, tickers, current_allocations, sequence_length=10):
    lstm_model, scaler, feature_cols, target_col = load_lstm_model(lstm_model_path, lstm_scaler_path)
    online_model = load_online_model(online_model_path, n_features=len(feature_cols))
    hybrid_preds = {}
    for ticker in tickers:
        ticker_data = recent_data[recent_data['ticker'] == ticker]
        if len(ticker_data) < sequence_length:
            continue
        lstm_pred = predict_next(lstm_model, scaler, feature_cols, target_col, ticker_data, sequence_length)
        online_pred = predict_online(online_model, feature_cols, ticker_data)
        # Combine predictions (simple average, can be weighted)
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
    sequence_length=10
)
print(suggested)
