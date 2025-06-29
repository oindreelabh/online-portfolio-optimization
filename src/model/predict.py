import numpy as np
from keras.models import load_model
import pandas as pd
import joblib

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

def suggest_rebalance(predictions, current_allocations):
    # predictions: dict {ticker: predicted_return}
    # current_allocations: dict {ticker: current_weight}
    sorted_tickers = sorted(predictions, key=predictions.get, reverse=True)
    suggested = {t: 1.0/len(predictions) for t in sorted_tickers}  # equal weight as example
    return suggested

# testing
recent_data = pd.read_csv("../../data/processed/stock_prices_latest.csv")
model, scaler, feature_cols, target_col = load_lstm_model('../../models/lstm_model.keras', '../../models/lstm_model_scaler.pkl')
pred = predict_next(model, scaler, feature_cols, target_col, recent_data)
rebalance = suggest_rebalance({'AAPL': 0.02, 'TSLA': 0.01}, {'AAPL': 0.7, 'TSLA': 0.3})
print(rebalance)