from src.utils.logger import setup_logger
import os
import numpy as np
import pandas as pd
import argparse
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.losses import MeanSquaredError
import joblib

logger = setup_logger(os.path.basename(__file__).replace(".py", ""))

def train_lstm_model(data, target_col, lookback=10, epochs=10, batch_size=16, model_save_path="lstm_model.keras"):
    logger.info("Preparing data for LSTM...")

    # Use all columns except 'date', 'ticker', and target as features
    feature_cols = [col for col in data.columns if col not in ['date', 'ticker', target_col]]
    all_cols = feature_cols + [target_col]

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data[all_cols])

    X, y = [], []
    for i in range(lookback, len(data_scaled)):
        X.append(data_scaled[i-lookback:i, :-1])  # all features
        y.append(data_scaled[i, -1])  # target
    X, y = np.array(X), np.array(y)
    # X shape: (samples, lookback, n_features)
    logger.info("Building LSTM model...")
    model = Sequential([
        LSTM(50, input_shape=(X.shape[1], X.shape[2])),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss=MeanSquaredError())
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1)
    model.save(model_save_path)
    logger.info(f"LSTM model trained and saved to {model_save_path}")
    joblib.dump({'scaler': scaler, 'feature_cols': feature_cols, 'target_col': target_col}, model_save_path.replace('.keras', '_scaler.pkl'))
    return model_save_path

def predict_lstm(model_path, scaler_path, recent_data, lookback=10):
    from tensorflow.keras.models import load_model
    model = load_model(model_path)
    scaler_dict = joblib.load(scaler_path)
    scaler = scaler_dict['scaler']
    feature_cols = scaler_dict['feature_cols']
    target_col = scaler_dict['target_col']

    # recent_data: DataFrame with same columns as original data
    recent_features = recent_data[feature_cols].values[-lookback:]
    # Stack dummy target column for scaling
    dummy_target = np.zeros((lookback, 1))
    recent_for_scaling = np.hstack([recent_features, dummy_target])
    recent_scaled = scaler.transform(recent_for_scaling)[:, :-1]
    X_pred = recent_scaled.reshape(1, lookback, len(feature_cols))
    pred_scaled = model.predict(X_pred)
    # Inverse transform to get actual value
    dummy = np.zeros((1, len(feature_cols) + 1))
    dummy[0, :-1] = recent_scaled[-1]
    dummy[0, -1] = pred_scaled
    pred_actual = scaler.inverse_transform(dummy)[0, -1]
    return pred_actual

#main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and predict with LSTM model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to input data CSV")
    parser.add_argument("--target_col", type=str, default="close", help="Target column name")
    parser.add_argument("--lookback", type=int, default=10, help="Lookback period for LSTM")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--model_save_path", type=str, help="Path to save the trained model")

    args = parser.parse_args()

    data = pd.read_csv(args.data_path)
    train_lstm_model(data, args.target_col, args.lookback, args.epochs, args.batch_size, args.model_save_path)
