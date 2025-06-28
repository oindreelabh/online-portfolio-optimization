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

def train_lstm_model(data, feature_col, target_col, lookback=10, epochs=10, batch_size=16, model_save_path="lstm_model.h5"):
    logger.info("Preparing data for LSTM...")
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data[[feature_col, target_col]])
    X, y = [], []
    for i in range(lookback, len(data_scaled)):
        X.append(data_scaled[i-lookback:i, 0])
        y.append(data_scaled[i, 1])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    logger.info("Building LSTM model...")
    model = Sequential([
        LSTM(50, input_shape=(X.shape[1], 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss=MeanSquaredError())
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1)
    model.save(model_save_path)
    logger.info(f"LSTM model trained and saved to {model_save_path}")
    joblib.dump(scaler, model_save_path.replace('.h5', '_scaler.pkl'))
    return model_save_path

def predict_lstm(model_path, scaler_path, recent_data, lookback=10):
    from tensorflow.keras.models import load_model
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    recent_scaled = scaler.transform(recent_data)[-lookback:, 0].reshape(1, lookback, 1)
    pred_scaled = model.predict(recent_scaled)
    # Inverse transform to get actual value
    dummy = np.zeros((1, scaler.n_features_in_))
    dummy[0, 1] = pred_scaled  # target_col is at index 1
    pred_actual = scaler.inverse_transform(dummy)[0, 1]
    return pred_actual

#main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and predict with LSTM model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to input data CSV")
    parser.add_argument("--feature_col", type=str, default="close", help="Feature column name")
    parser.add_argument("--target_col", type=str, default="returns", help="Target column name")
    parser.add_argument("--lookback", type=int, default=10, help="Lookback period for LSTM")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--model_save_path", type=str, default="lstm_model.h5", help="Path to save the trained model")

    args = parser.parse_args()

    data = pd.read_csv(args.data_path)
    train_lstm_model(data, args.feature_col, args.target_col, args.lookback, args.epochs, args.batch_size, args.model_save_path)
