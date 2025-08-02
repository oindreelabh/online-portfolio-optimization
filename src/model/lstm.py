from src.utils.logger import setup_logger
import os
import numpy as np
import pandas as pd
import argparse
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import joblib

logger = setup_logger(os.path.basename(__file__).replace(".py", ""))

def train_lstm_model(data, target_col, lookback=5, epochs=10, batch_size=16, model_save_path="lstm_model.keras"):
    logger.info("Preparing data for LSTM...")

    feature_cols = [col for col in data.columns if col not in ['date', 'ticker', target_col]]
    all_cols = feature_cols + [target_col]

    # Checking for NaN values before scaling
    if data[all_cols].isna().any().any():
        logger.warning("Found NaN values in data. Filling with zeros.")
        data[all_cols] = data[all_cols].fillna(0)

    # Checking for infinite values
    if np.isinf(data[all_cols].values).any():
        logger.warning("Found infinite values in data. Replacing with large finite values.")
        data[all_cols] = data[all_cols].replace([np.inf, -np.inf], np.finfo(np.float64).max)

    # Scaling to handle outliers better
    scaler = MinMaxScaler(feature_range=(-1, 1))
    data_scaled = scaler.fit_transform(data[all_cols])

    X, y = [], []
    for i in range(lookback, len(data_scaled)):
        X.append(data_scaled[i-lookback:i, :-1])
        y.append(data_scaled[i, -1])
    X, y = np.array(X), np.array(y)

    logger.info("Building LSTM model...")
    model = Sequential([
        LSTM(50, input_shape=(X.shape[1], X.shape[2]),
             recurrent_dropout=0.2,  # Adding dropout to prevent overfitting
             return_sequences=False),
        Dense(1)
    ])

    # Using a smaller learning rate and adding gradient clipping
    optimizer = Adam(learning_rate=0.001, clipnorm=1.0)

    model.compile(optimizer=optimizer, loss=MeanSquaredError())

    # Adding early stopping to prevent wasting time if training becomes unstable
    early_stopping = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)

    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[early_stopping])
    model.save(model_save_path)
    logger.info(f"LSTM model trained and saved to {model_save_path}")
    joblib.dump({'scaler': scaler, 'feature_cols': feature_cols, 'target_col': target_col},
                model_save_path.replace('.keras', '_scaler.pkl'))
    return model_save_path

#main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and predict with LSTM model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to input data CSV")
    parser.add_argument("--target_col", type=str, default="close", help="Target column name")
    parser.add_argument("--lookback", type=int, default=5, help="Lookback period for LSTM")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--model_save_path", type=str, help="Path to save the trained model")

    args = parser.parse_args()

    data = pd.read_csv(args.data_path)
    train_lstm_model(data, args.target_col, args.lookback, args.epochs, args.batch_size, args.model_save_path)
