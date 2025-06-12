from src.utils.logger import setup_logger
import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
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
    model.compile(optimizer='adam', loss='mse')
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
    dummy[0, 1] = pred_scaled  # Assuming target_col is at index 1
    pred_actual = scaler.inverse_transform(dummy)[0, 1]
    return pred_actual
