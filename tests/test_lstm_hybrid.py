import os
import pandas as pd
import numpy as np
from src.model import lstm_hybrid

def test_train_and_predict_lstm(tmp_path):
    # Load data
    data_path = "data/processed/merged_data.csv"
    df = pd.read_csv(data_path)
    feature_col = "close"
    target_col = "returns"
    lookback = 3  # small for test speed

    # Drop NA for test
    df = df[[feature_col, target_col]].dropna().reset_index(drop=True)
    # Use a small subset for quick test
    df = df.iloc[:20]

    # Train model
    model_path = tmp_path / "test_lstm_model.h5"
    lstm_hybrid.train_lstm_model(
        df, feature_col, target_col,
        lookback=lookback, epochs=1, batch_size=2,
        model_save_path=str(model_path)
    )

    # Prepare recent data for prediction
    scaler_path = str(model_path).replace('.h5', '_scaler.pkl')
    recent_data = df[[feature_col, target_col]].values[-lookback:]

    # Predict
    pred = lstm_hybrid.predict_lstm(
        str(model_path), scaler_path, recent_data, lookback=lookback
    )

    # Check prediction is a float
    assert isinstance(pred, float) or isinstance(pred, np.floating)