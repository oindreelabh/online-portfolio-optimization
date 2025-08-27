import numpy as np
from keras.models import load_model
import pandas as pd
import joblib
import os
import argparse
import json
import sys
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

def _prepare_scaled_features(recent_data: pd.DataFrame, feature_cols, target_col, scaler, sequence_length: int):
    """
    Prepare a scaled feature matrix (last `sequence_length` rows) using the provided scaler.
    Ensures no NaNs reach the scaler:
      - forward/back fill
      - neutral defaults for fully-missing technical indicators
      - median fallback
    """
    data = recent_data.copy().sort_index()
    cols = feature_cols + [target_col]

    # Initial fill
    data[cols] = data[cols].ffill().bfill()

    if len(data) < sequence_length:
        raise ValueError(f"Insufficient rows after cleaning. Need {sequence_length}, have {len(data)}")

    window = data.iloc[-sequence_length:].copy()

    # Detect fully-missing feature columns in the window
    fully_missing = [c for c in feature_cols if window[c].isna().all()]
    if fully_missing:
        imputed = {}
        close_ref_col = target_col  # assume target_col is close/price
        close_vals = window[close_ref_col]

        for col in fully_missing:
            lower = col.lower()

            if lower == "rsi":
                window[col] = 50.0
                imputed[col] = "neutral_rsi_50"
            elif "macd_signal" in lower:
                window[col] = 0.0
                imputed[col] = "neutral_macd_signal_0"
            elif "macd" in lower:
                window[col] = 0.0
                imputed[col] = "neutral_macd_0"
            elif lower.startswith("sma") or lower.startswith("ema"):
                # Use current close as proxy
                window[col] = close_vals.values
                imputed[col] = "proxy_close_for_ma"
            elif lower.startswith("bb_bbm"):
                window[col] = close_vals.values
                imputed[col] = "bb_mid=close"
            elif lower.startswith("bb_bbh"):
                window[col] = close_vals.values
                imputed[col] = "bb_high=close"
            elif lower.startswith("bb_bbl"):
                window[col] = close_vals.values
                imputed[col] = "bb_low=close"
            else:
                # Generic neutral fallback
                window[col] = close_vals.values
                imputed[col] = "generic_close_fallback"

        logger.warning(f"Imputed fully-missing indicators in prediction window: {imputed}")

    # Column-wise median fallback for any residual NaNs
    residual_nan_cols = [c for c in feature_cols if window[c].isna().any()]
    if residual_nan_cols:
        med_map = {}
        for c in residual_nan_cols:
            med = data[c].median()
            if not np.isfinite(med):
                med = 0.0
            window[c] = window[c].fillna(med)
            med_map[c] = med
        logger.warning(f"Filled residual NaNs with medians: {med_map}")

    # Final guard
    if window[feature_cols].isna().any().any():
        bad = window[feature_cols].isna().sum()
        raise ValueError(f"NaNs remain after imputation: {bad[bad>0].to_dict()}")

    # Build scaling frame
    x_features = window[feature_cols].astype(float).values
    dummy_target = np.zeros((sequence_length, 1), dtype=np.float32)
    all_cols = feature_cols + [target_col]
    to_scale = pd.DataFrame(np.hstack([x_features, dummy_target]), columns=all_cols)

    recent_scaled_df = pd.DataFrame(scaler.transform(to_scale), columns=all_cols)

    if recent_scaled_df.isna().any().any():
        bad_scaled = recent_scaled_df.isna().sum()
        raise ValueError(f"Scaler produced NaNs (likely corrupt scaler). Columns: {bad_scaled[bad_scaled>0].to_dict()}")

    x_scaled = recent_scaled_df.iloc[:, :-1].values
    last_close = float(window[target_col].iloc[-1])
    return x_scaled, last_close, all_cols

def predict_next(model, scaler, feature_cols, target_col, recent_data, sequence_length=10):
    """
    Predict the next-step return using the LSTM model.
    Returns a clipped percent return (e.g., 0.01 for +1%).
    """
    x_scaled, last_close, all_cols = _prepare_scaled_features(
        recent_data, feature_cols, target_col, scaler, sequence_length
    )

    # LSTM expects shape (1, seq_len, n_features)
    X = x_scaled.reshape(1, sequence_length, len(feature_cols))
    pred_scaled = model.predict(X, verbose=0)
    pred_scaled_value = float(np.ravel(pred_scaled)[-1])

    # Inverse transform to price by stitching scaled features + predicted scaled target
    dummy_row = np.zeros((1, len(all_cols)), dtype=np.float32)
    dummy_row[0, :-1] = x_scaled[-1]  # last scaled feature row
    dummy_row[0, -1] = pred_scaled_value
    pred_actual = float(scaler.inverse_transform(dummy_row)[0, -1])

    # Convert to percent return; guard and clip
    if not np.isfinite(pred_actual) or not np.isfinite(last_close) or last_close == 0:
        return np.nan
    predicted_return = (pred_actual - last_close) / last_close
    # Clip to a realistic next-step range
    predicted_return = float(np.clip(predicted_return, -0.3, 0.3))
    return predicted_return

def load_online_model(model_path, n_features):
    model = OnlineGradientDescentMomentum(n_features=n_features, model_path=model_path)
    return model

def predict_online(model, scaler, feature_cols, target_col, recent_data, sequence_length=10):
    """
    Predict next-step return using the online model.
    - Standardizes features with the same scaler to avoid magnitude blowups.
    - Auto-detects if the model output looks like a price; otherwise treats as return.
    - Clips output to a realistic range.
    """
    # Prepare scaled features for the last window
    x_scaled, last_close, _ = _prepare_scaled_features(
        recent_data, feature_cols, target_col, scaler, sequence_length
    )

    # Align feature count with model weights if needed
    expected_features = len(model.weights)
    actual_features = x_scaled.shape[1]
    if expected_features != actual_features:
        if expected_features > actual_features:
            padding = np.zeros((x_scaled.shape[0], expected_features - actual_features))
            x_scaled = np.hstack([x_scaled, padding])
        else:
            x_scaled = x_scaled[:, :expected_features]

    # Model prediction for each row in the window; take last as next-step
    preds = model.predict(x_scaled)
    raw_pred = float(np.ravel(preds)[-1])

    if not np.isfinite(raw_pred):
        return np.nan

    # Try both interpretations and choose the more plausible one
    candidate_return_raw = raw_pred  # if the model already predicts return
    candidate_return_from_price = (raw_pred - last_close) / last_close if last_close != 0 else np.nan

    # Heuristic: choose the candidate with smaller absolute value within sane bounds
    choices = []
    if np.isfinite(candidate_return_raw):
        choices.append(candidate_return_raw)
    if np.isfinite(candidate_return_from_price):
        choices.append(candidate_return_from_price)

    if not choices:
        return np.nan

    picked = min(choices, key=lambda v: abs(v))
    picked = float(np.clip(picked, -0.3, 0.3))
    return picked

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
    max_allocation = 0.25  # Diversification constraint: no single holding > 25%
    suggested = {}

    # Calculate suggested allocations based on both predictions and current weights
    for ticker in sorted_tickers:
        current = current_allocations.get(ticker, 0)
        # Target weight based on prediction rank
        target = rank_weights[ticker]

        # Adjust weight considering current allocation (blend of current and target)
        adjustment = (target - current) * max_change_pct
        suggested_weight = current + adjustment
        
        # Apply diversification constraint
        suggested[ticker] = min(suggested_weight, max_allocation)

    # Normalize to ensure weights sum to 1
    total_weight = sum(suggested.values())
    suggested = {t: round(w/total_weight, 4) for t, w in suggested.items()}
    
    # Final check and redistribution if any allocation still exceeds 25% after normalization
    while any(weight > max_allocation for weight in suggested.values()):
        excess_total = 0
        compliant_tickers = []
        
        for ticker, weight in suggested.items():
            if weight > max_allocation:
                excess = weight - max_allocation
                excess_total += excess
                suggested[ticker] = max_allocation
            else:
                compliant_tickers.append(ticker)
        
        # Redistribute excess to compliant tickers proportionally
        if compliant_tickers and excess_total > 0:
            compliant_total = sum(suggested[t] for t in compliant_tickers)
            if compliant_total > 0:
                for ticker in compliant_tickers:
                    additional = (suggested[ticker] / compliant_total) * excess_total
                    suggested[ticker] = min(suggested[ticker] + additional, max_allocation)
        else:
            # If no compliant tickers, distribute equally among all
            equal_weight = 1.0 / len(suggested)
            suggested = {t: min(equal_weight, max_allocation) for t in suggested.keys()}
            break
    
    # Final normalization
    total_weight = sum(suggested.values())
    suggested = {t: round(w/total_weight, 4) for t, w in suggested.items()}
    
    # Log diversification compliance
    max_actual = max(suggested.values()) if suggested else 0
    logger.info(f"Diversification check: Maximum allocation = {max_actual:.2%} (limit: {max_allocation:.2%})")

    return suggested

def hybrid_predict_and_rebalance(recent_data, lstm_model_path, lstm_scaler_path, online_model_path, tickers, current_allocations, sequence_length=10):
    try:
        lstm_model, scaler, feature_cols, target_col = load_lstm_model(lstm_model_path, lstm_scaler_path)
        online_model = load_online_model(online_model_path, n_features=len(feature_cols))
        hybrid_preds = {}

        for ticker in tickers:
            ticker_data = recent_data[recent_data['ticker'] == ticker]
            if len(ticker_data) < sequence_length:
                logger.info(f"Not enough data for {ticker} to make predictions. Skipping.")
                continue

            # LSTM predicted return
            lstm_ret = predict_next(lstm_model, scaler, feature_cols, target_col, ticker_data, sequence_length)
            # Online model predicted return (standardized features)
            online_ret = predict_online(online_model, scaler, feature_cols, target_col, ticker_data, sequence_length)

            # Skip if both are invalid
            vals = [v for v in [lstm_ret, online_ret] if np.isfinite(v)]
            if not vals:
                logger.warning(f"No valid predictions for {ticker}. Skipping.")
                continue

            # Simple average of available return estimates, then clip
            hybrid_ret = float(np.clip(np.mean(vals), -0.3, 0.3))
            hybrid_preds[ticker] = hybrid_ret

        if not hybrid_preds:
            return {
                'status': 'error',
                'predictions': {},
                'suggested_allocations': {},
                'message': 'No valid predictions generated. Check data quality and model artifacts.'
            }

        suggested = suggest_rebalance(hybrid_preds, current_allocations)
        return {
            'status': 'success',
            'predictions': hybrid_preds,
            'suggested_allocations': suggested,
            'message': f'Successfully generated predictions for {len(hybrid_preds)} tickers'
        }
    except Exception as e:
        logger.error(f"Error in hybrid prediction: {str(e)}")
        return {
            'status': 'error',
            'message': str(e),
            'predictions': {},
            'suggested_allocations': {}
        }

def parse_allocations(allocation_str):
    """Parse allocation string in format 'AAPL:0.7,TSLA:0.3'"""
    try:
        allocations = {}
        pairs = allocation_str.split(',')
        for pair in pairs:
            ticker, weight = pair.split(':')
            allocations[ticker.strip()] = float(weight.strip())
        return allocations
    except Exception as e:
        raise ValueError(f"Invalid allocation format. Expected 'TICKER:WEIGHT,TICKER:WEIGHT'. Error: {e}")

def main():
    parser = argparse.ArgumentParser(description='Generate portfolio rebalancing suggestions using hybrid LSTM-Online model')
    
    parser.add_argument('--data-path', required=True, 
                       help='Path to the recent stock data CSV file')
    parser.add_argument('--lstm-model-path', required=True,
                       help='Path to the trained LSTM model (.keras file)')
    parser.add_argument('--lstm-scaler-path', required=True,
                       help='Path to the LSTM model scaler (.pkl file)')
    parser.add_argument('--online-model-path', required=True,
                       help='Path to the online learning model (.pkl file)')
    parser.add_argument('--tickers', required=True,
                       help='Comma-separated list of tickers (e.g., AAPL,TSLA,GOOGL)')
    parser.add_argument('--current-allocations', required=True,
                       help='Current allocations in format TICKER:WEIGHT,TICKER:WEIGHT (e.g., AAPL:0.7,TSLA:0.3)')
    parser.add_argument('--sequence-length', type=int, default=10,
                       help='Sequence length for LSTM predictions (default: 10)')
    parser.add_argument('--output', choices=['json', 'pretty'], default='json',
                       help='Output format: json or pretty (default: json)')
    
    args = parser.parse_args()
    
    try:
        # Load and validate data
        if not os.path.exists(args.data_path):
            raise FileNotFoundError(f"Data file not found: {args.data_path}")
            
        recent_data = pd.read_csv(args.data_path)
        tickers = [t.strip() for t in args.tickers.split(',')]
        current_allocations = parse_allocations(args.current_allocations)
        
        # Validate that all specified tickers exist in data
        available_tickers = recent_data['ticker'].unique()
        missing_tickers = set(tickers) - set(available_tickers)
        if missing_tickers:
            logger.warning(f"Tickers not found in data: {missing_tickers}")
            tickers = [t for t in tickers if t in available_tickers]
        
        if not tickers:
            raise ValueError("No valid tickers found in the data")
            
        # Validate allocation weights sum to 1
        total_weight = sum(current_allocations.values())
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(f"Current allocations sum to {total_weight}, normalizing to 1.0")
            current_allocations = {k: v/total_weight for k, v in current_allocations.items()}
        
        # Generate predictions and suggestions
        result = hybrid_predict_and_rebalance(
            recent_data,
            args.lstm_model_path,
            args.lstm_scaler_path,
            args.online_model_path,
            tickers,
            current_allocations,
            args.sequence_length
        )
        
        # Output results
        if args.output == 'json':
            print(json.dumps(result, indent=2))
        else:
            if result['status'] == 'success':
                print("Portfolio Rebalancing Suggestions:")
                print("=" * 40)
                print(f"Status: {result['status']}")
                print(f"Message: {result['message']}")
                print("\nPredicted Returns:")
                for ticker, pred in result['predictions'].items():
                    print(f"  {ticker}: {pred:.4f}")
                print("\nSuggested Allocations:")
                for ticker, weight in result['suggested_allocations'].items():
                    print(f"  {ticker}: {weight:.2%}")
            else:
                print(f"Error: {result['message']}")
                sys.exit(1)
                
    except Exception as e:
        error_result = {
            'status': 'error',
            'message': str(e),
            'predictions': {},
            'suggested_allocations': {}
        }
        if args.output == 'json':
            print(json.dumps(error_result, indent=2))
        else:
            print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
