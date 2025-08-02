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

def predict_next(model, scaler, feature_cols, recent_data, sequence_length=10):
    # recent_data: DataFrame with latest `sequence_length` rows
    X_features = recent_data[feature_cols].values[-sequence_length:]

    # Stack dummy target column for scaling
    dummy_target = np.zeros((sequence_length, 1))
    recent_for_scaling = np.hstack([X_features, dummy_target])

    all_cols = feature_cols + ['close']
    recent_for_scaling_df = pd.DataFrame(recent_for_scaling, columns=all_cols)
    recent_scaled_df = pd.DataFrame(scaler.transform(recent_for_scaling_df), columns=all_cols)
    recent_scaled = recent_scaled_df.iloc[:, :-1].values

    X = recent_scaled.reshape(1, sequence_length, len(feature_cols))
    pred_scaled = model.predict(X)
    pred_value = pred_scaled.item()

    # Inverse transform to get the actual prediction
    dummy_df = pd.DataFrame(np.zeros((1, len(feature_cols) + 1)), columns=all_cols)
    dummy_df.iloc[0, :-1] = recent_scaled[-1]
    dummy_df.iloc[0, -1] = pred_value
    pred_actual = scaler.inverse_transform(dummy_df)[0, -1]

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
    suggested = {t: round(w/total_weight, 2) for t, w in suggested.items()}
    
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
    suggested = {t: round(w/total_weight, 2) for t, w in suggested.items()}
    
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
            lstm_pred = predict_next(lstm_model, scaler, feature_cols, ticker_data, sequence_length)
            online_pred = predict_online(online_model, feature_cols, ticker_data)
            # Combine predictions using a simple average / can try weighted later
            hybrid_pred = (lstm_pred + online_pred) / 2
            hybrid_preds[ticker] = hybrid_pred
            
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
