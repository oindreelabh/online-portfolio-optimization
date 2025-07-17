# src/evaluation/evaluate_lstm.py
import numpy as np
import pandas as pd
import argparse
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import load_model
from src.utils.logger import setup_logger
import os
import matplotlib.pyplot as plt

logger = setup_logger(os.path.basename(__file__).replace(".py", ""))

def mean_absolute_percentage_error(y_true, y_pred):
    """Calculate MAPE avoiding division by zero"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Filter out zero values in y_true to avoid division by zero
    non_zero = (y_true != 0)
    return np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100

def filter_nans(y_actual, y_predictions):
    """Filter out NaN values from both actual and predicted arrays"""
    mask = ~np.isnan(y_actual) & ~np.isnan(y_predictions)
    if np.sum(mask) == 0:
        logger.warning("All values are NaN after filtering!")
        return [], []

    if np.sum(~mask) > 0:
        logger.warning(f"Filtered out {np.sum(~mask)} NaN values from {len(mask)} total values")

    return y_actual[mask], y_predictions[mask]

def evaluate_lstm_model(data_path, model_path, scaler_path, sequence_length=5, output_dir=None):
    """Evaluate LSTM model on test data"""
    # Load model, scaler and feature info
    try:
        model = load_model(model_path)
        scaler_dict = joblib.load(scaler_path)
        scaler = scaler_dict['scaler']
        feature_cols = scaler_dict['feature_cols']
        target_col = scaler_dict['target_col']
    except Exception as e:
        logger.error(f"Error loading model or scaler: {e}")
        return {"error": str(e)}

    # Load test data
    try:
        test_data = pd.read_csv(data_path)

        # Check for NaN values in input data
        nan_count = test_data[feature_cols + [target_col]].isna().sum().sum()
        if nan_count > 0:
            logger.warning(f"Found {nan_count} NaN values in test data, filling with zeros")
            test_data[feature_cols + [target_col]] = test_data[feature_cols + [target_col]].fillna(0)

        # Check for infinite values
        inf_count = np.isinf(test_data[feature_cols + [target_col]].values).sum()
        if inf_count > 0:
            logger.warning(f"Found {inf_count} infinite values in test data, replacing with large finite values")
            test_data[feature_cols + [target_col]] = test_data[feature_cols + [target_col]].replace([np.inf, -np.inf], np.finfo(np.float64).max)
    except Exception as e:
        logger.error(f"Error loading or preprocessing test data: {e}")
        return {"error": str(e)}

    # Group by ticker
    results = {}
    all_predictions = []
    all_actual = []

    for ticker, group in test_data.groupby('ticker'):
        logger.info(f"Evaluating LSTM model for {ticker}")

        if len(group) <= sequence_length:
            logger.warning(f"Not enough data for {ticker} (need > {sequence_length} rows)")
            continue

        try:
            # Scale data
            X_features = group[feature_cols].values
            y_actual = group[target_col].values[sequence_length:]

            # Prepare X sequences
            X_sequences = []
            for i in range(len(X_features) - sequence_length):
                X_sequences.append(X_features[i:i+sequence_length])

            X_sequences = np.array(X_sequences)

            # Scale X using proper DataFrame with feature names
            X_scaled_list = []
            for i in range(len(X_sequences)):
                sequence = X_sequences[i]
                dummy_target = np.zeros((sequence_length, 1))
                sequence_for_scaling = np.hstack([sequence, dummy_target])
                all_cols = feature_cols + [target_col]
                sequence_df = pd.DataFrame(sequence_for_scaling, columns=all_cols)
                sequence_scaled = scaler.transform(sequence_df)[:, :-1]
                X_scaled_list.append(sequence_scaled)

            X_scaled = np.array(X_scaled_list)

            # Predict
            y_pred_scaled = model.predict(X_scaled)

            # For each prediction, inverse transform using proper DataFrame
            y_predictions = []
            for i in range(len(y_pred_scaled)):
                dummy_df = pd.DataFrame(np.zeros((1, len(feature_cols) + 1)), columns=feature_cols + [target_col])
                dummy_df.iloc[0, :-1] = X_scaled[i][-1]  # Last timestep features
                dummy_df.iloc[0, -1] = y_pred_scaled[i][0]  # Prediction
                pred_actual = scaler.inverse_transform(dummy_df)[0, -1]
                y_predictions.append(pred_actual)

            y_predictions = np.array(y_predictions)

            # Filter out NaN values before calculating metrics
            y_actual_filtered, y_predictions_filtered = filter_nans(y_actual, y_predictions)

            if len(y_actual_filtered) == 0:
                logger.warning(f"No valid predictions for {ticker} after filtering NaNs")
                results[ticker] = {
                    'MSE': np.nan,
                    'RMSE': np.nan,
                    'MAE': np.nan,
                    'R²': np.nan,
                    'MAPE': np.nan
                }
                continue

            # Metrics
            mse = mean_squared_error(y_actual_filtered, y_predictions_filtered)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_actual_filtered, y_predictions_filtered)
            r2 = r2_score(y_actual_filtered, y_predictions_filtered)
            mape = mean_absolute_percentage_error(y_actual_filtered, y_predictions_filtered)

            results[ticker] = {
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'R²': r2,
                'MAPE': mape
            }

            all_predictions.extend(y_predictions_filtered)
            all_actual.extend(y_actual_filtered)

            # Plot predictions vs actual for this ticker
            if output_dir:
                plt.figure(figsize=(12, 6))
                plt.plot(y_actual_filtered, label='Actual')
                plt.plot(y_predictions_filtered, label='Predicted')
                plt.title(f'LSTM Predictions vs Actual for {ticker}')
                plt.legend()
                plt.savefig(f'{output_dir}/{ticker}_lstm_predictions.png')
                plt.close()

        except Exception as e:
            logger.error(f"Error evaluating ticker {ticker}: {e}")
            results[ticker] = {"error": str(e)}

    # If we have valid predictions, calculate overall metrics
    if len(all_actual) > 0:
        # Filter out any remaining NaNs in the collected predictions
        all_actual_filtered, all_predictions_filtered = filter_nans(np.array(all_actual), np.array(all_predictions))

        if len(all_actual_filtered) > 0:
            overall_mse = mean_squared_error(all_actual_filtered, all_predictions_filtered)
            overall_rmse = np.sqrt(overall_mse)
            overall_mae = mean_absolute_error(all_actual_filtered, all_predictions_filtered)
            overall_r2 = r2_score(all_actual_filtered, all_predictions_filtered)
            overall_mape = mean_absolute_percentage_error(all_actual_filtered, all_predictions_filtered)

            results['overall'] = {
                'MSE': overall_mse,
                'RMSE': overall_rmse,
                'MAE': overall_mae,
                'R²': overall_r2,
                'MAPE': overall_mape
            }

            # Save results to CSV
            if output_dir:
                results_df = pd.DataFrame.from_dict(results, orient='index')
                results_df.to_csv(f'{output_dir}/lstm_evaluation_results.csv')

                # Plot overall predictions vs actual
                plt.figure(figsize=(12, 6))
                plt.scatter(all_actual_filtered, all_predictions_filtered, alpha=0.5)
                plt.plot([min(all_actual_filtered), max(all_actual_filtered)],
                         [min(all_actual_filtered), max(all_actual_filtered)], 'r--')
                plt.xlabel('Actual')
                plt.ylabel('Predicted')
                plt.title('LSTM Model: Predicted vs Actual')
                plt.savefig(f'{output_dir}/lstm_overall_predictions.png')
                plt.close()
        else:
            logger.error("No valid predictions after filtering all data")
            results['overall'] = {"error": "No valid predictions after filtering"}
    else:
        logger.error("No valid predictions collected")
        results['overall'] = {"error": "No valid predictions collected"}

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LSTM model performance")
    parser.add_argument("--data_path", type=str, required=True, help="Path to test data CSV")
    parser.add_argument("--model_path", type=str, required=True, help="Path to LSTM model file")
    parser.add_argument("--scaler_path", type=str, required=True, help="Path to scaler file")
    parser.add_argument("--sequence_length", type=int, default=5, help="Sequence length for LSTM")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save evaluation results")

    args = parser.parse_args()

    if args.output_dir and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    results = evaluate_lstm_model(
        args.data_path,
        args.model_path,
        args.scaler_path,
        args.sequence_length,
        args.output_dir
    )

    # Print overall results
    if "error" not in results.get('overall', {}):
        logger.info("\nLSTM Model Evaluation Results (Overall):")
        for metric, value in results['overall'].items():
            logger.info(f"{metric}: {value:.4f}")
    else:
        logger.error(f"\nError in evaluation: {results['overall'].get('error')}")