"""
Performance comparison of multiple models:
- LSTM
- OGDM (online gradient descent momentum)
- Hybrid (average of LSTM + OGDM)
- Equal-weight baseline
- Return-persistence baseline (uses prior period return as forecast)

INPUT DATA REQUIREMENTS
A CSV with at least: date, ticker, close
Optional sentiment columns ignored here.

MODEL ARTIFACTS
Requires:
- LSTM model (.keras) + scaler (.pkl) built like existing pipeline
- OGDM model (.pkl)

BACKTEST LOGIC (rolling walk-forward):
For each date after warmup window:
  1. Slice prior sequence_length rows per ticker for LSTM/OGDM feature window.
  2. Generate per-ticker predicted next return for each model.
  3. Convert predicted returns into weights (rank-based) for strategy portfolios.
  4. Realized next-day return = close_{t+1}/close_t - 1 (if available).
  5. Accumulate portfolio equity assuming daily rebalancing to model weights.

OUTPUTS
- metrics_table.csv (MSE, MAE, directional accuracy, cumulative return, volatility, Sharpe)
- predictions_long.csv (date, ticker, model, predicted_return)
- portfolio_equity.csv (date, model, equity)
- Optional interactive HTML plots if --make-plots passed.

USAGE
python -m src.evaluation.performance_comparison \
  --data-csv data/processed/stock_prices_historical.csv \
  --lstm-model models/lstm_model.keras \
  --lstm-scaler models/lstm_model_scaler.pkl \
  --ogdm-model models/ogdm_model.pkl \
  --sequence-length 10 \
  --output-dir evaluation_results/perf \
  --make-plots
"""
import os
import argparse
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# Lazy imports (only if user supplies models)
def _lazy_import_models():
    from keras.models import load_model
    import joblib
    from src.model.online_learning import OnlineGradientDescentMomentum
    return load_model, joblib, OnlineGradientDescentMomentum

from src.model.lstm_ogdm_hybrid import predict_next, predict_online, load_lstm_model, load_online_model, suggest_rebalance

def compute_realized_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Add realized next-day return per ticker."""
    df = df.sort_values(["ticker", "date"]).copy()
    df["next_close"] = df.groupby("ticker")["close"].shift(-1)
    df["realized_return"] = (df["next_close"] - df["close"]) / df["close"]
    return df

def rank_weight_allocation(preds: Dict[str, float]) -> Dict[str, float]:
    """Convert predicted returns dict into rank-based allocation (sums to 1)."""
    if not preds:
        return {}
    sorted_tickers = sorted(preds, key=preds.get, reverse=True)
    n = len(sorted_tickers)
    raw = {t: (n - i) for i, t in enumerate(sorted_tickers)}
    total = sum(raw.values())
    return {t: raw[t] / total for t in sorted_tickers}

def calc_portfolio_equity(returns_frame: pd.DataFrame,
                          weights_history: Dict[str, Dict[str, float]],
                          start_equity: float = 1.0) -> pd.DataFrame:
    """
    Build equity curve for each model given per-date weights and realized returns.
    returns_frame: date,ticker,realized_return
    weights_history: {model: {date: {ticker: weight}}}
    """
    models = list(weights_history.keys())
    dates = sorted(returns_frame["date"].unique())
    records = []
    for model in models:
        equity = start_equity
        for d in dates:
            day_weights = weights_history[model].get(d)
            if not day_weights:
                continue
            day_returns = returns_frame.loc[returns_frame["date"] == d, ["ticker", "realized_return"]]
            merged = day_returns.merge(
                pd.DataFrame({"ticker": list(day_weights.keys()),
                              "weight": list(day_weights.values())}),
                on="ticker", how="inner")
            if merged.empty:
                continue
            port_ret = float(np.nansum(merged["weight"] * merged["realized_return"]))
            equity *= (1 + port_ret)
            records.append({"date": d, "model": model, "equity": equity})
    return pd.DataFrame(records)

def compute_metrics(pred_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-model predictive & portfolio metrics.
    pred_df columns: date,ticker,model,predicted_return,realized_return
    """
    rows = []
    for model, grp in pred_df.groupby("model"):
        valid = grp.dropna(subset=["predicted_return", "realized_return"])
        if valid.empty:
            continue
        mse = float(np.mean((valid["predicted_return"] - valid["realized_return"]) ** 2))
        mae = float(np.mean(np.abs(valid["predicted_return"] - valid["realized_return"])))
        directional = float(np.mean(np.sign(valid["predicted_return"]) == np.sign(valid["realized_return"])))
        avg_ret = float(valid["realized_return"].mean())
        vol = float(valid["realized_return"].std(ddof=1))
        sharpe = avg_ret / vol if vol > 0 else np.nan
        cum_return = float((1 + valid["realized_return"]).prod() - 1)
        rows.append({
            "model": model,
            "mse": mse,
            "mae": mae,
            "directional_accuracy": directional,
            "avg_return": avg_ret,
            "volatility": vol,
            "sharpe": sharpe,
            "cumulative_return": cum_return
        })
    return pd.DataFrame(rows)

def backtest_models(df: pd.DataFrame,
                    lstm_paths: Tuple[str, str],
                    ogdm_path: str,
                    sequence_length: int,
                    max_days: int = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Core rolling backtest returning (predictions_long, equity_curves_long).
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = compute_realized_returns(df)
    if max_days:
        unique_dates = sorted(df["date"].unique())
        df = df[df["date"].isin(unique_dates[-max_days:])]

    # Load models
    lstm_model, scaler, feature_cols, target_col = load_lstm_model(*lstm_paths)
    ogdm_model = load_online_model(ogdm_path, n_features=len(feature_cols))

    tickers = sorted(df["ticker"].unique())
    dates = sorted(df["date"].unique())
    warmup_dates = dates[sequence_length:-1]  # last date has no realized next-day return

    prediction_rows = []
    weights_history = {m: {} for m in ["LSTM", "OGDM", "HYBRID", "EQUAL_WEIGHT", "RETURN_PERSIST"]}
    for d in warmup_dates:
        day_slice = df[df["date"] <= d]
        preds_lstm = {}
        preds_ogdm = {}
        preds_persist = {}
        for t in tickers:
            t_slice = day_slice[day_slice["ticker"] == t].sort_values("date")
            if len(t_slice) < sequence_length + 1:
                continue
            try:
                lstm_ret = predict_next(lstm_model, scaler, feature_cols, target_col, t_slice, sequence_length)
            except Exception:
                lstm_ret = np.nan
            try:
                ogdm_ret = predict_online(ogdm_model, scaler, feature_cols, target_col, t_slice, sequence_length)
            except Exception:
                ogdm_ret = np.nan
            preds_lstm[t] = lstm_ret
            preds_ogdm[t] = ogdm_ret
            # Return-persistence baseline: last realized_return (shifted)
            last_row = t_slice.iloc[-2]  # -1 row is same day; -2 row realized_return for -1
            preds_persist[t] = last_row.get("realized_return", np.nan)

        # Hybrid
        preds_hybrid = {}
        for t in preds_lstm:
            vals = [v for v in [preds_lstm.get(t), preds_ogdm.get(t)] if np.isfinite(v)]
            preds_hybrid[t] = float(np.mean(vals)) if vals else np.nan

        # Equal weight baseline does not need predictions (use placeholder zeros)
        preds_equal = {t: 0.0 for t in preds_lstm.keys()}

        model_pred_maps = {
            "LSTM": preds_lstm,
            "OGDM": preds_ogdm,
            "HYBRID": preds_hybrid,
            "EQUAL_WEIGHT": preds_equal,
            "RETURN_PERSIST": preds_persist
        }

        # Build weights
        weights_history["EQUAL_WEIGHT"][d] = {t: 1 / len(preds_lstm) for t in preds_lstm} if preds_lstm else {}
        weights_history["RETURN_PERSIST"][d] = rank_weight_allocation(preds_persist)
        weights_history["LSTM"][d] = rank_weight_allocation(preds_lstm)
        weights_history["OGDM"][d] = rank_weight_allocation(preds_ogdm)
        weights_history["HYBRID"][d] = rank_weight_allocation(preds_hybrid)

        # Collect per ticker predictions
        realized_map = df.loc[df["date"] == d, ["ticker", "realized_return"]].set_index("ticker")["realized_return"].to_dict()
        for model_name, pred_map in model_pred_maps.items():
            for t, p in pred_map.items():
                prediction_rows.append({
                    "date": d,
                    "ticker": t,
                    "model": model_name,
                    "predicted_return": p,
                    "realized_return": realized_map.get(t, np.nan)
                })

    predictions_long = pd.DataFrame(prediction_rows)
    equity_curves_long = calc_portfolio_equity(
        returns_frame=df[["date", "ticker", "realized_return"]],
        weights_history=weights_history
    )
    return predictions_long, equity_curves_long

def make_plots(equity_df: pd.DataFrame,
               metrics_df: pd.DataFrame,
               output_dir: str) -> None:
    """Generate and save comparison plots."""
    if equity_df.empty:
        return
    os.makedirs(output_dir, exist_ok=True)
    fig_equity = px.line(equity_df, x="date", y="equity", color="model",
                         title="Portfolio Equity Over Time")
    fig_equity.write_html(os.path.join(output_dir, "equity_curves.html"))
    # Drawdown
    dd_records = []
    for model, grp in equity_df.groupby("model"):
        peak = -np.inf
        for _, r in grp.sort_values("date").iterrows():
            peak = max(peak, r.equity)
            dd = (r.equity / peak) - 1
            dd_records.append({"date": r.date, "model": model, "drawdown": dd})
    dd_df = pd.DataFrame(dd_records)
    fig_dd = px.line(dd_df, x="date", y="drawdown", color="model",
                     title="Drawdowns")
    fig_dd.write_html(os.path.join(output_dir, "drawdowns.html"))
    # Bar metrics (Sharpe & cumulative)
    fig_sharpe = px.bar(metrics_df, x="model", y="sharpe", title="Sharpe Ratios")
    fig_sharpe.write_html(os.path.join(output_dir, "sharpe.html"))

def main():
    parser = argparse.ArgumentParser(description="Compare predictive & portfolio performance across models.")
    parser.add_argument("--data-csv", required=True)
    parser.add_argument("--lstm-model", required=True)
    parser.add_argument("--lstm-scaler", required=True)
    parser.add_argument("--ogdm-model", required=True)
    parser.add_argument("--sequence-length", type=int, default=10)
    parser.add_argument("--max-days", type=int, default=None, help="Limit recent days for faster debug.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--make-plots", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.read_csv(args.data_csv)
    for col in ["date", "ticker", "close"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    predictions_long, equity_curves = backtest_models(
        df=df,
        lstm_paths=(args.lstm_model, args.lstm_scaler),
        ogdm_path=args.ogdm_model,
        sequence_length=args.sequence_length,
        max_days=args.max_days
    )
    metrics_df = compute_metrics(predictions_long)

    predictions_long.to_csv(os.path.join(args.output_dir, "predictions_long.csv"), index=False)
    equity_curves.to_csv(os.path.join(args.output_dir, "portfolio_equity.csv"), index=False)
    metrics_df.to_csv(os.path.join(args.output_dir, "metrics_table.csv"), index=False)

    if args.make_plots:
        make_plots(equity_curves, metrics_df, args.output_dir)

    print("Saved outputs to", args.output_dir)

if __name__ == "__main__":
    main()
