import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Optional
from src.utils.logger import setup_logger

logger = setup_logger(os.path.basename(__file__).replace(".py", ""))


# ---------------------- Metric Computation Functions ---------------------- #

def annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Calculate annualized geometric return from periodic returns.
    Uses compound growth: prod(1 + r) ** (periods_per_year / n) - 1.
    """
    returns = returns.dropna()
    if returns.empty:
        return np.nan
    cumulative_growth = (1.0 + returns).prod()
    n_periods = len(returns)
    if n_periods == 0 or cumulative_growth <= 0:
        return np.nan
    return cumulative_growth ** (periods_per_year / n_periods) - 1.0


def annualized_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Calculate annualized volatility (standard deviation) of returns.
    """
    returns = returns.dropna()
    if returns.empty:
        return np.nan
    return returns.std(ddof=0) * np.sqrt(periods_per_year)


def sharpe_ratio(returns: pd.Series,
                 risk_free_rate: float = 0.0,
                 periods_per_year: int = 252) -> float:
    """
    Calculate annualized Sharpe Ratio.
    risk_free_rate is annual; converted to per-period internally.
    """
    returns = returns.dropna()
    if returns.empty:
        return np.nan
    rf_per_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    excess = returns - rf_per_period
    vol = excess.std(ddof=0)
    if vol == 0 or np.isnan(vol):
        return np.nan
    mean_excess = excess.mean()
    return (mean_excess / vol) * np.sqrt(periods_per_year)


def max_drawdown(returns: pd.Series) -> float:
    """
    Compute Maximum Drawdown from periodic returns series.
    Returns the most negative drawdown (e.g., -0.35 means -35%).
    """
    returns = returns.dropna()
    if returns.empty:
        return np.nan
    equity = (1 + returns).cumprod()
    rolling_peak = equity.cummax()
    drawdowns = equity / rolling_peak - 1.0
    return drawdowns.min()


def compute_performance_metrics(returns: pd.Series,
                                risk_free_rate: float = 0.0,
                                periods_per_year: int = 252) -> Dict[str, float]:
    """
    Aggregate standard performance metrics for a single return series.
    """
    return {
        "annualized_return": annualized_return(returns, periods_per_year),
        "annualized_volatility": annualized_volatility(returns, periods_per_year),
        "sharpe_ratio": sharpe_ratio(returns, risk_free_rate, periods_per_year),
        "max_drawdown": max_drawdown(returns)
    }


# ---------------------- Backtesting Helpers ---------------------- #

def backtest_from_signals(price_series: pd.Series,
                          position_series: pd.Series,
                          transaction_cost_bps: float = 0.0) -> pd.Series:
    """
    Convert price data and position signals into net periodic returns.
    Assumptions:
      - position_series aligns with price_series (same index)
      - positions are target end-of-period exposures in [-1, 0, 1] (or any float)
      - transaction cost applied on notional change: cost = cost_rate * |pos_t - pos_{t-1}|
      - transaction_cost_bps: round-trip basis points per unit turnover (e.g., 10 = 0.10%)
    """
    # Ensure alignment and numeric
    prices = price_series.astype(float)
    positions = position_series.astype(float).fillna(0.0)

    # Compute simple returns from prices
    price_returns = prices.pct_change().fillna(0.0)

    # Shift positions to represent exposure during period t (enter at start)
    shifted_positions = positions.shift(1).fillna(0.0)

    gross_returns = shifted_positions * price_returns

    # Turnover and costs
    turnover = positions.diff().abs().fillna(0.0)
    cost_rate = transaction_cost_bps / 10000.0  # bps to decimal
    costs = turnover * cost_rate

    net_returns = gross_returns - costs
    return net_returns


def compare_strategies(strategy_returns: Dict[str, pd.Series],
                       risk_free_rate: float = 0.0,
                       periods_per_year: int = 252) -> pd.DataFrame:
    """
    Compute metrics for multiple strategies.
    Returns DataFrame indexed by strategy name.
    """
    metrics = {}
    for name, rets in strategy_returns.items():
        if not isinstance(rets, pd.Series):
            logger.warning(f"Strategy {name} returns not a Series, skipping")
            continue
        metrics[name] = compute_performance_metrics(
            rets, risk_free_rate=risk_free_rate, periods_per_year=periods_per_year
        )
    return pd.DataFrame(metrics).T


# ---------------------- CLI Orchestration ---------------------- #

def load_and_prepare(filepath: str,
                     date_col: str = "date") -> pd.DataFrame:
    """
    Load CSV and parse date column if present.
    """
    df = pd.read_csv(filepath)
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col).reset_index(drop=True)
        df.set_index(date_col, inplace=True)
    return df


def build_strategy_returns(df: pd.DataFrame,
                           price_col: str = "close",
                           position_cols: Optional[Dict[str, str]] = None,
                           transaction_cost_bps: float = 0.0) -> Dict[str, pd.Series]:
    """
    Given a DataFrame with price and strategy position columns, compute net returns per strategy.
    position_cols: mapping of strategy_name -> column_name containing positions.
    """
    if position_cols is None:
        position_cols = {
            "ogdm": "ogdm_pos",
            "lstm": "lstm_pos",
            "hybrid": "hybrid_pos",
            "baseline": "baseline_pos"
        }
    missing_price = price_col not in df.columns
    if missing_price:
        raise ValueError(f"Price column '{price_col}' not found")
    strategy_returns = {}
    for strat, col in position_cols.items():
        if col not in df.columns:
            logger.warning(f"Missing column for strategy {strat}: {col}, skipping")
            continue
        strategy_returns[strat] = backtest_from_signals(
            df[price_col], df[col], transaction_cost_bps=transaction_cost_bps
        )
    return strategy_returns


def plot_equity_curves(strategy_returns: Dict[str, pd.Series],
                       output_path: Optional[str] = None):
    """
    Plot cumulative equity curves for strategies.
    """
    if not strategy_returns:
        logger.warning("No strategy returns to plot")
        return
    plt.figure(figsize=(12, 6))
    for name, rets in strategy_returns.items():
        equity = (1 + rets.fillna(0)).cumprod()
        plt.plot(equity.index, equity.values, label=name.upper())
    plt.title("Strategy Equity Curves")
    plt.xlabel("Date")
    plt.ylabel("Equity (Growth of 1)")
    plt.legend()
    plt.grid(alpha=0.3)
    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def save_metrics(metrics_df: pd.DataFrame, filepath: str):
    """
    Save metrics DataFrame to CSV.
    """
    metrics_df.to_csv(filepath, index=True)
    logger.info(f"Saved metrics to {filepath}")


def run_backtest(data_path: str,
                 output_dir: Optional[str],
                 price_col: str,
                 transaction_cost_bps: float,
                 risk_free_rate: float,
                 periods_per_year: int):
    """
    High-level orchestration: load data, compute strategy returns, metrics, outputs.
    """
    df = load_and_prepare(data_path)
    strategy_returns = build_strategy_returns(
        df,
        price_col=price_col,
        transaction_cost_bps=transaction_cost_bps
    )
    metrics_df = compare_strategies(
        strategy_returns,
        risk_free_rate=risk_free_rate,
        periods_per_year=periods_per_year
    )

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        metrics_path = os.path.join(output_dir, "backtest_metrics.csv")
        save_metrics(metrics_df, metrics_path)
        plot_equity_curves(
            strategy_returns,
            output_path=os.path.join(output_dir, "equity_curves.png")
        )

    # Log concise summary
    logger.info("Backtest Performance Metrics:")
    for strat, row in metrics_df.iterrows():
        logger.info(
            f"{strat}: "
            f"AnnRet={row['annualized_return']:.4f} | "
            f"AnnVol={row['annualized_volatility']:.4f} | "
            f"Sharpe={row['sharpe_ratio']:.3f} | "
            f"MaxDD={row['max_drawdown']:.3f}"
        )


def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments.
    """
    parser = argparse.ArgumentParser(
        description="Backtesting framework for OGDM, LSTM, Hybrid, Baseline strategies"
    )
    parser.add_argument("--data_path", type=str, required=True,
                        help="CSV with columns: date, close, ogdm_pos, lstm_pos, hybrid_pos, baseline_pos")
    parser.add_argument("--price_col", type=str, default="close",
                        help="Price column name (default: close)")
    parser.add_argument("--transaction_cost_bps", type=float, default=10.0,
                        help="Transaction cost in basis points per unit turnover (default: 10)")
    parser.add_argument("--risk_free_rate", type=float, default=0.0,
                        help="Annual risk-free rate (decimal, default: 0.0)")
    parser.add_argument("--periods_per_year", type=int, default=252,
                        help="Trading periods per year (default: 252)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save metrics and plots")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_backtest(
        data_path=args.data_path,
        output_dir=args.output_dir,
        price_col=args.price_col,
        transaction_cost_bps=args.transaction_cost_bps,
        risk_free_rate=args.risk_free_rate,
        periods_per_year=args.periods_per_year
    )
