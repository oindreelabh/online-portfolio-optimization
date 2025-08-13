"""
Portfolio metrics & visualization utility.

INPUT
- prices CSV: date,ticker,close
- optional weights CSV: date,ticker,weight (if omitted, equal-weight each date)

OUTPUT
- portfolio_timeseries.csv (date, equity, drawdown, rolling_vol)
- plots (HTML) if requested

USAGE
python -m src.evaluation.portfolio_metrics \
  --prices-csv data/processed/stock_prices_historical.csv \
  --weights-csv path/to/weights.csv \
  --output-dir evaluation_results/metrics \
  --roll-window 20 \
  --make-plots
"""
import os
import argparse
import numpy as np
import pandas as pd
import plotly.express as px

def load_prices(prices_csv: str) -> pd.DataFrame:
    df = pd.read_csv(prices_csv)
    df["date"] = pd.to_datetime(df["date"])
    return df

def load_weights(weights_csv: str) -> pd.DataFrame:
    df = pd.read_csv(weights_csv)
    df["date"] = pd.to_datetime(df["date"])
    return df

def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["ticker", "date"]).copy()
    df["ret"] = df.groupby("ticker")["close"].pct_change()
    return df

def build_daily_weights(weights_df: pd.DataFrame, tickers: list) -> dict:
    daily = {}
    for d, grp in weights_df.groupby("date"):
        subset = grp.set_index("ticker")["weight"].to_dict()
        total = sum(subset.values())
        if total > 0:
            subset = {k: v / total for k, v in subset.items()}
        else:
            subset = {t: 1 / len(tickers) for t in tickers}
        daily[d] = subset
    return daily

def compute_equity(returns_df: pd.DataFrame, daily_weights: dict, start_equity: float = 1.0) -> pd.DataFrame:
    out = []
    equity = start_equity
    for d in sorted(returns_df["date"].unique()):
        day_r = returns_df[returns_df["date"] == d]
        w_map = daily_weights.get(d)
        if not w_map:
            # fallback equal-weight
            tickers = day_r["ticker"].unique()
            w_map = {t: 1 / len(tickers) for t in tickers}
        merged = day_r[["ticker", "ret"]].copy()
        merged["weight"] = merged["ticker"].map(w_map).fillna(0)
        port_ret = float(np.nansum(merged["weight"] * merged["ret"]))
        equity *= (1 + port_ret)
        out.append({"date": d, "port_ret": port_ret, "equity": equity})
    eq_df = pd.DataFrame(out)
    peak = eq_df["equity"].cummax()
    eq_df["drawdown"] = eq_df["equity"] / peak - 1
    return eq_df

def add_rolling_vol(eq_df: pd.DataFrame, returns_df: pd.DataFrame, daily_weights: dict, window: int) -> pd.DataFrame:
    # For simplicity compute realized portfolio return already present as port_ret
    eq_df["rolling_vol"] = eq_df["port_ret"].rolling(window).std() * np.sqrt(252)
    return eq_df

def make_plots(eq_df: pd.DataFrame, output_dir: str) -> None:
    fig_eq = px.line(eq_df, x="date", y="equity", title="Portfolio Equity")
    fig_dd = px.line(eq_df, x="date", y="drawdown", title="Drawdown")
    fig_vol = px.line(eq_df, x="date", y="rolling_vol", title="Rolling Annualized Vol")
    fig_eq.write_html(os.path.join(output_dir, "equity.html"))
    fig_dd.write_html(os.path.join(output_dir, "drawdown.html"))
    fig_vol.write_html(os.path.join(output_dir, "rolling_vol.html"))

def main():
    p = argparse.ArgumentParser(description="Compute portfolio metrics: equity, drawdowns, rolling volatility.")
    p.add_argument("--prices-csv", required=True)
    p.add_argument("--weights-csv", default=None)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--roll-window", type=int, default=20)
    p.add_argument("--make-plots", action="store_true")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    prices = load_prices(args.prices_csv)
    prices = compute_returns(prices)
    tickers = sorted(prices["ticker"].unique())

    if args.weights_csv and os.path.exists(args.weights_csv):
        weights = load_weights(args.weights_csv)
    else:
        weights = pd.DataFrame({"date": prices["date"].unique().repeat(len(tickers)),
                                "ticker": tickers * len(prices["date"].unique()),
                                "weight": 1 / len(tickers)})

    daily_weights = build_daily_weights(weights, tickers)
    eq_df = compute_equity(prices[["date", "ticker", "ret"]], daily_weights)
    eq_df = add_rolling_vol(eq_df, prices, daily_weights, args.roll_window)

    eq_df.to_csv(os.path.join(args.output_dir, "portfolio_timeseries.csv"), index=False)

    if args.make_plots:
        make_plots(eq_df, args.output_dir)

    print("Saved metrics to", args.output_dir)

if __name__ == "__main__":
    main()
