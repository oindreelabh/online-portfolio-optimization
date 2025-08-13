"""
Sentiment influence analysis.

INPUT
data CSV (wide or long) containing at least:
  date, ticker, close, reddit_sentiment, news_sentiment (names configurable)
predictions CSV:
  date, ticker, model, predicted_return
(Optional) allocations CSV:
  date, ticker, weight   (if provided, compute allocation deltas)

OUTPUT
- sentiment_correlations.csv (pearson correlations)
- sentiment_effects.html (scatter + regression fits)

USAGE
python -m src.analysis.sentiment_influence \
  --data-csv data/processed/recent_data_with_sentiment.csv \
  --predictions-csv evaluation_results/perf/predictions_long.csv \
  --model HYBRID \
  --reddit-col reddit_sentiment \
  --news-col news_sentiment \
  --allocations-csv path/to/weights.csv \
  --output-dir analysis_results/sentiment \
  --lag-days 1
"""
import os
import argparse
import numpy as np
import pandas as pd
import plotly.express as px

def prepare_data(data_csv: str,
                 predictions_csv: str,
                 model: str,
                 reddit_col: str,
                 news_col: str,
                 lag_days: int) -> pd.DataFrame:
    data = pd.read_csv(data_csv)
    preds = pd.read_csv(predictions_csv)
    data["date"] = pd.to_datetime(data["date"])
    preds["date"] = pd.to_datetime(preds["date"])
    preds = preds[preds["model"] == model]
    # Keep needed columns
    keep_cols = ["date", "ticker", reddit_col, news_col]
    missing = set(keep_cols).difference(data.columns)
    if missing:
        raise ValueError(f"Missing sentiment columns: {missing}")
    merged = preds.merge(data[keep_cols], on=["date", "ticker"], how="left")

    # Lag sentiment if desired (sentiment leading predictions)
    if lag_days > 0:
        for c in [reddit_col, news_col]:
            merged[f"{c}_lag"] = merged.groupby("ticker")[c].shift(lag_days)
    else:
        for c in [reddit_col, news_col]:
            merged[f"{c}_lag"] = merged[c]
    return merged

def add_allocation_deltas(df: pd.DataFrame, allocations_csv: str) -> pd.DataFrame:
    if not allocations_csv or not os.path.exists(allocations_csv):
        return df
    alloc = pd.read_csv(allocations_csv)
    alloc["date"] = pd.to_datetime(alloc["date"])
    alloc = alloc.sort_values(["ticker", "date"])
    alloc["weight_prev"] = alloc.groupby("ticker")["weight"].shift(1)
    alloc["alloc_delta"] = alloc["weight"] - alloc["weight_prev"]
    df = df.merge(alloc[["date", "ticker", "weight", "alloc_delta"]],
                  on=["date", "ticker"], how="left")
    return df

def compute_correlations(df: pd.DataFrame,
                         reddit_col: str,
                         news_col: str) -> pd.DataFrame:
    targets = []
    if "predicted_return" in df.columns:
        targets.append("predicted_return")
    if "alloc_delta" in df.columns:
        targets.append("alloc_delta")
    rows = []
    for sentiment_source in [f"{reddit_col}_lag", f"{news_col}_lag"]:
        for target in targets:
            sub = df[[sentiment_source, target]].dropna()
            if sub.empty:
                corr = np.nan
            else:
                corr = sub.corr().iloc[0, 1]
            rows.append({"sentiment_feature": sentiment_source,
                         "target": target,
                         "pearson_corr": corr})
    return pd.DataFrame(rows)

def make_plots(df: pd.DataFrame,
               reddit_col: str,
               news_col: str,
               output_dir: str) -> None:
    for senti in [f"{reddit_col}_lag", f"{news_col}_lag"]:
        if "predicted_return" in df.columns:
            fig = px.scatter(df, x=senti, y="predicted_return",
                             trendline="ols",
                             title=f"{senti} vs predicted_return")
            fig.write_html(os.path.join(output_dir, f"{senti}_predicted_return.html"))
        if "alloc_delta" in df.columns:
            fig2 = px.scatter(df, x=senti, y="alloc_delta",
                              trendline="ols",
                              title=f"{senti} vs alloc_delta")
            fig2.write_html(os.path.join(output_dir, f"{senti}_alloc_delta.html"))

def main():
    ap = argparse.ArgumentParser(description="Analyze influence of sentiment on predictions / allocations.")
    ap.add_argument("--data-csv", required=True)
    ap.add_argument("--predictions-csv", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--allocations-csv", default=None)
    ap.add_argument("--reddit-col", default="reddit_sentiment")
    ap.add_argument("--news-col", default="news_sentiment")
    ap.add_argument("--lag-days", type=int, default=1)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    df = prepare_data(args.data_csv, args.predictions_csv, args.model,
                      args.reddit_col, args.news_col, args.lag_days)
    df = add_allocation_deltas(df, args.allocations_csv)
    corr_df = compute_correlations(df, args.reddit_col, args.news_col)
    corr_df.to_csv(os.path.join(args.output_dir, "sentiment_correlations.csv"), index=False)
    make_plots(df, args.reddit_col, args.news_col, args.output_dir)
    print("Saved sentiment influence outputs to", args.output_dir)

if __name__ == "__main__":
    main()
