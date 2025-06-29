import pandas as pd

from src.utils.logger import setup_logger
import os
from src.features.finbert_sentiment import add_finbert_sentiment
from src.features.finvader_sentiment import add_finvader_sentiment
from src.utils.helpers import write_df_to_csv
import argparse

logger = setup_logger(os.path.basename(__file__).replace(".py", ""))

def add_dual_sentiment(input_df: pd.DataFrame, text_col: str, sentiment_prefix: str) -> pd.DataFrame:
    """
    Adds FinBERT and Llama-2 sentiment columns and their average.
    """
    logger.info("Applying FinBERT sentiment...")
    df = add_finbert_sentiment(input_df, text_col, prefix="finbert")
    logger.info("Applying finvader financial sentiment...")
    df = add_finvader_sentiment(df, text_col, prefix="finvader")
    logger.info("Calculating average sentiment score...")
    df[f"{sentiment_prefix}_sentiment_score"] = df[["finbert_score", "finvader_score"]].mean(axis=1)
    logger.info("Dual sentiment annotation completed successfully.")
    return df

def aggregate_sentiment(df: pd.DataFrame, sentiment_col: str, group_cols=None) -> pd.DataFrame:
    """
    Aggregate sentiment scores by ticker and date.
    Handles 'tickers' column as list by exploding it to individual rows with 'ticker'.
    Converts 'date' to datetime64[ns] for compatibility.
    """
    if group_cols is None:
        group_cols = ['ticker', 'date']
    df['date'] = pd.to_datetime(df['date'])
    if 'tickers' in df.columns:
        df = df.explode('tickers')
        df = df.rename(columns={'tickers': 'ticker'})
    agg_df = df.groupby(group_cols)[sentiment_col].mean().reset_index()
    return agg_df



def merge_features(yf_df: pd.DataFrame, reddit_agg: pd.DataFrame, news_agg: pd.DataFrame) -> pd.DataFrame:
    """
    Merge yfinance data with aggregated Reddit and news sentiment scores.
    """
    yf_df['date'] = pd.to_datetime(yf_df['date'])
    reddit_agg['date'] = pd.to_datetime(reddit_agg['date'])
    news_agg['date'] = pd.to_datetime(news_agg['date'])
    merged = yf_df.merge(reddit_agg, on=['ticker', 'date'], how='left')
    merged = merged.merge(news_agg, on=['ticker', 'date'], how='left')
    merged['reddit_sentiment_score'] = merged['reddit_sentiment_score'].fillna(0)
    merged['news_sentiment_score'] = merged['news_sentiment_score'].fillna(0)
    return merged


def run_merge_pipeline(
    yf_path: str, reddit_path: str, news_path: str, output_dir: str, output_file: str
):
    """
    Load CSVs, aggregate sentiment, merge, and save to output_path.
    Handles 'tickers' column in reddit and news data.
    """
    yf_df = pd.read_csv(yf_path, parse_dates=['date'])
    reddit_df = pd.read_csv(reddit_path, converters={'tickers': eval}, parse_dates=['date'])
    news_df = pd.read_csv(news_path, converters={'tickers': eval}, parse_dates=['date'])

    reddit_df = add_dual_sentiment(reddit_df, "cleaned_text", "reddit")
    news_df = add_dual_sentiment(news_df, "cleaned_text", "news")

    reddit_agg = aggregate_sentiment(reddit_df, 'reddit_sentiment_score')
    news_agg = aggregate_sentiment(news_df, 'news_sentiment_score')

    merged_df = merge_features(yf_df, reddit_agg, news_agg)
    write_df_to_csv(merged_df, output_dir, output_file)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run feature merging pipeline")
    parser.add_argument("--yf_path", type=str, required=True, help="Path to yfinance data CSV")
    parser.add_argument("--reddit_path", type=str, required=True, help="Path to Reddit data CSV")
    parser.add_argument("--news_path", type=str, required=True, help="Path to news data CSV")
    parser.add_argument("--output_dir", type=str, required=True, help="Output path for merged data CSV")
    parser.add_argument("--output_file", type=str, required=True, help="Output path for merged data CSV")

    args = parser.parse_args()

    run_merge_pipeline(args.yf_path, args.reddit_path, args.news_path, args.output_dir, args.output_file)
    logger.info("Feature merging pipeline completed successfully.")


