import pandas as pd

from src.features.finbert_sentiment import add_finbert_sentiment
from src.utils.logger import setup_logger
import os

logger = setup_logger(os.path.basename(__file__).replace(".py", ""))


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
    # Ensure 'ticker' and 'date' columns are in datetime format for merging
    yf_df['date'] = pd.to_datetime(yf_df['date'])
    reddit_agg['date'] = pd.to_datetime(reddit_agg['date'])
    news_agg['date'] = pd.to_datetime(news_agg['date'])
    merged = yf_df.merge(reddit_agg, on=['ticker', 'date'], how='left')
    merged = merged.merge(news_agg, on=['ticker', 'date'], how='left')
    merged['reddit_score'] = merged['reddit_score'].fillna(0)
    merged['news_score'] = merged['news_score'].fillna(0)
    return merged


def run_merge_pipeline(
    yf_path: str, reddit_path: str, news_path: str, output_path: str
):
    """
    Load CSVs, aggregate sentiment, merge, and save to output_path.
    Handles 'tickers' column in reddit and news data.
    """
    # Ensure 'tickers' column is loaded as list, and parse 'date' as datetime
    yf_df = pd.read_csv(yf_path, parse_dates=['date'])
    reddit_df = pd.read_csv(reddit_path, converters={'tickers': eval}, parse_dates=['date'])
    news_df = pd.read_csv(news_path, converters={'tickers': eval}, parse_dates=['date'])

    reddit_agg = aggregate_sentiment(reddit_df, 'reddit_score')
    news_agg = aggregate_sentiment(news_df, 'news_score')

    merged_df = merge_features(yf_df, reddit_agg, news_agg)
    merged_df.to_csv(output_path, index=False)


