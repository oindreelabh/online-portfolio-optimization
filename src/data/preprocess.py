# src/data/preprocess.py

import pandas as pd
import re
from nltk.corpus import stopwords
import os
import argparse
from src.utils.logger import setup_logger

stop_words = set(stopwords.words('english'))

logger = setup_logger(os.path.basename(__file__).replace(".py", ""))

def preprocess_yfinance(df: pd.DataFrame) -> pd.DataFrame:
    # Fill missing values, calculate returns, remove outliers, etc.
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['ticker', 'date'])
    df = df.groupby('ticker').apply(lambda x: x.interpolate()).reset_index(drop=True)
    df['returns'] = df.groupby('ticker')['close'].pct_change()
    q_low = df['returns'].quantile(0.01)
    q_high = df['returns'].quantile(0.99)
    df = df[(df['returns'] > q_low) & (df['returns'] < q_high)]
    return df.reset_index(drop=True)

def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = text.lower()
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words]
    return " ".join(tokens)

def preprocess_reddit(df: pd.DataFrame) -> pd.DataFrame:
    df['cleaned_text'] = df['full_text'].apply(clean_text)
    return df

def preprocess_financelayer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates(subset=['title', 'published_at'])
    df = df.dropna(subset=['title', 'description'])
    df['published_at'] = pd.to_datetime(df['published_at'], errors='coerce')
    df['cleaned_text'] = df['title'].fillna('') + ' ' + df['description'].fillna('')
    df['cleaned_text'] = df['cleaned_text'].apply(clean_text)
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess raw data files.")
    parser.add_argument('--yfinance_y2', action='store_true', help="Preprocess yfinance data")
    parser.add_argument('--yfinance_new', action='store_true', help="Preprocess yfinance recent data")
    parser.add_argument('--reddit', action='store_true', help="Preprocess Reddit posts")
    parser.add_argument('--financelayer', action='store_true', help="Preprocess FinanceLayer news")
    parser.add_argument('--raw_dir', type=str, default='data/raw', help="Directory for raw data files")
    parser.add_argument('--processed_dir', type=str, default='data/processed', help="Directory for processed data files")
    args = parser.parse_args()

    if not (args.yfinance_y2 or args.yfinance_new or args.reddit or args.financelayer):
        parser.error("At least one of --yfinance_y2, --yfinance_new, --reddit, or --financelayer must be specified.")

    try:
        yf_df_hist = pd.read_csv(f'{args.raw_dir}/{args.yfinance_y2}')
        yf_df_recent = pd.read_csv(f'{args.raw_dir}/{args.yfinance_new}')
        yf_df_hist = preprocess_yfinance(yf_df_hist)
        yf_df_recent = preprocess_yfinance(yf_df_recent)
        yf_df_hist.save(f'{args.processed_dir}/{args.yfinance_y2}', index=False)
        yf_df_recent.save(f'{args.processed_dir}/{args.yfinance_new}', index=False)
    except FileNotFoundError as e:
        logger.error(f"Error: {e}. Please ensure the raw data files exist in the specified path.")

    try:
        reddit_df = pd.read_csv(f'{args.raw_dir}/{args.reddit}')
        reddit_df = preprocess_reddit(reddit_df)
        reddit_df.to_csv(f'{args.processed_dir}/{args.reddit}', index=False)
    except FileNotFoundError as e:
        logger.error(f"Error: {e}. Please ensure the raw Reddit posts file exists in the specified path.")

    try:
        finlayer_df = pd.read_csv(f'{args.raw_dir}/{args.financelayer}')
        finlayer_df = preprocess_financelayer(finlayer_df)
        finlayer_df.to_csv(f'{args.processed_dir}/{args.financelayer}', index=False)
    except FileNotFoundError as e:
        logger.error(f"Error: {e}. Please ensure the raw FinanceLayer news file exists in the specified path.")
