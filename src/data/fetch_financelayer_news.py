import os
import requests
import pandas as pd
import argparse
from dotenv import load_dotenv
from datetime import datetime, timedelta

from src.data.preprocess import preprocess_financelayer
from src.utils.logger import setup_logger
from src.utils.constants import TICKERS
from src.utils.helpers import write_df_to_csv

load_dotenv()
logger = setup_logger(os.path.basename(__file__).replace(".py", ""))

API_KEY = os.getenv("FINANCELAYER_API_KEY")

def get_date_range():
    today = datetime.now()
    prev_day = today - timedelta(days=30)  # Fetch news from the last 30 days
    date_from = prev_day.strftime('%Y-%m-%d')
    date_to = today.strftime('%Y-%m-%d')
    return date_from, date_to

def fetch_financelayer_news(keywords, limit=100):
    url = "https://api.apilayer.com/financelayer/news"
    headers = {"apikey": API_KEY}
    date_from, date_to = get_date_range()
    params = {
        "keywords": ",".join(keywords),
        "dateFrom": date_from,
        "dateTo": date_to,
        "limit": limit
    }
    response = requests.get(url, headers=headers, params=params)
    data = response.json()
    articles = data.get("data", [])
    if not articles:
        logger.warning("No news articles found.")
        return pd.DataFrame()
    df = pd.DataFrame(articles)
    # Add a 'date' column from the published date field in the API response
    if 'published_at' in df.columns:
        df['date'] = pd.to_datetime(df['published_at']).dt.date
    elif 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date']).dt.date
    else:
        # If no date field, use current date as fallback
        df['date'] = pd.Timestamp.now().date()
    logger.info(f"Fetched {len(df)} news articles from FinanceLayer.")
    df = preprocess_financelayer(df)
    return df

def fetch_and_store_news(filename):
    df = fetch_financelayer_news(TICKERS, limit=100)
    if not df.empty:
        csv_path = write_df_to_csv(df, "../../data/raw", filename)
        logger.info(f"Latest data written successfully to {csv_path}.")
    else:
        logger.info("No news data to store.")
#
if __name__ == "__main__":
    logger.info("Starting FinanceLayer news fetching and storing...")
    parser = argparse.ArgumentParser(description="Fetch and store FinanceLayer news data.")
    parser.add_argument("--filename", type=str, default="financelayer_news.csv", help="Filename to save the fetched news")
    args = parser.parse_args()
    fetch_and_store_news(args.filename)
    logger.info("FinanceLayer news fetching and storing completed.")