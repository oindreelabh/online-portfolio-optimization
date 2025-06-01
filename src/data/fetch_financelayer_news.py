import os
import requests
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime, timedelta

from src.data.preprocess import preprocess_financelayer
from src.utils.logger import setup_logger
from src.utils.constants import TICKERS
from src.db.write_to_neon import write_df_to_neon

load_dotenv()
logger = setup_logger(os.path.basename(__file__).replace(".py", ""))

API_KEY = os.getenv("FINANCELAYER_API_KEY")

def get_date_range():
    today = datetime.now()
    yesterday = today - timedelta(days=1)
    date_from = yesterday.strftime('%Y-%m-%d')
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
    logger.info(f"Fetched {len(df)} news articles from FinanceLayer.")
    df = preprocess_financelayer(df)
    return df

def fetch_and_store_news():
    df = fetch_financelayer_news(TICKERS, limit=100)
    if not df.empty:
        write_df_to_neon(df, "financial_news_raw")
        logger.info("News data written to Neon DB.")
    else:
        logger.info("No news data to store.")