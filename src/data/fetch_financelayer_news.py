import os
import requests
import pandas as pd
from dotenv import load_dotenv
from src.utils.logger import setup_logger
from src.utils.constants import TICKERS
from src.db.write_to_neon import write_df_to_neon

load_dotenv()
logger = setup_logger(os.path.basename(__file__).replace(".py", ""))

API_KEY = os.getenv("FINANCELAYER_API_KEY")

def fetch_financelayer_news(keywords, limit=50, date="today"):
    url = "https://api.apilayer.com/financelayer/news"
    headers = {"apikey": API_KEY}
    params = {
        "keywords": ",".join(keywords),
        "date": date,
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
    return df

def fetch_and_store_news():
    df = fetch_financelayer_news(TICKERS, limit=50, date="today")
    if not df.empty:
        write_df_to_neon(df, "financial_news_raw")
        logger.info("News data written to Neon DB.")
    else:
        logger.info("No news data to store.")