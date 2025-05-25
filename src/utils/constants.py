TICKERS = [
    "AAPL", "TSLA", "AMZN", "MSFT", "NVDA",
    "GME", "AMC", "META","JPM", "SPY",
    "UNH", "C"
]

SUBREDDITS = [
    "stocks", "wallstreetbets", "investing", "stockmarket", "pennystocks"
]

import os
from dotenv import load_dotenv

load_dotenv()

TABLE_NAME = "stock_prices"
NEON_DB_URL = os.getenv("NEON_DB_URL")