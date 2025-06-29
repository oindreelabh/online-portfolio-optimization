TICKERS = [
    "AAPL", "TSLA", "AMZN", "MSFT", "NVDA",
    "GME", "AMC", "META","JPM", "SPY",
    "UNH", "GOOGL"
]

SUBREDDITS = [
    "stocks", "wallstreetbets", "investing", "stockmarket", "pennystocks"
]

from dotenv import load_dotenv

load_dotenv()

TABLE_NAME = "stock_prices"