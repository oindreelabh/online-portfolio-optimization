from src.utils.logger import setup_logger
import os

import pandas as pd
from datetime import datetime, timedelta
import requests
from dotenv import load_dotenv

from src.utils.constants import TICKERS
from src.utils.helpers import RAW_DATA_DIR

logger = setup_logger(os.path.basename(__file__).replace(".py", ""))

# Load environment variables
load_dotenv()
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")


HEADERS = {"X-Finnhub-Token": FINNHUB_API_KEY}
BASE_URL = "https://finnhub.io/api/v1"

# Time range
END_DATE = datetime.today()
START_DATE = END_DATE - timedelta(days=30)  # last 30 days
FROM_UNIX = int(START_DATE.timestamp())
TO_UNIX = int(END_DATE.timestamp())


def fetch_candles(symbol):
    url = f"{BASE_URL}/stock/candle"
    params = {"symbol": symbol, "resolution": "D", "from": FROM_UNIX, "to": TO_UNIX}
    r = requests.get(url, headers=HEADERS, params=params)
    data = r.json()

    if data.get("s") != "ok":
        logger.warning(f"No candle data for {symbol}")
        return pd.DataFrame()

    df = pd.DataFrame({
        "date": pd.to_datetime(data["t"], unit='s'),
        "open": data["o"],
        "high": data["h"],
        "low": data["l"],
        "close": data["c"],
        "volume": data["v"]
    })
    df["ticker"] = symbol
    return df


def fetch_earnings(symbol):
    url = f"{BASE_URL}/stock/earnings"
    params = {"symbol": symbol}
    r = requests.get(url, headers=HEADERS, params=params)
    data = r.json()
    df = pd.DataFrame(data)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = symbol
    return df[["ticker", "date", "epsActual", "epsEstimate", "revenueActual", "revenueEstimate"]]


def fetch_insider_trades(symbol):
    url = f"{BASE_URL}/stock/insider-transactions"
    params = {"symbol": symbol, "from": START_DATE.strftime("%Y-%m-%d"), "to": END_DATE.strftime("%Y-%m-%d")}
    r = requests.get(url, headers=HEADERS, params=params)
    data = r.json().get("data", [])
    df = pd.DataFrame(data)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["transactionDate"])
    df["ticker"] = symbol
    return df[["ticker", "date", "name", "transactionType", "share", "price"]]


def fetch_news(symbol):
    url = f"{BASE_URL}/company-news"
    params = {"symbol": symbol, "from": START_DATE.strftime("%Y-%m-%d"), "to": END_DATE.strftime("%Y-%m-%d")}
    r = requests.get(url, headers=HEADERS, params=params)
    data = r.json()
    df = pd.DataFrame(data)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["datetime"], unit="s").dt.date
    df["ticker"] = symbol
    return df[["ticker", "date", "headline", "summary", "source", "url"]]


def fetch_all():
    all_data = []

    for symbol in TICKERS:
        logger.info(f"Fetching Finnhub data for {symbol}")

        try:
            df_candle = fetch_candles(symbol)
            df_earnings = fetch_earnings(symbol)
            df_insider = fetch_insider_trades(symbol)
            df_news = fetch_news(symbol)

            # Merge all on ticker + date
            merged_df = df_candle.copy()
            for df_other in [df_earnings, df_insider, df_news]:
                if not df_other.empty:
                    merged_df = pd.merge(
                        merged_df, df_other,
                        on=["ticker", "date"],
                        how="left"
                    )

            all_data.append(merged_df)

        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")

    # Combine all symbols
    if all_data:
        result_df = pd.concat(all_data, ignore_index=True)
        filename = f"finnhub_data_merged_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        file_path = os.path.join(RAW_DATA_DIR, filename)
        result_df.to_csv(file_path, index=False)
        logger.info(f"Saved merged Finnhub data to {file_path}")
    else:
        logger.warning("No data collected.")


if __name__ == "__main__":
    fetch_all()
