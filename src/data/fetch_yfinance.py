import os
import yfinance as yf
import pandas as pd
from src.utils.logger import setup_logger
from src.utils.helpers import write_df_to_csv
from src.utils.constants import TICKERS
from src.data.preprocess import preprocess_yfinance

logger = setup_logger(os.path.basename(__file__).replace(".py", ""))

def fetch_yfinance_data(tickers: list[str], period="1mo", interval="1d") -> pd.DataFrame:
    try:
        logger.info(f"Fetching yfinance data for tickers: {tickers} | Period: {period}")
        df = yf.download(tickers, period=period, interval=interval, group_by='ticker', auto_adjust=False)
        logger.info("Successfully fetched yfinance data.")

        flat_df = (
            df.stack(level=0, future_stack=True)
            .reset_index()
            .rename(columns={
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
                "Ticker": "ticker"
            })[["date", "ticker", "open", "high", "low", "close", "volume"]]
        )
        logger.info("Successfully transformed yfinance data to flat format.")
        flat_df = preprocess_yfinance(flat_df)
        logger.info("Successfully preprocessed yfinance data.")
        return flat_df

    except Exception as e:
        logger.error(f"Error fetching yfinance data: {e}")
        raise

def fetch_and_store_historical_data():
    try:
        logger.info("Fetching historical data for last 2 years...")
        df = fetch_yfinance_data(TICKERS, period="2y", interval="1d")
        csv_path = write_df_to_csv(df, "../../data/raw", "stock_prices_historical.csv")
        logger.info(f"Historical data written successfully to {csv_path}.")
    except Exception as e:
        logger.error(f"Failed to fetch/store historical data: {e}")

def fetch_and_store_latest_data():
    try:
        logger.info("Fetching latest data for last 1 week...")
        df = fetch_yfinance_data(TICKERS, period="1mo", interval="1d")
        csv_path = write_df_to_csv(df, "../../data/raw", "stock_prices_latest.csv")
        logger.info(f"Latest data written successfully to {csv_path}.")
    except Exception as e:
        logger.error(f"Failed to fetch/store latest data: {e}")

# if __name__ == "__main__":
#     fetch_and_store_historical_data()
#     fetch_and_store_latest_data()
#     logger.info("YFinance data fetching and storing completed.")

