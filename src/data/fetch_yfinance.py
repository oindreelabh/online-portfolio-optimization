import os
import yfinance as yf
import pandas as pd
import argparse
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

def fetch_and_store_historical_data(filename, raw_dir, market_ticker='^GSPC'):
    try:
        logger.info("Fetching historical data for last 5 years...")
        # Combine stock tickers with market ticker
        all_tickers = TICKERS + [market_ticker]
        logger.info(f"Including market ticker: {market_ticker}")
        df = fetch_yfinance_data(all_tickers, period="5y", interval="1d")
        csv_path = write_df_to_csv(df, raw_dir, filename)
        logger.info(f"Historical data written successfully to {csv_path}.")
    except Exception as e:
        logger.error(f"Failed to fetch/store historical data: {e}")

def fetch_and_store_latest_data(filename, raw_dir, market_ticker='^GSPC'):
    try:
        logger.info("Fetching latest data for last 1 month...")
        # Combine stock tickers with market ticker
        all_tickers = TICKERS + [market_ticker]
        logger.info(f"Including market ticker: {market_ticker}")
        df = fetch_yfinance_data(all_tickers, period="1mo", interval="1d")
        csv_path = write_df_to_csv(df, raw_dir, filename)
        logger.info(f"Latest data written successfully to {csv_path}.")
    except Exception as e:
        logger.error(f"Failed to fetch/store latest data: {e}")

if __name__ == "__main__":
    logger.info("Starting YFinance data fetching and storing...")
    parser = argparse.ArgumentParser(description="Fetch and store YFinance data.")
    parser.add_argument('--historical', type=str, help="Fetch historical data for last 5 years")
    parser.add_argument('--latest', type=str, help="Fetch latest data for last 1 month")
    parser.add_argument('--raw_dir', type=str, help="Directory for raw data files")
    parser.add_argument('--market_ticker', type=str, default='^GSPC', help="Market index ticker (default: S&P 500)")
    args = parser.parse_args()
    
    fetch_and_store_historical_data(args.historical, args.raw_dir, args.market_ticker)
    fetch_and_store_latest_data(args.latest, args.raw_dir, args.market_ticker)
    logger.info("YFinance data fetching and storing completed.")

