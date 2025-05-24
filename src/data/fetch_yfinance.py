import yfinance as yf
import pandas as pd
import os
import logging
from datetime import datetime

from src.utils.constants import TICKERS
from src.utils.helpers import LOG_DIR, RAW_DATA_DIR

print("Logs folder is at:", LOG_DIR)
print("Raw data folder is at:", RAW_DATA_DIR)

# Setup logging
log_file_path = os.path.join(LOG_DIR, "fetch_yfinance.log")
logging.basicConfig(
    filename=log_file_path,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def fetch_stock_data(ticker: str, start_date="2018-01-01", end_date=None, interval="1d") -> pd.DataFrame:
    """
    Fetch historical stock data for a given ticker using yfinance.
    """
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')

    logging.info(f"Fetching data for {ticker} from {start_date} to {end_date}")
    df = yf.download(ticker, start=start_date, end=end_date, interval=interval)

    if df.empty:
        logging.warning(f"No data found for {ticker}")
        return pd.DataFrame()

    df.reset_index(inplace=True)
    df["Ticker"] = ticker
    return df

def fetch_and_merge_all_tickers(output_path=os.path.join(RAW_DATA_DIR, "merged_stock_data.csv")):
    """
    Fetch data for all tickers, merge into one DataFrame, and save as single CSV.
    """
    tickers = TICKERS

    all_data = []

    for ticker in tickers:
        df = fetch_stock_data(ticker)
        if not df.empty:
            all_data.append(df)

    if all_data:
        merged_df = pd.concat(all_data, ignore_index=True)
        merged_df.to_csv(output_path, index=False)
        logging.info(f"Saved merged data for {len(tickers)} tickers to {output_path}")
    else:
        logging.warning("No data collected for any tickers. CSV not written.")

# If run as a script
if __name__ == "__main__":
    fetch_and_merge_all_tickers()
