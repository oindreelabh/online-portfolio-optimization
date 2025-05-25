import os
import yfinance as yf
import pandas as pd
from src.utils.logger import setup_logger
from src.db.write_to_neon import write_df_to_neon
from src.utils.constants import TICKERS

logger = setup_logger(os.path.basename(__file__).replace(".py", ""))

def fetch_yfinance_data(tickers: list[str], period="1mo", interval="1d") -> pd.DataFrame:
    try:
        logger.info(f"Fetching yfinance data for tickers: {tickers}")
        df = yf.download(tickers, period=period, interval=interval, group_by='ticker', auto_adjust=False)
        logger.info("Successfully fetched yfinance data.")

        # Flatten multi-index: (Price, Ticker) → separate rows
        flat_df = (
            df.stack(level=0, future_stack=True)  # Make each ticker a row group
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
        return flat_df

    except Exception as e:
        logger.error(f"Error fetching yfinance data: {e}")
        raise

def main():
    try:
        df = fetch_yfinance_data(TICKERS)
        logger.info("Writing data to Neon DB...")
        write_df_to_neon(df)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")

if __name__ == "__main__":
    main()
