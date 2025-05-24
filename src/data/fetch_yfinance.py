import os
import pandas as pd
import yfinance as yf
from datetime import datetime

from src.utils.logger import setup_logger
from src.utils.helpers import RAW_DATA_DIR
from src.utils.constants import TICKERS

logger = setup_logger(os.path.basename(__file__).replace(".py", ""))


def fetch_yfinance_data(tickers):
    """
    Fetches historical stock data for a list of tickers from Yahoo Finance,
    flattens the data, and saves it to a CSV file.

    Args:
        tickers (list): A list of stock ticker symbols (e.g., ["AAPL", "MSFT"]).

    Returns:
        str: The file path of the saved CSV, or None if no data was fetched or an error occurred.
    """
    if not tickers:
        logger.warning("No tickers provided to fetch_yfinance_data.")
        return None

    try:
        logger.info(f"Fetching yfinance data for: {', '.join(tickers)}")

        # Download data for all tickers at once.
        # This will return a DataFrame with a MultiIndex for columns (e.g., (Open, AAPL), (High, TSLA)).
        df_multi_index = yf.download(tickers, period="1mo", interval="1d")

        if df_multi_index.empty:
            logger.warning("No data returned for the specified tickers from yfinance.")
            return None

        # Flatten the MultiIndex columns into a long (tidy) format.
        # 1. `stack(level=0)`: This moves the top level of the MultiIndex (the metrics like 'Open', 'High', 'Close')
        #    from columns to a new level in the DataFrame's index.
        #    The columns will now be the ticker symbols.
        # 2. `reset_index()`: This converts all levels of the index (including 'Date' and the newly stacked metric)
        #    into regular columns.
        combined_df = df_multi_index.stack(level=0).reset_index()

        # Rename the columns for clarity.
        # 'level_1' typically becomes the ticker symbol after stacking the metric.
        # The original metric names ('Open', 'High', etc.) will be the new columns.
        combined_df.rename(columns={'level_1': 'ticker'}, inplace=True)

        # Drop 'Adj Close' if it exists and is not desired in the final output.
        # yfinance often includes 'Adj Close' by default.
        if 'Adj Close' in combined_df.columns:
            combined_df.drop(columns=['Adj Close'], inplace=True)

        combined_df.head(10)

        # Define the desired order of columns for the final CSV.
        final_columns_order = ["Date", "Open", "High", "Low", "Close", "Volume", "ticker"]

        # Filter to only include columns that actually exist in combined_df
        # and ensure they are in the desired order.
        existing_final_columns = [col for col in final_columns_order if col in combined_df.columns]
        combined_df = combined_df[existing_final_columns]

        # Save the combined, flattened DataFrame to a CSV file.
        file_name = f"yfinance_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        file_path = os.path.join(RAW_DATA_DIR, file_name)
        combined_df.to_csv(file_path, index=False)
        logger.info(f"Successfully saved yfinance data to {file_path}")
        return file_path

    except Exception as e:
        logger.error(f"An error occurred while fetching or processing yfinance data: {e}")
        return None


if __name__ == "__main__":
    # IMPORTANT: Ensure your `TICKERS` variable (from src.utils.constants)
    # is a list of individual string ticker symbols, e.g., ["AAPL", "MSFT", "GOOGL"].
    # If it's a single string like "AAPL,MSFT,GOOGL", you'll need to split it:
    # TICKERS = TICKERS.split(',')

    # Run the data fetching process
    output_file = fetch_yfinance_data(TICKERS)
    if output_file:
        print(f"Data successfully fetched and saved to: {output_file}")
    else:
        print("Failed to fetch yfinance data.")
