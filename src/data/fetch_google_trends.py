import os
import logging
from datetime import datetime
import pandas as pd
from pytrends.request import TrendReq

from src.utils.constants import TICKERS
from src.utils.helpers import LOG_DIR, RAW_DATA_DIR

print("Logs folder is at:", LOG_DIR)
print("Raw data folder is at:", RAW_DATA_DIR)

# Setup logging
log_file_path = os.path.join(LOG_DIR, "fetch_google_trends.log")
logging.basicConfig(
    filename=log_file_path,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def fetch_trends_data(tickers, timeframe='today 3-m'):
    """
    Fetch Google Trends data for given tickers over specified timeframe.
    timeframe examples: 'today 3-m' (last 3 months), 'now 7-d' (last 7 days)
    """
    logging.info(f"Starting Google Trends fetch for tickers: {tickers}")

    pytrends = TrendReq(hl='en-US', tz=360)
    all_data = pd.DataFrame()

    # pytrends allows max 5 keywords per request, so batch if >5
    batch_size = 5
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        try:
            pytrends.build_payload(batch, cat=0, timeframe=timeframe, geo='', gprop='')
            data = pytrends.interest_over_time()
            if data.empty:
                logging.warning(f"No data returned for batch: {batch}")
                continue
            # Drop 'isPartial' column if present
            if 'isPartial' in data.columns:
                data = data.drop(columns=['isPartial'])

            if all_data.empty:
                all_data = data
            else:
                # Join new batch data on index (date)
                all_data = all_data.join(data, how='outer')
            logging.info(f"Fetched data for batch: {batch}")
        except Exception as e:
            logging.error(f"Error fetching trends data for {batch}: {e}")

    return all_data


def save_data(df, output_dir=EXTERNAL_DATA_DIR):
    """
    Save the merged Google Trends data with timestamped filename.
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"google_trends_{timestamp}.csv"
    filepath = os.path.join(output_dir, filename)
    df.to_csv(filepath)
    logging.info(f"Saved Google Trends data to {filepath}")

def main():
    df = fetch_trends_data(TICKERS)
    if not df.empty:
        save_data(df)
    else:
        logging.info("No Google Trends data fetched.")

if __name__ == "__main__":
    main()
