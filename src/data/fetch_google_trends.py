import os
import pandas as pd
import time
import random
from pytrends.request import TrendReq

from src.utils.constants import TICKERS
from src.utils.logger import setup_logger
from src.db.write_to_neon import write_df_to_neon

logger = setup_logger(os.path.basename(__file__).replace(".py", ""))

def fetch_trends_data(tickers, timeframe='today 3-m'):
    """
    Fetch Google Trends data for given tickers over specified timeframe.
    timeframe examples: 'today 3-m' (last 3 months), 'now 7-d' (last 7 days)
    """
    logger.info(f"Starting Google Trends fetch for tickers: {tickers}")

    pytrends = TrendReq(hl='en-US', tz=360)
    all_data = pd.DataFrame()

    # pytrends allows max 5 keywords per request, so batch if >5
    batch_size = 5
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        try:
            pytrends.build_payload(batch, cat=0, timeframe=timeframe, geo='', gprop='')
            time.sleep(random.uniform(2, 5))  # Sleep 2–5 seconds between requests
            data = pytrends.interest_over_time()
            if data.empty:
                logger.warning(f"No data returned for batch: {batch}")
                continue
            # Drop 'isPartial' column if present
            if 'isPartial' in data.columns:
                data = data.drop(columns=['isPartial'])

            if all_data.empty:
                all_data = data
            else:
                # Join new batch data on index (date)
                all_data = all_data.join(data, how='outer')
            logger.info(f"Fetched data for batch: {batch}")
        except Exception as e:
            logger.error(f"Error fetching trends data for {batch}: {e}")

    return all_data

def main():
    df = fetch_trends_data(TICKERS)
    if not df.empty:
        write_df_to_neon(df, "google_trends")
    else:
        logger.info("No Google Trends data fetched.")

if __name__ == "__main__":
    main()
