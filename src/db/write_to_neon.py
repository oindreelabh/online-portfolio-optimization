# src/db/write_to_neon.py

import os
import psycopg2
from io import StringIO
import pandas as pd
from src.utils.logger import setup_logger
from src.utils.constants import TABLE_NAME, NEON_DB_URL

logger = setup_logger(os.path.basename(__file__).replace(".py", ""))

def write_df_to_neon(df: pd.DataFrame):
    """
    Write a DataFrame with columns: date, ticker, open, high, low, close, volume
    to Neon PostgreSQL database.
    """
    try:
        logger.info(f"Starting data write to Neon table `{TABLE_NAME}`...")

        # Convert DataFrame to CSV-like format in memory
        buffer = StringIO()
        df.to_csv(buffer, index=False, header=False)
        buffer.seek(0)

        # DB connection and copy insert
        conn = psycopg2.connect(NEON_DB_URL)
        cur = conn.cursor()
        cur.copy_from(buffer, TABLE_NAME, sep=",", columns=("date", "ticker", "open", "high", "low", "close", "volume"))
        conn.commit()
        cur.close()
        conn.close()

        logger.info("✅ Data successfully written to Neon.")
    except Exception as e:
        logger.error(f"❌ Error writing to Neon: {e}")
        raise
