import os
import psycopg2
from src.utils.logger import setup_logger
from src.utils.constants import TABLE_NAME, NEON_DB_URL

logger = setup_logger(os.path.basename(__file__).replace(".py", ""))

CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
    date DATE NOT NULL,
    ticker TEXT NOT NULL,
    open DOUBLE PRECISION,
    high DOUBLE PRECISION,
    low DOUBLE PRECISION,
    close DOUBLE PRECISION,
    volume BIGINT,
    PRIMARY KEY (date, ticker)
);
"""

def create_table():
    try:
        logger.info("Connecting to Neon database...")
        conn = psycopg2.connect(NEON_DB_URL)
        cur = conn.cursor()
        cur.execute(CREATE_TABLE_SQL)
        conn.commit()
        cur.close()
        conn.close()
        logger.info(f"✅ Table `{TABLE_NAME}` ensured in Neon DB.")
    except Exception as e:
        logger.error(f"❌ Failed to create table: {e}")
        raise

create_table()