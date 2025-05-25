# src/db/write_to_neon.py

import os
import psycopg2
from src.utils.logger import setup_logger
from src.db.setup_neon import create_table_if_not_exists

logger = setup_logger(os.path.basename(__file__).replace(".py", ""))

def write_df_to_neon(df, table_name):
    """Overwrite the table in Neon DB with a new DataFrame."""
    if df.empty:
        logger.warning("Attempted to write an empty DataFrame. Skipping.")
        return

    try:
        create_table_if_not_exists(df, table_name)

        conn = psycopg2.connect(os.getenv("NEON_DB_URL"))
        cursor = conn.cursor()

        # Truncate table to clear old data
        truncate_query = f"TRUNCATE TABLE {table_name};"
        cursor.execute(truncate_query)
        logger.info(f"Table `{table_name}` truncated before inserting new data.")

        # Prepare insert
        columns = list(df.columns)
        placeholders = ", ".join(["%s"] * len(columns))
        column_names = ", ".join([f'"{col}"' for col in columns])
        insert_query = f"""
        INSERT INTO {table_name} ({column_names})
        VALUES ({placeholders});
        """
        data = [tuple(row) for row in df.itertuples(index=False, name=None)]
        cursor.executemany(insert_query, data)

        conn.commit()
        cursor.close()
        conn.close()
        logger.info(f"Successfully wrote {len(df)} rows to `{table_name}`.")
    except Exception as e:
        logger.error(f"Failed to overwrite data in table `{table_name}`: {e}")
        raise