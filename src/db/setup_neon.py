import os
import psycopg2
from src.utils.logger import setup_logger
from src.utils.constants import NEON_DB_URL

logger = setup_logger(os.path.basename(__file__).replace(".py", ""))

def map_dtype(dtype):
    """Map pandas dtype to PostgreSQL dtype."""
    if "int" in str(dtype):
        return "BIGINT"
    elif "float" in str(dtype):
        return "DOUBLE PRECISION"
    elif "bool" in str(dtype):
        return "BOOLEAN"
    elif "datetime" in str(dtype):
        return "TIMESTAMP"
    else:
        return "TEXT"

def create_table_if_not_exists(df, table_name):
    """Create table in Neon DB based on DataFrame schema."""
    try:
        conn = psycopg2.connect(NEON_DB_URL)
        cursor = conn.cursor()

        columns = []
        for col, dtype in df.dtypes.items():
            pg_type = map_dtype(dtype)
            columns.append(f'"{col}" {pg_type}')

        column_definitions = ", ".join(columns)
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            {column_definitions}
        );
        """
        cursor.execute(create_table_query)
        conn.commit()
        cursor.close()
        conn.close()
        logger.info(f"Table `{table_name}` created or already exists.")
    except Exception as e:
        logger.error(f"Failed to create table {table_name}: {e}")
        raise