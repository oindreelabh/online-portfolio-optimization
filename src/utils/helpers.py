import os

# Root of the whole project (2 levels up from utils folder)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# Common directories
LOG_DIR = os.path.join(ROOT_DIR, "logs")

# Ensure dirs exist
for path in [LOG_DIR]:
    os.makedirs(path, exist_ok=True)

import pandas as pd

def write_df_to_csv(df: pd.DataFrame, directory: str, filename: str) -> str:
    """
    Save a DataFrame as a CSV file in the specified directory.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        directory (str): Directory path where the CSV will be saved.
        filename (str): Name of the CSV file.

    Returns:
        str: The full path to the saved CSV file.
    """
    os.makedirs(directory, exist_ok=True)
    csv_path = os.path.join(directory, filename)
    df.to_csv(csv_path, index=False)
    return csv_path

