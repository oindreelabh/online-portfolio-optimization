import os
import subprocess
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# Common directories
LOG_DIR = os.path.join(ROOT_DIR, "logs")

# Ensure dirs exist
for path in [LOG_DIR]:
    os.makedirs(path, exist_ok=True)

def write_df_to_csv(df: pd.DataFrame, directory: str, filename: str):
    os.makedirs(directory, exist_ok=True)
    csv_path = os.path.join(directory, filename)
    df.to_csv(csv_path, index=False)
    return csv_path

def run_command(cmd_list):
    """
    Run a shell command (list form) and return (ok, stdout, stderr).
    """
    try:
        proc = subprocess.run(cmd_list, capture_output=True, text=True, check=False)
        return proc.returncode == 0, proc.stdout, proc.stderr
    except Exception as e:
        return False, "", str(e)

