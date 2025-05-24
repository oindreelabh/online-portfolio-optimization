import os

# Root of the whole project (2 levels up from utils folder)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# Common directories
LOG_DIR = os.path.join(ROOT_DIR, "logs")
RAW_DATA_DIR = os.path.join(ROOT_DIR, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(ROOT_DIR, "data", "processed")
EXTERNAL_DATA_DIR = os.path.join(ROOT_DIR, "data", "external")

# Ensure dirs exist
for path in [LOG_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, EXTERNAL_DATA_DIR]:
    os.makedirs(path, exist_ok=True)
