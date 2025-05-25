import os

# Root of the whole project (2 levels up from utils folder)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# Common directories
LOG_DIR = os.path.join(ROOT_DIR, "logs")

# Ensure dirs exist
for path in [LOG_DIR]:
    os.makedirs(path, exist_ok=True)
