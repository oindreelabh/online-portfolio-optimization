import os
import logging
from src.utils.helpers import LOG_DIR

def setup_logger(script_name=None, log_level=logging.INFO):
    if script_name is None:
        script_name = os.path.basename(__file__).replace(".py", "")

    log_file = os.path.join(LOG_DIR, f"{script_name}.log")

    logger = logging.getLogger(script_name)
    logger.setLevel(log_level)

    # Avoid duplicate handlers
    if not logger.handlers:
        fh = logging.FileHandler(log_file)
        fh.setLevel(log_level)

        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh.setFormatter(formatter)

        logger.addHandler(fh)

    return logger
