import os
import json
import requests
import pandas as pd
from dotenv import load_dotenv

from src.utils.logger import setup_logger

logger = setup_logger(os.path.basename(__file__).replace(".py", ""))

load_dotenv()

OPENFIGI_KEY = os.getenv("OPENFIGI_API_KEY")
OPENFIGI_URL = "https://api.openfigi.com/v3/mapping"

def build_ticker_keyword_map_openfigi(tickers: list[str], raw_dir):
    fallback_file = os.path.join(raw_dir, "ticker_keyword_map.csv")
    headers = {"Content-Type": "application/json"}
    if OPENFIGI_KEY:
        headers["X-OPENFIGI-APIKEY"] = OPENFIGI_KEY

    payload = [{"idType": "TICKER", "idValue": t} for t in tickers]

    try:
        logger.info(f"Requesting metadata for {len(tickers)} tickers from OpenFIGI")
        resp = requests.post(OPENFIGI_URL, headers=headers, data=json.dumps(payload), timeout=20)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.error(f"OpenFIGI request failed: {e}")
        if os.path.exists(fallback_file):
            logger.info("Loading fallback mapping from disk")
            df_fallback = pd.read_csv(fallback_file)
            return {r.ticker: r.keywords.split(";") for r in df_fallback.itertuples()}
        return {}

    mapping = {}
    for tick, record in zip(tickers, data):
        info = record.get("data", [])
        if not info:
            keywords = [tick]
        else:
            item = info[0]
            name = item.get("name")
            keywords = [tick, name] if name else [tick]
        mapping[tick] = list({k for k in keywords if k})

    df_out = pd.DataFrame([
        {"ticker": t, "keywords": ";".join(mapping[t])}
        for t in mapping
    ])
    try:
        df_out.to_csv(fallback_file, index=False)
        logger.info(f"Saved fallback OpenFIGI mapping to {fallback_file}")
    except Exception as e:
        logger.warning(f"Failed to write fallback: {e}")

    return mapping

