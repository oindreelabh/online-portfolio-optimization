import os
import pandas as pd
from finvader import SentimentIntensityAnalyzer
from src.utils.logger import setup_logger

logger = setup_logger(os.path.basename(__file__).replace(".py", ""))

analyzer = SentimentIntensityAnalyzer()

def add_finvader_sentiment(df: pd.DataFrame, text_col: str, prefix: str = "finvader") -> pd.DataFrame:
    """
    Adds FinVADER sentiment columns to the DataFrame.
    """
    logger.info(f"Applying FinVADER sentiment to column '{text_col}'")
    def get_scores(text):
        try:
            return analyzer.polarity_scores(str(text))
        except Exception as e:
            logger.error(f"Error analyzing text: {e}")
            return {"neg": None, "neu": None, "pos": None}

    scores = df[text_col].apply(get_scores)
    df[f"{prefix}_neg"] = scores.apply(lambda x: x["neg"])
    df[f"{prefix}_neu"] = scores.apply(lambda x: x["neu"])
    df[f"{prefix}_pos"] = scores.apply(lambda x: x["pos"])
    df[f"{prefix}_score"] = df[f"{prefix}_pos"] - df[f"{prefix}_neg"]
    logger.info("FinVADER sentiment annotation completed.")
    return df