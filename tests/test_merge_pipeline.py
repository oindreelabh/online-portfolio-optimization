import os
import tempfile
import pandas as pd
from src.features import merge_features

def create_temp_csv(df):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    df.to_csv(tmp.name, index=False)
    return tmp.name

def test_run_merge_pipeline(monkeypatch):
    # Minimal yfinance, reddit, and news data
    yf_df = pd.DataFrame({
        "ticker": ["AAPL"],
        "date": [pd.Timestamp("2024-01-01")],
        "open": [100], "close": [110]
    })
    reddit_df = pd.DataFrame({
        "tickers": [["AAPL"]],
        "date": [pd.Timestamp("2024-01-01")],
        "cleaned_text": ["Great earnings!"]
    })
    news_df = pd.DataFrame({
        "tickers": [["AAPL"]],
        "date": [pd.Timestamp("2024-01-01")],
        "cleaned_text": ["Apple stock rises."]
    })

    # Mock add_dual_sentiment to add a reddit_score/news_score column
    def mock_add_dual_sentiment(df, text_col):
        df = df.copy()
        if "reddit" in text_col or "cleaned_text" in df.columns:
            df["reddit_score"] = 0.5
            df["news_score"] = 0.7
        return df

    monkeypatch.setattr(merge_features, "add_dual_sentiment", mock_add_dual_sentiment)

    # Create temp CSVs
    yf_path = create_temp_csv(yf_df)
    reddit_path = create_temp_csv(reddit_df)
    news_path = create_temp_csv(news_df)
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".csv").name

    # Run pipeline
    merge_features.run_merge_pipeline(yf_path, reddit_path, news_path, output_path)

    # Check output
    out_df = pd.read_csv(output_path)
    assert "reddit_score" in out_df.columns
    assert "news_score" in out_df.columns
    assert out_df.loc[0, "reddit_score"] == 0.5
    assert out_df.loc[0, "news_score"] == 0.7

    # Cleanup
    os.remove(yf_path)
    os.remove(reddit_path)
    os.remove(news_path)
    os.remove(output_path)