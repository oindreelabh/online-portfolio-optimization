# src/features/reddit_sentiment.py

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
from src.utils.logger import setup_logger

logger = setup_logger("reddit_sentiment")

# Load FinBERT model only once
tokenizer = AutoTokenizer.from_pretrained('yiyanghkust/finbert-pretrain')
model = AutoModelForSequenceClassification.from_pretrained('yiyanghkust/finbert-pretrain')

def analyze_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        sentiment = probs.argmax().item()
        sentiment_label = ['negative', 'neutral', 'positive'][sentiment]
        return sentiment_label, probs.squeeze().tolist()

def add_reddit_sentiment(df: pd.DataFrame, upvote_weight=True, min_upvotes=5):
    if df.empty or 'cleaned_text' not in df.columns:
        logger.warning("DataFrame is empty or missing 'cleaned_text' column.")
        return df

    # Noise reduction: filter low-upvote posts
    if upvote_weight and 'score' in df.columns:
        df = df[df['score'] >= min_upvotes]

    logger.info("Running FinBERT sentiment analysis on Reddit posts...")
    sentiments = df['cleaned_text'].apply(analyze_sentiment)
    df['sentiment_label'], df['sentiment_scores'] = zip(*sentiments)

    # Score mapping: negative=-1, neutral=0, positive=1
    sentiment_map = {'negative': -1, 'neutral': 0, 'positive': 1}
    df['sentiment_score'] = df['sentiment_label'].map(sentiment_map)

    # Weight sentiment by upvotes (score)
    if upvote_weight and 'score' in df.columns:
        df['weighted_sentiment'] = df['sentiment_score'] * df['score']
    else:
        df['weighted_sentiment'] = df['sentiment_score']

    logger.info("Reddit sentiment analysis complete.")
    return df
