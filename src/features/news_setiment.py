from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from src.utils.logger import setup_logger
import os

logger = setup_logger(os.path.basename(__file__).replace(".py", ""))

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

def add_news_sentiment(df):
    if df.empty or 'title' not in df.columns:
        logger.warning("DataFrame is empty or missing 'title' column.")
        return df
    logger.info("Running FinBERT sentiment analysis on news titles...")
    sentiments = df['title'].apply(analyze_sentiment)
    df['sentiment'], df['sentiment_scores'] = zip(*sentiments)
    logger.info("Sentiment analysis complete.")
    return df

# Example usage in a pipeline:
# df = pd.read_sql("SELECT * FROM financial_news_raw", engine)
# df = add_news_sentiment(df)
# write_df_to_neon(df, "financial_news_with_sentiment")
