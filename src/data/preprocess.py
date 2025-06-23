# src/data/preprocess.py

import pandas as pd
import re
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

def preprocess_yfinance(df: pd.DataFrame) -> pd.DataFrame:
    # Fill missing values, calculate returns, remove outliers, etc.
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['ticker', 'date'])
    df = df.groupby('ticker').apply(lambda x: x.interpolate()).reset_index(drop=True)
    df['returns'] = df.groupby('ticker')['close'].pct_change()
    q_low = df['returns'].quantile(0.01)
    q_high = df['returns'].quantile(0.99)
    df = df[(df['returns'] > q_low) & (df['returns'] < q_high)]
    return df.reset_index(drop=True)

def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = text.lower()
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words]
    return " ".join(tokens)

def preprocess_reddit(df: pd.DataFrame) -> pd.DataFrame:
    df['cleaned_text'] = df['full_text'].apply(clean_text)
    return df

def preprocess_financelayer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates(subset=['title', 'published_at'])
    df = df.dropna(subset=['title', 'description'])
    df['published_at'] = pd.to_datetime(df['published_at'], errors='coerce')
    df['cleaned_text'] = df['title'].fillna('') + ' ' + df['description'].fillna('')
    df['cleaned_text'] = df['cleaned_text'].apply(clean_text)
    return df
