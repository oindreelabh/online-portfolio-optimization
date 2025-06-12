import sys
import os

# Making sure Airflow can find my project code
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd

from src.data.fetch_yfinance import fetch_and_store_latest_data
from src.data.fetch_reddit import fetch_and_store_recent_reddit_posts
from src.data.fetch_financelayer_news import fetch_and_store_news
from src.features.finbert_sentiment import add_finbert_sentiment
from src.features.merge_features import run_merge_pipeline

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime.now(),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'portfolio_optimization_pipeline',
    default_args=default_args,
    description='Daily pipeline for portfolio optimization data fetch and feature merge',
    schedule='@daily',
    catchup=False
) as dag:

    fetch_yfinance = PythonOperator(
        task_id='fetch_yfinance',
        python_callable=fetch_and_store_latest_data
    )

    fetch_reddit = PythonOperator(
        task_id='fetch_reddit',
        python_callable=fetch_and_store_recent_reddit_posts
    )

    fetch_finance_news = PythonOperator(
        task_id='fetch_finance_news',
        python_callable=fetch_and_store_news
    )

    def sentiment_analysis_task():
        reddit_df = pd.read_csv('data/raw/reddit_posts.csv', converters={'tickers': eval}, parse_dates=['date'])
        news_df = pd.read_csv('data/raw/finance_news.csv', converters={'tickers': eval}, parse_dates=['date'])
        reddit_df = add_finbert_sentiment(reddit_df, text_col='cleaned_text', prefix='reddit')
        news_df = add_finbert_sentiment(news_df, text_col='cleaned_text', prefix='news')
        reddit_df.to_csv('data/processed/reddit_with_sentiment.csv', index=False)
        news_df.to_csv('data/processed/news_with_sentiment.csv', index=False)

    sentiment_analysis = PythonOperator(
        task_id='sentiment_analysis',
        python_callable=sentiment_analysis_task
    )

    def merge_features_task():
        run_merge_pipeline(
            yf_path='data/raw/stock_prices_yesterday.csv',
            reddit_path='data/processed/reddit_with_sentiment.csv',
            news_path='data/processed/news_with_sentiment.csv',
            output_path='data/processed/merged_features.csv'
        )

    merge_features = PythonOperator(
        task_id='merge_features',
        python_callable=merge_features_task
    )

    # Task dependencies
    fetch_yfinance >> [fetch_reddit, fetch_finance_news] >> sentiment_analysis >> merge_features
