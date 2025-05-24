from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime
import sys
import os

# Make sure Airflow can find your project code
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from src.data.fetch_yfinance import fetch_all_tickers
from src.data.fetch_reddit import fetch_reddit_data
from src.data.preprocess import preprocess_all
from features.build_features import build_feature_set
from models.online_model import run_online_learning
from alerts.send_alerts import send_alerts

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
}

with DAG("portfolio_optimization_pipeline",
         default_args=default_args,
         schedule_interval="0 9 * * *",  # Runs daily at 9:00 AM
         catchup=False) as dag:

    fetch_yfinance_task = PythonOperator(
        task_id="fetch_yfinance_data",
        python_callable=fetch_all_tickers
    )

    fetch_reddit_task = PythonOperator(
        task_id="fetch_reddit_data",
        python_callable=fetch_reddit_data
    )

    preprocess_task = PythonOperator(
        task_id="preprocess_data",
        python_callable=preprocess_all
    )

    feature_task = PythonOperator(
        task_id="build_features",
        python_callable=build_feature_set
    )

    model_task = PythonOperator(
        task_id="run_online_model",
        python_callable=run_online_learning
    )

    alert_task = PythonOperator(
        task_id="send_email_alerts",
        python_callable=send_alerts
    )

    # Define execution order
    [fetch_yfinance_task, fetch_reddit_task] >> preprocess_task >> feature_task >> model_task >> alert_task
