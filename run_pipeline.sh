#!/bin/bash

set -e  # Exit on error

yFinance_historical="stock_prices_historical.csv"
yFinance_latest="stock_prices_latest.csv"
reddit="reddit_posts.csv"
finlayer="finance_new.csv"
recent_merged="recent_data_with_sentiment.csv"

raw_dir="data/raw"
processed_dir="data/processed"

echo "Step 1: Fetching data..."
python -m src.data.fetch_yfinance \
--historical $yFinance_historical \
--latest $yFinance_latest
if [ $? -ne 0 ]; then
    echo "Error fetching yfinance data. Exiting."
    exit 1
fi
python -m src.data.fetch_reddit --filename $reddit
if [ $? -ne 0 ]; then
    echo "Error fetching Reddit data. Exiting."
    exit 1
fi
python -m src.data.fetch_financelayer_news --filename $finlayer
if [ $? -ne 0 ]; then
    echo "Error fetching Financelayer news data. Exiting."
    exit 1
fi

echo "Step 2: Preprocessing data..."
python -m src.data.preprocess \
--raw_dir $raw_dir \
--processed_dir $processed_dir \
--yfinance_y2 $yFinance_historical \
--yfinance_new $yFinance_latest \
--reddit $reddit \
--financelayer $finlayer
if [ $? -ne 0 ]; then
    echo "Error preprocessing data. Exiting."
    exit 1
fi

echo "Step 3: Creating sentiment features..."
python -m src.features.merge_features \
--yf_path $processed_dir/$yFinance_latest \
--reddit_path $processed_dir/$reddit \
--news_path $processed_dir/$finlayer \
--output_path $processed_dir/$recent_merged
if [ $? -ne 0 ]; then
    echo "Error adding sentiment features. Exiting."
    exit 1
fi

echo "Step 4: Running LSTM model on 2 years of yfinance data..."
python -m src.model.lstm_hybrid \
--data_path $processed_dir/$yFinance_historical
if [ $? -ne 0 ]; then
    echo "Error running LSTM model. Exiting."
    exit 1
fi

echo "Step 5: Running online learning model on recent data with sentiment features..."

echo "Pipeline completed successfully."