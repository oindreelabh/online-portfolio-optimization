#!/bin/bash

set -e  # Exit on error

yFinance_historical="stock_prices_historical.csv"
yFinance_latest="stock_prices_latest.csv"
reddit="reddit_posts.csv"
finlayer="finance_news.csv"
recent_merged="recent_data_with_sentiment.csv"

raw_dir="data/raw"
processed_dir="data/processed"
model_dir="models"

lstm_model_name="lstm_model.keras"
ogdm_model_name="ogdm_model.pkl"
markowitz_model_name="markowitz_model.pkl"
capm_model_name="capm_model.pkl"

# echo "Step 1: Fetching data..."
# python -m src.data.fetch_yfinance \
# --historical $yFinance_historical \
# --latest $yFinance_latest \
# --raw_dir $raw_dir
# if [ $? -ne 0 ]; then
#    echo "Error fetching yfinance data. Exiting."
#    exit 1
# fi
# python -m src.data.fetch_reddit --filename $reddit --raw_dir $raw_dir
# if [ $? -ne 0 ]; then
#    echo "Error fetching Reddit data. Exiting."
#    exit 1
# fi
# python -m src.data.fetch_financelayer_news --filename $finlayer --raw_dir $raw_dir
# if [ $? -ne 0 ]; then
#    echo "Error fetching Financelayer news data. Exiting."
#    exit 1
# fi

# echo "Step 2: Adding Technical Indicators..."
# python -m src.features.build_ta_features \
# --input_file_y2 $yFinance_historical \
# --input_file_new $yFinance_latest \
# --raw_dir $raw_dir \
# --processed_dir $processed_dir
# if [ $? -ne 0 ]; then
#    echo "Error adding Technical indicators data. Exiting."
#    exit 1
# fi

# echo "Step 3: Preprocessing data..."
# python -m src.data.preprocess \
# --raw_dir $raw_dir \
# --processed_dir $processed_dir \
# --yfinance_y2 $yFinance_historical \
# --yfinance_new $yFinance_latest \
# --reddit $reddit \
# --financelayer $finlayer
# if [ $? -ne 0 ]; then
#    echo "Error preprocessing data. Exiting."
#    exit 1
# fi

# echo "Step 4: Creating sentiment features..."
# python -m src.features.merge_features \
# --yf_path $processed_dir/$yFinance_latest \
# --reddit_path $processed_dir/$reddit \
# --news_path $processed_dir/$finlayer \
# --output_dir $processed_dir \
# --output_file $recent_merged
# if [ $? -ne 0 ]; then
#    echo "Error adding sentiment features. Exiting."
#    exit 1
# fi

# echo "Step 5: Running LSTM model on 2 years of yfinance data..."
# python -m src.model.lstm \
# --data_path $processed_dir/$yFinance_historical \
# --model_save_path $model_dir/$lstm_model_name

# if [ $? -ne 0 ]; then
#    echo "Error running LSTM model. Exiting."
#    exit 1
# fi

# echo "Step 5: Running online learning model on recent data with sentiment features..."
# python -m src.model.online_learning \
# --data_path $processed_dir/$recent_merged \
# --model_path $model_dir/$ogdm_model_name \
# --target_col "close"
# if [ $? -ne 0 ]; then
#    echo "Error running OGDM model. Exiting."
#    exit 1
# fi

# echo "Step 6: Running Markowitz portfolio optimization..."
# python -m src.model.markowitz \
# --data_path $processed_dir/$yFinance_historical \
# --model_save_path $model_dir/$markowitz_model_name
# if [ $? -ne 0 ]; then
#     echo "Error running Markowitz model. Exiting."
#     exit 1
# fi

# echo "Step 7: Running CAPM portfolio optimization..."
# python -m src.model.capm_model \
# --data_path $processed_dir/$yFinance_historical \
# --model_save_path $model_dir/$capm_model_name \
# --market_return 0.10
# if [ $? -ne 0 ]; then
#     echo "Error running CAPM model. Exiting."
#     exit 1
# fi

# # Evaluate LSTM model
# echo "Step 6: Evaluating LSTM model..."
# python -m src.evaluation.evaluate_lstm \
# --data_path data/processed/stock_prices_latest.csv \
# --model_path models/lstm_model.keras \
# --scaler_path models/lstm_model_scaler.pkl \
# --sequence_length 5 \
# --output_dir evaluation_results/lstm

## Evaluate Online Learning model
#python src/evaluation/evaluate_online.py --data_path data/processed/test_data.csv --model_path models/ogdm_model.pkl --target_col close --output_dir evaluation_results/online
#
## Evaluate Hybrid model with portfolio backtesting
#python src/evaluation/evaluate_hybrid.py --data_path data/processed/test_data.csv --lstm_model_path models/lstm_model.keras --lstm_scaler_path models/lstm_model_scaler.pkl --online_model_path models/ogdm_model.pkl --sequence_length 5 --target_col close --output_dir evaluation_results/hybrid

# # Run alerts before launching dashboard (will send email if SMTP/env are set)
# echo "Step: Generating market alerts and sending email..."
# python -m src.alerts.send_alerts \
#   --data_path "$processed_dir/$recent_merged" \
#   --sentiment_col "reddit_sentiment_score" \
#   --period weekly \
#   --subject_prefix "Portfolio" \
#   --output_dir "evaluation_results/alerts" || echo "Warning: Alerts step failed (continuing)."

echo "Starting Streamlit dashboard..."
echo "Launching Portfolio Optimization Dashboard..."
echo "Dashboard will be available at: http://localhost:8501"
echo "Press Ctrl+C to stop the dashboard"
streamlit run src/dashboard/streamlit_dashboard.py --server.port 8501 --server.address localhost
if [ $? -ne 0 ]; then
    echo "Error running Streamlit dashboard."
    exit 1
fi

echo "Pipeline completed successfully."

echo "Cleaning up __pycache__ folders..."
find src -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
echo "Cleanup completed."