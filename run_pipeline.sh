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

# # Run alerts before launching dashboard (will send email if SMTP/env are set)
# echo "Step: Generating market alerts and sending email..."
# python -m src.alerts.send_alerts \
#   --data_path "$processed_dir/$recent_merged" \
#   --sentiment_col "reddit_sentiment_score" \
#   --period weekly \
#   --subject_prefix "Portfolio" \
#   --output_dir "evaluation_results/alerts" || echo "Warning: Alerts step failed (continuing)."

# # Advanced Analytics Artifact Generation
# echo "Generating advanced analytics artifacts (if prerequisites exist)..."

# hist_prices="data/processed/stock_prices_historical.csv"
# recent_sent="data/processed/recent_data_with_sentiment.csv"
# lstm_model_path="models/lstm_model.keras"
# lstm_scaler_path="models/lstm_model_scaler.pkl"
# ogdm_model_path="models/ogdm_model.pkl"

# Performance comparison
# if [ -f "$hist_prices" ] && [ -f "$lstm_model_path" ] && [ -f "$lstm_scaler_path" ] && [ -f "$ogdm_model_path" ]; then
#   if [ ! -f "evaluation_results/perf/metrics_table.csv" ]; then
#     echo "Running performance_comparison..."
#     python -m src.evaluation.performance_comparison \
#       --data-csv "$hist_prices" \
#       --lstm-model "$lstm_model_path" \
#       --lstm-scaler "$lstm_scaler_path" \
#       --ogdm-model "$ogdm_model_path" \
#       --sequence-length 5 \
#       --output-dir evaluation_results/perf || echo "Warning: performance comparison failed."
#   fi
# else
#   echo "Skipping performance comparison (missing models or data)."
# fi

# # Portfolio metrics
# if [ -f "$hist_prices" ] && [ ! -f "evaluation_results/metrics/portfolio_timeseries.csv" ]; then
#   echo "Running portfolio_metrics..."
#   python -m src.evaluation.portfolio_metrics \
#     --prices-csv "$hist_prices" \
#     --output-dir evaluation_results/metrics \
#     --roll-window 20 \
#     --make-plots || echo "Warning: portfolio metrics failed."
# fi

# # Allocation evolution (synthetic equal-weight if no weights)
# if [ -f "$hist_prices" ] && [ ! -f "analysis_results/alloc_evolution/allocation_summary.csv" ]; then
#   echo "Preparing synthetic weights for allocation evolution..."
#   mkdir -p analysis_results/alloc_evolution
#   python - <<'PYEOF'
# import pandas as pd, os
# hist="data/processed/stock_prices_historical.csv"
# out="analysis_results/alloc_evolution/synthetic_weights.csv"
# if os.path.exists(hist) and not os.path.exists(out):
#     df=pd.read_csv(hist)
#     if {"date","ticker","close"}.issubset(df.columns):
#         df["date"]=pd.to_datetime(df["date"])
#         last=sorted(df["date"].unique())[-60:]
#         rows=[]
#         for d,g in df[df["date"].isin(last)].groupby("date"):
#             t=list(sorted(g["ticker"].unique()))
#             if not t: continue
#             w=1/len(t)
#             rows.extend({"date":d,"ticker":ti,"weight":w} for ti in t)
#         pd.DataFrame(rows).to_csv(out,index=False)
# PYEOF
#   python -m src.analysis.allocation_evolution \
#     --weights-csv analysis_results/alloc_evolution/synthetic_weights.csv \
#     --output-dir analysis_results/alloc_evolution || echo "Warning: allocation evolution failed."
# fi

# # Transaction cost impact
# if [ -f "evaluation_results/perf/predictions_long.csv" ] && [ ! -f "analysis_results/transaction_cost/tc_impact.csv" ]; then
#   echo "Running transaction_cost_impact..."
#   python -m src.analysis.transaction_cost_impact \
#     --predictions-csv evaluation_results/perf/predictions_long.csv \
#     --model-name HYBRID \
#     --output-dir analysis_results/transaction_cost || echo "Warning: transaction cost impact failed."
# fi

# # Sentiment influence
# if [ -f "$recent_sent" ] && [ -f "evaluation_results/perf/predictions_long.csv" ] && [ ! -f "analysis_results/sentiment/sentiment_correlations.csv" ]; then
#   echo "Running sentiment_influence..."
#   python -m src.analysis.sentiment_influence \
#     --data-csv "$recent_sent" \
#     --predictions-csv evaluation_results/perf/predictions_long.csv \
#     --model HYBRID \
#     --output-dir analysis_results/sentiment || echo "Warning: sentiment influence failed."
# fi

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