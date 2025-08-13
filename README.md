# Portfolio Optimization using Online Learning and Alternative Data

## Overview

This project implements a sophisticated portfolio optimization system that combines traditional financial models with modern machine learning techniques and alternative data sources. The system leverages LSTM neural networks, online gradient descent, CAPM, and Markowitz models to generate data-driven portfolio rebalancing recommendations.

## Key Features

- **Hybrid Model Architecture**: Combines LSTM and Online Gradient Descent Momentum (OGDM) models
- **Alternative Data Integration**: Incorporates Reddit sentiment and financial news sentiment
- **Technical Indicators**: RSI, MACD, Bollinger Bands, and volume-based features
- **Sentiment Analysis**: FinBERT, FinVADER-based sentiment scoring for social media and news
- **Real-time Predictions**: Online learning for adapting to market changes
- **Portfolio Optimization**: CAPM and Markowitz model implementations
- **Interactive Dashboard**: Streamlit-based visualization and monitoring

## Architecture

```
├── src/
│   ├── data/                 # Data fetching and preprocessing
│   ├── features/             # Feature engineering and sentiment analysis
│   ├── model/                # Machine learning models
│   ├── evaluation/           # Model evaluation and backtesting
│   ├── dashboard/            # Streamlit dashboard
│   └── utils/                # Utilities and constants
├── data/                     # Data storage (raw and processed)
└── models/                   # Trained model artifacts
```

## Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup
```bash
# Clone the repository
git clone <repository-url>
cd online-portfolio-optimization

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys:
# - REDDIT_CLIENT_ID
# - REDDIT_CLIENT_SECRET
# - REDDIT_USER_AGENT
# - FINANCELAYER_API_KEY
```

## Data Sources

### 1. Stock Market Data (Yahoo Finance)
- Historical and real-time stock prices
- Volume and adjusted close prices
- Technical indicators calculation

### 2. Reddit Sentiment Data
- Posts from r/stocks, r/wallstreetbets, r/investing, r/stockmarket, r/pennystocks
- Comment aggregation and sentiment analysis
- Ticker mention extraction

### 3. Financial News Data
- Financial news articles from FinanceLayer API
- Company-specific news filtering
- Sentiment analysis using FinBERT

## Models

### 1. LSTM Model (`src/model/lstm_hybrid.py`)
- **Purpose**: Sequence-based price prediction
- **Architecture**: 50-unit LSTM layer with dropout
- **Features**: Technical indicators, volume ratios
- **Training**: Adam optimizer with early stopping

### 2. Online Gradient Descent Momentum (`src/model/online_learning.py`)
- **Purpose**: Real-time model adaptation
- **Algorithm**: Gradient descent with momentum
- **Learning Rate**: 0.01 (configurable)
- **Updates**: Incremental learning from new data

### 3. CAPM Model (`src/model/capm_model.py`)
- **Purpose**: Risk-return optimization
- **Beta Calculation**: Linear regression against market index
- **Expected Returns**: Based on CAPM formula
- **Risk Adjustment**: Beta-weighted portfolio allocation

### 4. Hybrid Prediction System
- **Combination**: LSTM + OGDM predictions
- **Weighting**: Simple average (configurable)
- **Portfolio Allocation**: Rank-based with diversification constraints

### 5. Advanced Evaluation & Analytics Components (New)
These scripts provide deeper comparative analysis, allocation dynamics, cost impact, and sentiment influence:
- performance_comparison.py (src/evaluation): Walk-forward backtest comparing LSTM, OGDM, Hybrid, Equal-Weight, Return-Persistence baselines. Outputs:
  - metrics_table.csv
  - predictions_long.csv
  - portfolio_equity.csv
  - equity_curves.html, drawdowns.html, sharpe.html
- portfolio_metrics.py (src/evaluation): Aggregates equity, drawdowns, rolling volatility (portfolio_timeseries.csv + HTML plots).
- allocation_evolution.py (src/analysis): Tracks allocation shifts, Herfindahl index, top weight (allocation_summary.csv + evolution & concentration HTML charts).
- transaction_cost_impact.py (src/analysis): Compares naive vs constrained rebalancing (tc_impact.csv + turnover plot).
- sentiment_influence.py (src/analysis): Correlates sentiment signals with prediction and allocation changes (sentiment_correlations.csv + scatter/regression HTMLs).

## Usage

### 1. Data Pipeline
Run the complete data pipeline:
```bash
# Make script executable
chmod +x run_pipeline.sh

# Run pipeline
./run_pipeline.sh
```

### 2. Individual Components

#### Fetch Data
```bash
# Historical stock data (5 years)
python -m src.data.fetch_yfinance --historical stock_prices_historical.csv --raw_dir data/raw

# Latest stock data (14 days)
python -m src.data.fetch_yfinance --latest stock_prices_latest.csv --raw_dir data/raw

# Reddit posts
python -m src.data.fetch_reddit --filename reddit_posts.csv --raw_dir data/raw

# Financial news
python -m src.data.fetch_financelayer_news --filename finance_news.csv --raw_dir data/raw
```

#### Feature Engineering
```bash
# Add technical indicators
python -m src.features.build_ta_features \
  --input_file_y2 stock_prices_historical.csv \
  --input_file_new stock_prices_latest.csv \
  --raw_dir data/raw \
  --processed_dir data/processed

# Merge sentiment features
python -m src.features.merge_features \
  --yf_path data/processed/stock_prices_latest.csv \
  --reddit_path data/processed/reddit_posts.csv \
  --news_path data/processed/finance_news.csv \
  --output_dir data/processed \
  --output_file recent_data_with_sentiment.csv
```

#### Model Training
```bash
# Train LSTM model
python -m src.model.lstm_hybrid \
  --data_path data/processed/stock_prices_historical.csv \
  --model_save_path models/lstm_model.keras

# Train OGDM model
python -m src.model.online_learning \
  --data_path data/processed/recent_data_with_sentiment.csv \
  --model_path models/ogdm_model.pkl \
  --target_col close
```

#### Portfolio Predictions
```bash
# Generate portfolio recommendations
python -m src.model.predict \
  --data-path data/processed/recent_data_with_sentiment.csv \
  --lstm-model-path models/lstm_model.keras \
  --lstm-scaler-path models/lstm_model_scaler.pkl \
  --online-model-path models/ogdm_model.pkl \
  --tickers AAPL,TSLA,GOOGL,MSFT \
  --current-allocations AAPL:0.25,TSLA:0.25,GOOGL:0.25,MSFT:0.25 \
  --output json
```

#### Model Evaluation
```bash
# Evaluate LSTM model
python -m src.evaluation.evaluate_lstm \
  --data_path data/processed/stock_prices_latest.csv \
  --model_path models/lstm_model.keras \
  --scaler_path models/lstm_model_scaler.pkl \
  --output_dir evaluation_results/lstm
```

#### Advanced Evaluation & Analytics (New)
Run after models are trained and data processed.

```bash
# 1. Model performance comparison & portfolio backtest
python -m src.evaluation.performance_comparison \
  --data-csv data/processed/stock_prices_historical.csv \
  --lstm-model models/lstm_model.keras \
  --lstm-scaler models/lstm_model_scaler.pkl \
  --ogdm-model models/ogdm_model.pkl \
  --sequence-length 5 \
  --output-dir evaluation_results/perf \
  --make-plots

# 2. Portfolio metrics (equity, drawdowns, rolling vol)
python -m src.evaluation.portfolio_metrics \
  --prices-csv data/processed/stock_prices_historical.csv \
  --output-dir evaluation_results/metrics \
  --roll-window 20 \
  --make-plots

# 3. Allocation evolution (requires weights CSV; can synthesize equal-weight)
python -m src.analysis.allocation_evolution \
  --weights-csv analysis_results/alloc_evolution/synthetic_weights.csv \
  --output-dir analysis_results/alloc_evolution

# 4. Transaction cost impact (uses predictions_long.csv from performance comparison)
python -m src.analysis.transaction_cost_impact \
  --predictions-csv evaluation_results/perf/predictions_long.csv \
  --model-name HYBRID \
  --output-dir analysis_results/transaction_cost \
  --cost-rate 0.001

# 5. Sentiment influence (needs sentiment-enriched data + predictions)
python -m src.analysis.sentiment_influence \
  --data-csv data/processed/recent_data_with_sentiment.csv \
  --predictions-csv evaluation_results/perf/predictions_long.csv \
  --model HYBRID \
  --output-dir analysis_results/sentiment \
  --lag-days 1
```

Outputs are consumed by the Advanced Analytics dashboard tab.

### 3. Dashboard
Launch the interactive dashboard:
```bash
streamlit run src/dashboard/streamlit_dashboard.py
```

#### Advanced Analytics Tab (New)
The Streamlit dashboard now includes an "Advanced Analytics" tab:
- Auto-detects artifacts in:
  - evaluation_results/perf
  - evaluation_results/metrics
  - analysis_results/alloc_evolution
  - analysis_results/transaction_cost
  - analysis_results/sentiment
- Displays: performance metrics table, equity & drawdown curves, allocation concentration, turnover & cost savings, sentiment correlations.
- Includes a "Auto Generate Missing Artifacts" button to run the evaluation scripts (if models & data exist).

### 4. Pipeline Enhancements (New)
run_pipeline.sh now (optionally) generates evaluation & analysis artifacts BEFORE launching the dashboard:
- performance_comparison
- portfolio_metrics
- allocation_evolution (synthetic equal-weight if no weights supplied)
- transaction_cost_impact
- sentiment_influence

Adjust or uncomment earlier data/model steps as needed.

## Configuration

### Supported Tickers
```python
TICKERS = [
    "AAPL", "TSLA", "AMZN", "MSFT", "NVDA",
    "GME", "AMC", "META", "JPM", "SPY",
    "UNH", "GOOGL"
]
```

### Subreddits
```python
SUBREDDITS = [
    "stocks", "wallstreetbets", "investing", 
    "stockmarket", "pennystocks"
]
```

### Model Parameters
- **LSTM Sequence Length**: 5-10 days
- **OGDM Learning Rate**: 0.01
- **Portfolio Diversification**: Max 25% per holding
- **Rebalancing Constraint**: Max 20% change per period

## Output Format

### Prediction Results
```json
{
  "status": "success",
  "predictions": {
    "AAPL": 0.0234,
    "TSLA": -0.0156,
    "GOOGL": 0.0189
  },
  "suggested_allocations": {
    "AAPL": 0.35,
    "TSLA": 0.20,
    "GOOGL": 0.45
  },
  "message": "Successfully generated predictions for 3 tickers"
}
```

### Evaluation Metrics
- **MSE/RMSE**: Mean squared/root mean squared error
- **MAE**: Mean absolute error
- **R²**: Coefficient of determination
- **MAPE**: Mean absolute percentage error

### Additional Analytics Outputs (New)
Example: metrics_table.csv
```csv
model,mse,mae,directional_accuracy,avg_return,volatility,sharpe,cumulative_return
HYBRID,0.00042,0.0153,0.56,0.0012,0.0125,0.096,0.182
```

transaction_cost impact (tc_impact.csv)
```csv
date,naive_turnover,constrained_turnover,naive_cost,constrained_cost,turnover_reduction,cost_saving
2024-06-10,0.34,0.18,0.00034,0.00018,0.16,0.00016
```

sentiment_correlations.csv
```csv
sentiment_feature,target,pearson_corr
reddit_sentiment_lag,predicted_return,0.12
news_sentiment_lag,alloc_delta,0.08
```

## Technical Details

### Sentiment Analysis
- **Model**: ProsusAI/finbert
- **Processing**: Title + content + top comments
- **Score**: Positive probability - Negative probability
- **Aggregation**: Daily average by ticker

### Technical Indicators
- **RSI**: 14-period Relative Strength Index
- **MACD**: 12-26-9 Moving Average Convergence Divergence
- **Bollinger Bands**: 20-period with 2 standard deviations
- **Volume Ratios**: Current vs. moving average

### Risk Management
- **Diversification**: Maximum 25% allocation per asset
- **Gradual Rebalancing**: Maximum 20% change per period
- **Volatility Filtering**: 1st-99th percentile return filtering

## Logging

The system uses structured logging for monitoring:
- Data fetching and processing steps
- Model training progress
- Prediction generation
- Error handling and warnings

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with appropriate tests
4. Submit a pull request

## Changelog (New)
- Added comparative backtesting & analytics scripts.
- Added advanced_analytics_tab.py to dashboard.
- Added auto-generation of analytics artifacts (pipeline + dashboard button).
- Extended README with evaluation workflow & artifact description.

## Disclaimer

This system is for educational and research purposes. Past performance does not guarantee future results. Always consult with financial professionals before making investment decisions.

