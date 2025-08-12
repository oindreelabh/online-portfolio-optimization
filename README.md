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

#### Alerts (Email Notifications)
Configure SMTP and email addresses in .env:
```
SMTP_HOST=smtp.yourprovider.com
SMTP_PORT=587
SMTP_USER=your_smtp_user
SMTP_PASSWORD=your_smtp_password
EMAIL_FROM=alerts@yourdomain.com
EMAIL_TO=you@yourdomain.com,teammate@yourdomain.com
```

Send a weekly alert (dry run prints content instead of sending):
```bash
python -m src.alerts.send_alerts \
  --yf_path data/processed/stock_prices_latest.csv \
  --reddit_path data/processed/reddit_posts.csv \
  --news_path data/processed/finance_news.csv \
  --period weekly \
  --dry_run
```

Send a monthly alert:
```bash
python -m src.alerts.send_alerts \
  --yf_path data/processed/stock_prices_latest.csv \
  --period monthly
```

Options:
- --threshold_return: override return threshold (default 5% weekly, 10% monthly)
- --threshold_sentiment: override abs sentiment threshold (default 0.3)
- --output_dir: write a CSV report in addition to email
- --email_to / --email_from / --smtp_*: override .env at runtime

### 3. Dashboard
Launch the interactive dashboard:
```bash
streamlit run src/dashboard/streamlit_dashboard.py
```

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

## Disclaimer

This system is for educational and research purposes. Past performance does not guarantee future results. Always consult with financial professionals before making investment decisions.

