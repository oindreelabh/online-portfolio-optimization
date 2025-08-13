import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys
from datetime import datetime

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import hybrid model functions and Markowitz optimizer
from src.model.lstm_ogdm_hybrid import hybrid_predict_and_rebalance, parse_allocations
from src.model.markowitz import MarkowitzOptimizer
from src.model.capm_model import CAPMOptimizer

# Page configuration
st.set_page_config(
    page_title="Portfolio Optimization Dashboard",  # Title shown in browser tab
    page_icon=":chart_with_upwards_trend:",        # Emoji or path to favicon
    layout="wide",                                 # "centered" or "wide"
    initial_sidebar_state="auto"               # "auto", "expanded", or "collapsed"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .positive {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .negative {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header"> Portfolio Optimization Dashboard</h1>', unsafe_allow_html=True)

# Sidebar for model selection and parameters
st.sidebar.header("Model Configuration")

# Model selection
model_type = st.sidebar.selectbox(
    "Select Model Type",
    ["LSTM-OGDM Hybrid", "Markowitz", "CAPM"]
)
model_type = model_type.strip().lower()
model_path_dict = {
    "markowitz": "markowitz_model.pkl",
    "capm": "capm_model.pkl"
}

def render_tab_portfolio(model_type: str, project_root: str) -> None:
    """
    Render the Portfolio Optimization & Prediction tab.
    Encapsulates sidebar inputs, hybrid/markowitz/capm execution, and results rendering.
    """
    st.header("Portfolio Optimization & Prediction")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Portfolio Parameters")
        
        # Portfolio configuration for hybrid model
        if model_type == "lstm-ogdm hybrid":
            # Multiple stock selection
            default_tickers = ["AAPL", "AMZN"]
            tickers_input = st.text_input(
                "Stock Tickers (comma-separated)", 
                value=",".join(default_tickers),
                help="Enter stock ticker symbols separated by commas"
            )
            tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
            
            # Current portfolio allocations
            st.subheader("Current Portfolio Allocations")
            current_allocations = {}
            equal_weight = 1.0 / len(tickers) if tickers else 0.25
            
            for ticker in tickers:
                weight = st.slider(
                    f"{ticker} Weight", 
                    0.0, 1.0, equal_weight, 0.01,
                    key=f"weight_{ticker}"
                )
                current_allocations[ticker] = weight
            
            # Normalize weights to sum to 1
            total_weight = sum(current_allocations.values())
            if total_weight > 0:
                current_allocations = {k: v/total_weight for k, v in current_allocations.items()}
            
            # Display normalized weights
            st.write("Normalized Weights:")
            for ticker, weight in current_allocations.items():
                st.write(f"• {ticker}: {weight:.2%}")
            
            # Sequence length for LSTM
            sequence_length = st.slider("LSTM Sequence Length", 5, 30, 5)
            
        else:
            # Single stock input for other models
            symbol = st.text_input("Stock Symbol", value="AAPL", help="Enter stock ticker symbol")
            
            # Time horizon
            prediction_days = st.slider("Prediction Horizon (days)", 1, 30, 5)
            
            # Technical indicators
            st.subheader("Technical Indicators")
            rsi = st.slider("RSI", 0.0, 100.0, 50.0)
            macd = st.slider("MACD", -5.0, 5.0, 0.0)
            bb_position = st.slider("Bollinger Band Position", 0.0, 1.0, 0.5)
            volume_ratio = st.slider("Volume Ratio", 0.0, 5.0, 1.0)
            
            # Market sentiment
            st.subheader("Market Sentiment")
            vix = st.slider("VIX (Volatility Index)", 10.0, 80.0, 20.0)
            market_sentiment = st.selectbox("Market Sentiment", ["Positive", "Neutral", "Negative"])
        
        # Predict button
        predict_button = st.button("Generate Prediction", type="primary")
    
    with col2:
        st.subheader("Prediction Results")
        
        if predict_button:
            with st.spinner("Generating predictions..."):
                
                if model_type == "lstm-ogdm hybrid":
                    # Load actual data from CSV file
                    data_path = os.path.join(project_root, "data", "processed", "recent_data_with_sentiment.csv")
                    
                    # Check if data file exists
                    if not os.path.exists(data_path):
                        st.error(f"Data file not found: {data_path}")
                        st.info("Please ensure the recent_data_with_sentiment.csv file exists in the data/processed directory.")
                        st.stop()
                    
                    try:
                        # Load the actual data
                        sample_data = pd.read_csv(data_path)
                        
                        # Filter data for selected tickers
                        sample_data = sample_data[sample_data['ticker'].isin(tickers)]
                        
                        if sample_data.empty:
                            available_tickers = pd.read_csv(data_path)['ticker'].unique().tolist()
                            st.error(f"No data found for selected tickers: {tickers}")
                            st.info(f"Available tickers in data: {', '.join(available_tickers)}")
                            st.stop()
                        
                        st.info(f"Loaded {len(sample_data)} records for {len(sample_data['ticker'].unique())} tickers from actual data")
                        
                    except Exception as e:
                        st.error(f"Error loading data file: {str(e)}")
                        st.stop()
                    
                    try:
                        # Model paths - LSTM model, scaler, and online model are required
                        base_model_dir = os.path.join(project_root, "models")
                        lstm_model_path = os.path.join(base_model_dir, "lstm_model.keras")
                        lstm_scaler_path = os.path.join(base_model_dir, "lstm_model_scaler.pkl")
                        online_model_path = os.path.join(base_model_dir, "ogdm_model.pkl")
                        
                        # Check if required model files exist
                        missing_files = []
                        for name, path in [("LSTM model", lstm_model_path), ("LSTM scaler", lstm_scaler_path), ("OGDM model", online_model_path)]:
                            if not os.path.exists(path):
                                missing_files.append(name)
                        
                        if missing_files:
                            st.error(f"Missing model files: {', '.join(missing_files)}")
                            st.info("Please ensure LSTM model, scaler, and OGDM model files are trained and saved in the models directory.")
                            st.stop()
                        
                        # Generate predictions using hybrid model
                        result = hybrid_predict_and_rebalance(
                            sample_data,
                            lstm_model_path,
                            lstm_scaler_path, 
                            online_model_path,
                            tickers,
                            current_allocations,
                            sequence_length
                        )
                        
                        if result['status'] == 'success':
                            st.success(result['message'])
                            
                            # Check if we have valid predictions and allocations
                            if not result['predictions'] or not result['suggested_allocations']:
                                st.warning("No valid predictions or allocations generated. This may be due to insufficient data")
                                st.info("Please try with different tickers or check if you have enough historical data.")
                                # Do not exit the Streamlit app; stop this run block
                                st.stop()
                            
                            # Sanitize predictions: drop non-finite and keep in plausible range
                            clean_predictions = {
                                t: float(np.clip(p, -0.3, 0.3))
                                for t, p in result['predictions'].items()
                                if p is not None and np.isfinite(p)
                            }
                            if len(clean_predictions) < len(result['predictions']):
                                st.info("Some tickers returned invalid predictions and were skipped.")
                            
                            # Display predictions
                            st.subheader("Predicted Returns")
                            pred_df = pd.DataFrame([
                                {'Ticker': ticker, 'Predicted Return': f"{pred:.4f}"}
                                for ticker, pred in clean_predictions.items()
                            ])
                            st.dataframe(pred_df, use_container_width=True)
                            
                            # Display suggested allocations
                            st.subheader("Suggested Portfolio Allocation")
                            
                            col_a, col_b = st.columns(2)
                            
                            with col_a:
                                st.write("**Current vs Suggested:**")
                                allocation_df = pd.DataFrame([
                                    {
                                        'Ticker': ticker, 
                                        'Current': f"{current_allocations.get(ticker, 0):.2%}",
                                        'Suggested': f"{result['suggested_allocations'].get(ticker, 0):.2%}"
                                    }
                                    for ticker in tickers if ticker in result['suggested_allocations']
                                ])
                                st.dataframe(allocation_df, use_container_width=True)
                            
                            with col_b:
                                # Pie chart of suggested allocation
                                if result['suggested_allocations']:
                                    fig_pie = px.pie(
                                        values=list(result['suggested_allocations'].values()),
                                        names=list(result['suggested_allocations'].keys()),
                                        title="Suggested Portfolio Allocation"
                                    )
                                    # Add unique key to avoid duplicate element IDs
                                    st.plotly_chart(fig_pie, use_container_width=True, key="hybrid_alloc_pie")
                                else:
                                    st.info("No allocation suggestions available")
                            
                            # Portfolio metrics
                            st.subheader("Portfolio Metrics")
                            col_a, col_b, col_c = st.columns(3)
                            
                            # Calculate weighted return safely
                            if clean_predictions and result['suggested_allocations']:
                                weighted_return = sum(
                                    clean_predictions.get(t, 0.0) * result['suggested_allocations'].get(t, 0.0)
                                    for t in result['suggested_allocations'].keys()
                                )
                                
                                # Safe calculation of diversification score
                                allocation_values = list(result['suggested_allocations'].values())
                                max_allocation = max(allocation_values) if allocation_values else 0
                                diversification_score = 1 - max_allocation
                                
                                col_a.metric("Expected Portfolio Return", f"{weighted_return:.4f}")
                                col_b.metric("Diversification Score", f"{diversification_score:.2%}")
                                col_c.metric("Number of Holdings", len(result['suggested_allocations']))
                            else:
                                col_a.metric("Expected Portfolio Return", "N/A")
                                col_b.metric("Diversification Score", "N/A")
                                col_c.metric("Number of Holdings", "0")
                        else:
                            st.error(f"Prediction failed: {result['message']}")
                    except Exception as e:
                        st.error(f"Error loading or running model: {str(e)}")
                        st.stop()
    
    # Debug information expander
    with st.expander("Debug Information", expanded=False):
        st.write("This section provides additional information for debugging purposes.")
        st.write(f"Selected Model Type: {model_type}")
        st.write(f"Tickers: {tickers}")
        st.write(f"Current Allocations: {current_allocations}")
        if model_type == "lstm-ogdm hybrid":
            st.write(f"Sequence Length: {sequence_length}")
        st.write("---")
        st.write("If you encounter any issues, please check the above parameters and ensure that the model and data files are correctly set up.")

def render_tab_historical() -> None:
    """
    Render the Historical Analysis tab with price/volume charts and stats.
    Replaced synthetic sample data with real historical data loaded from
    data/processed/historical.csv (expected columns: date, ticker, close, volume).
    """
    st.header("Historical Analysis")

    # Path to historical data
    historical_path = os.path.join(project_root, "data", "processed", "stock_prices_historical.csv")

    if not os.path.exists(historical_path):
        st.error(f"Historical data file not found: {historical_path}")
        st.info("Place a CSV named stock_prices_historical.csv under data/processed with columns: date, ticker, close, volume")
        return

    try:
        df_hist = pd.read_csv(historical_path)
    except Exception as e:
        st.error(f"Failed to read stock_prices_historical.csv: {e}")
        return

    # Basic validation
    required_cols = {"date", "ticker", "close"}
    missing = required_cols.difference(df_hist.columns)
    if missing:
        st.error(f"Missing required columns in stock_prices_historical.csv: {', '.join(missing)}")
        return

    # Optional volume column
    has_volume = "volume" in df_hist.columns

    # Parse dates
    try:
        df_hist["date"] = pd.to_datetime(df_hist["date"])
    except Exception as e:
        st.error(f"Failed parsing date column: {e}")
        return

    # Sidebar-like controls inside tab
    tickers_available = sorted(df_hist["ticker"].unique())
    default_selection = tickers_available[:5]
    selected_tickers = st.multiselect(
        "Select Tickers",
        options=tickers_available,
        default=default_selection,
        help="Choose one or more tickers to visualize"
    )
    if not selected_tickers:
        st.warning("Select at least one ticker.")
        return

    min_date, max_date = df_hist["date"].min(), df_hist["date"].max()
    date_range = st.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    else:
        start_date, end_date = min_date, max_date

    # Filter
    mask = (
        df_hist["ticker"].isin(selected_tickers) &
        (df_hist["date"] >= start_date) &
        (df_hist["date"] <= end_date)
    )
    df_view = df_hist.loc[mask].copy().sort_values(["date", "ticker"])

    if df_view.empty:
        st.warning("No data after applying filters.")
        return

    # Price chart
    col1, col2 = st.columns(2)
    with col1:
        fig_price = px.line(
            df_view,
            x="date",
            y="close",
            color="ticker",
            title="Historical Price Movement",
            labels={"close": "Close", "date": "Date"}
        )
        st.plotly_chart(fig_price, use_container_width=True, key="hist_price_real")

    # Volume chart (if available)
    with col2:
        if has_volume:
            fig_volume = px.bar(
                df_view,
                x="date",
                y="volume",
                color="ticker",
                title="Trading Volume",
                labels={"volume": "Volume", "date": "Date"}
            )
            st.plotly_chart(fig_volume, use_container_width=True, key="hist_volume_real")
        else:
            st.info("Volume column not found; skipping volume chart.")

    # Summary metrics over filtered set
    st.subheader("Statistical Summary")
    # Aggregate per ticker or overall
    summary = (
        df_view.groupby("ticker")["close"]
        .agg(["mean", "std", "max", "min"])
        .rename(columns={
            "mean": "Mean Price",
            "std": "Volatility",
            "max": "Max Price",
            "min": "Min Price"
        })
    )

    st.dataframe(summary.style.format({
        "Mean Price": "{:.2f}",
        "Volatility": "{:.2f}",
        "Max Price": "{:.2f}",
        "Min Price": "{:.2f}"
    }), use_container_width=True)

    # Overall metrics
    overall_mean = df_view["close"].mean()
    overall_vol = df_view["close"].std()
    overall_max = df_view["close"].max()
    overall_min = df_view["close"].min()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Overall Mean", f"${overall_mean:.2f}")
    c2.metric("Overall Volatility", f"{overall_vol:.2f}")
    c3.metric("Overall Max", f"${overall_max:.2f}")
    c4.metric("Overall Min", f"${overall_min:.2f}")

def render_tab_performance() -> None:
    """
    Render the Model Performance Metrics tab with metrics, confusion matrix, and feature importance.
    """
    st.header("Model Performance Metrics")
    
    # Performance metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Classification Metrics")
        metrics_data = {
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            'Value': [0.85, 0.82, 0.88, 0.85]
        }
        metrics_df = pd.DataFrame(metrics_data)
        
        fig_metrics = px.bar(metrics_df, x='Metric', y='Value', title='Model Performance Metrics')
        fig_metrics.update_layout(yaxis=dict(range=[0, 1]))
        # Add unique key
        st.plotly_chart(fig_metrics, use_container_width=True, key="perf_metrics")
    
    with col2:
        st.subheader("Confusion Matrix")
        confusion_matrix = np.array([[85, 15], [12, 88]])
        fig_cm = px.imshow(confusion_matrix, 
                          text_auto=True, 
                          aspect="auto",
                          title="Confusion Matrix",
                          labels=dict(x="Predicted", y="Actual"))
        # Add unique key
        st.plotly_chart(fig_cm, use_container_width=True, key="perf_cm")
    
    # Feature importance
    st.subheader("Feature Importance")
    features = ['RSI', 'MACD', 'Volume', 'Moving Average', 'Bollinger Bands', 'VIX']
    importance = np.random.rand(len(features))
    
    feature_df = pd.DataFrame({
        'Feature': features,
        'Importance': importance
    }).sort_values('Importance', ascending=True)
    
    fig_importance = px.bar(feature_df, x='Importance', y='Feature', 
                           orientation='h', title='Feature Importance')
    # Add unique key
    st.plotly_chart(fig_importance, use_container_width=True, key="feat_importance")

# Main dashboard layout
tab1, tab2, tab3 = st.tabs(["Portfolio Optimization", "Historical Analysis", "Model Performance"])

with tab1:
    # Call the refactored function instead of inline code
    render_tab_portfolio(model_type, project_root)

with tab2:
    render_tab_historical()

with tab3:
    render_tab_performance()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Portfolio Optimization Dashboard | Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)

