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
from src.dashboard.advanced_analytics_tab import render_tab_advanced_analytics

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
    Supports hybrid (LSTM-OGDM), Markowitz (.pkl), and CAPM (.pkl) models.
    """
    st.header("Portfolio Optimization & Prediction")
    
    # Initialize shared variables to avoid reference errors in debug
    tickers = []
    current_allocations = {}
    sequence_length = None
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Portfolio Parameters")
        
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
        
        elif model_type == "markowitz":
            st.markdown("Markowitz Mean-Variance Optimization")
            tickers_input = st.text_input(
                "Tickers (comma-separated)", 
                value="AAPL,AMZN,MSFT,GOOGL",
                help="Works with either (a) wide price CSV (date as index column + ticker columns) or (b) long format (date,ticker,close, ...). Auto-detects & pivots."
            )
            tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
            data_csv_path = st.text_input(
                "Price / Feature CSV (long or wide)",
                value=os.path.join(project_root, "data", "processed", "stock_prices_historical.csv"),
                help="If long format: must contain columns date,ticker,close. Will be pivoted automatically to wide (close prices)."
            )
            target_return_input = st.number_input(
                "Target Annual Return (0 = maximize Sharpe)",
                min_value=0.0,
                value=0.0,
                step=0.01
            )
            target_return = target_return_input if target_return_input > 0 else None
        
        elif model_type == "capm":
            st.markdown("CAPM-based Portfolio Optimization")
            tickers_input = st.text_input(
                "Tickers (comma-separated)",
                value="AAPL,AMZN,MSFT,GOOGL",
                help="Tickers must exist in returns CSV (long format)"
            )
            tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
            data_csv_path = st.text_input(
                "Returns Data CSV (long format)",
                value=os.path.join(project_root, "data", "processed", "capm_returns.csv"),
                help="Columns required: date,ticker,returns"
            )
            market_ticker = st.text_input("Market Ticker", value="^GSPC")
            market_return = st.number_input(
                "Expected Market Return",
                min_value=-1.0,
                max_value=2.0,
                value=0.10,
                step=0.01
            )
        # Predict button (all models) - removed type for compatibility
        predict_button = st.button("Generate Prediction")
    
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
                
                elif model_type == "markowitz":
                    try:
                        def ensure_wide_price_csv(original_path: str, proj_root: str) -> str:
                            """
                            Ensure a wide-format CSV (date index, ticker columns of close prices) exists.
                            If the provided CSV is long format (contains 'ticker' column), pivot it.
                            
                            Returns path to a wide-format CSV file.
                            """
                            if not os.path.exists(original_path):
                                raise FileNotFoundError(f"Price CSV not found: {original_path}")
                            
                            df_raw = pd.read_csv(original_path)
                            lower_cols = {c.lower() for c in df_raw.columns}
                            
                            # Detect long format by presence of ticker + close columns
                            if {"ticker", "close"}.issubset(lower_cols):
                                # Normalize column names for robustness
                                rename_map = {c: c.lower() for c in df_raw.columns}
                                df_raw = df_raw.rename(columns=rename_map)
                                if "date" not in df_raw.columns:
                                    raise ValueError("Long format file must contain 'date' column.")
                                
                                st.info("Detected long format price file. Pivoting on close prices...")
                                # Pivot to wide using close prices
                                wide_df = (
                                    df_raw.pivot(index="date", columns="ticker", values="close")
                                    .sort_index()
                                )
                                
                                # Drop columns that are completely NaN
                                wide_df = wide_df.dropna(axis=1, how="all")
                                
                                if wide_df.empty:
                                    raise ValueError("Pivot result is empty. Check input data.")
                                
                                # Save to a temp wide CSV
                                temp_dir = os.path.join(proj_root, "data", "processed")
                                os.makedirs(temp_dir, exist_ok=True)
                                wide_path = os.path.join(temp_dir, "_wide_markowitz_prices.csv")
                                wide_df.to_csv(wide_path)
                                st.success(f"Pivot complete. Wide file created: {wide_path}")
                                st.caption(f"Wide tickers ({len(wide_df.columns)}): {', '.join(list(wide_df.columns)[:30])}" + (" ..." if len(wide_df.columns) > 30 else ""))
                                return wide_path
                            
                            # Assume already wide if ticker column not present
                            st.info("Assuming provided CSV is already in wide format (date + ticker columns).")
                            return original_path
                        
                        # Prepare wide-format path
                        wide_csv_path = ensure_wide_price_csv(data_csv_path, project_root)
                        
                        model_file = os.path.join(project_root, "models", model_path_dict["markowitz"])
                        optimizer = None
                        
                        # Try loading existing optimizer
                        if os.path.exists(model_file):
                            optimizer = MarkowitzOptimizer.load_model(model_file)
                            st.info(f"Loaded Markowitz model: {model_file}")
                        else:
                            st.warning("Saved Markowitz model not found. Creating a new instance.")
                        
                        # Initialize or refresh optimizer if needed
                        if optimizer is None:
                            optimizer = MarkowitzOptimizer(tickers=[], csv_file_path=wide_csv_path)
                        
                        # If CSV path changed or returns not loaded, (re)fetch
                        if (getattr(optimizer, "csv_file_path", None) != wide_csv_path) or (optimizer.returns is None):
                            optimizer.csv_file_path = wide_csv_path
                            optimizer.fetch_data()
                        
                        # Clean user tickers
                        cleaned_tickers = [t.strip().upper() for t in tickers if t.strip()]
                        available_tickers = list(optimizer.mean_returns.index)
                        
                        # Intersect
                        selected_tickers = [t for t in cleaned_tickers if t in available_tickers] if cleaned_tickers else available_tickers
                        
                        if cleaned_tickers and not selected_tickers:
                            st.error("None of the selected tickers are present in the data after pivot.")
                            st.info(f"Available tickers ({len(available_tickers)}): {', '.join(available_tickers[:60])}" + (" ..." if len(available_tickers) > 60 else ""))
                            st.stop()
                        
                        optimizer.tickers = selected_tickers
                        
                        # Reduce mean returns & covariance to selected tickers
                        optimizer.mean_returns = optimizer.mean_returns[optimizer.tickers]
                        optimizer.cov_matrix = optimizer.cov_matrix.loc[optimizer.tickers, optimizer.tickers]
                        
                        if len(optimizer.tickers) < 2:
                            st.warning("Need at least two tickers for diversification. Proceeding with single asset stats.")
                        
                        # Run optimization
                        weights_array = optimizer.optimize(target_return=target_return if 'target_return' in locals() else None)
                        weights = dict(zip(optimizer.tickers, weights_array))
                        port_return, port_vol, sharpe = optimizer.portfolio_stats(weights_array)
                        
                        st.success("Markowitz optimization complete.")
                        st.subheader("Optimal Weights")
                        weight_df = pd.DataFrame(
                            [{"Ticker": k, "Weight": f"{v:.2%}"} for k, v in weights.items()]
                        )
                        st.dataframe(weight_df, use_container_width=True)
                        
                        fig_w = px.pie(
                            names=list(weights.keys()),
                            values=list(weights.values()),
                            title="Portfolio Allocation (Markowitz)"
                        )
                        st.plotly_chart(fig_w, use_container_width=True, key="markowitz_pie")
                        
                        col_a, col_b, col_c = st.columns(3)
                        col_a.metric("Expected Annual Return", f"{port_return:.2%}")
                        col_b.metric("Annual Volatility", f"{port_vol:.2%}")
                        col_c.metric("Sharpe Ratio", f"{sharpe:.3f}")
                    
                    except Exception as exc:
                        st.error(f"Markowitz optimization failed: {exc}")
                        st.stop()
                
                elif model_type == "capm":
                    try:
                        model_file = os.path.join(project_root, "models", model_path_dict["capm"])
                        capm_model = None
                        if os.path.exists(model_file):
                            capm_model = CAPMOptimizer.load_model(model_file)
                            st.info(f"Loaded CAPM model: {model_file}")
                        else:
                            st.warning("Saved CAPM model not found. Creating a new instance.")
                        
                        if capm_model is None:
                            capm_model = CAPMOptimizer(
                                tickers=tickers,
                                market_ticker=market_ticker,
                                csv_file_path=data_csv_path
                            )
                        # Ensure data path set
                        if capm_model.csv_file_path is None:
                            capm_model.csv_file_path = data_csv_path
                        if not os.path.exists(capm_model.csv_file_path):
                            st.error(f"Returns CSV not found: {capm_model.csv_file_path}")
                            st.stop()
                        
                        # Run optimization
                        weights, expected_returns, betas = capm_model.optimize_portfolio(market_return=market_return)
                        
                        st.success("CAPM optimization complete.")
                        st.subheader("Weights (Risk-adjusted)")
                        weights_df = pd.DataFrame([
                            {
                                "Ticker": t,
                                "Weight": f"{weights[t]:.2%}",
                                "Expected Return": f"{expected_returns[t]:.2%}",
                                "Beta": f"{betas[t]:.3f}"
                            }
                            for t in weights
                            if (not tickers) or (t in tickers)
                        ])
                        st.dataframe(weights_df, use_container_width=True)
                        
                        fig_capm_pie = px.pie(
                            names=list(weights.keys()),
                            values=list(weights.values()),
                            title="Portfolio Allocation (CAPM)"
                        )
                        st.plotly_chart(fig_capm_pie, use_container_width=True, key="capm_pie")
                        
                        # Bar charts
                        fig_er = px.bar(
                            x=list(expected_returns.keys()),
                            y=list(expected_returns.values()),
                            title="Expected Returns (CAPM)",
                            labels={"x": "Ticker", "y": "Expected Return"}
                        )
                        st.plotly_chart(fig_er, use_container_width=True, key="capm_er_bar")
                        
                        fig_beta = px.bar(
                            x=list(betas.keys()),
                            y=list(betas.values()),
                            title="Betas (Systematic Risk)",
                            labels={"x": "Ticker", "y": "Beta"}
                        )
                        st.plotly_chart(fig_beta, use_container_width=True, key="capm_beta_bar")
                    
                    except Exception as e:
                        st.error(f"CAPM optimization failed: {e}")
                        st.stop()
    
    # Debug information
    with st.expander("Debug Information", expanded=False):
        st.write(f"Selected Model Type: {model_type}")
        st.write(f"Tickers: {tickers}")
        if model_type == "lstm-ogdm hybrid":
            st.write(f"Current Allocations: {current_allocations}")
            st.write(f"Sequence Length: {sequence_length}")
        # Markowitz/CAPM specific inputs
        if model_type == "markowitz":
            st.write(f"Target Return: {target_return if 'target_return' in locals() else None}")
        if model_type == "capm":
            st.write(f"Market Return: {market_return if 'market_return' in locals() else None}")
        st.write("---")
        st.write("Verify data & model files exist in the models and data/processed directories.")

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
    Render Model Performance using real evaluation artifacts:
      - metrics_table.csv (walk-forward backtest: HYBRID, LSTM, OGDM, baselines)
      - portfolio_equity.csv (equity curves)
      - lstm_evaluation_results.csv (per-ticker regression error metrics)
    Falls back gracefully if artifacts are missing.
    """
    st.header("Model Performance Metrics")

    perf_dir = os.path.join(project_root, "evaluation_results", "perf")
    lstm_dir = os.path.join(project_root, "evaluation_results", "lstm")

    metrics_path = os.path.join(perf_dir, "metrics_table.csv")
    equity_path = os.path.join(perf_dir, "portfolio_equity.csv")
    lstm_eval_path = os.path.join(lstm_dir, "lstm_evaluation_results.csv")

    def _safe_read(path, **kwargs):
        if os.path.exists(path):
            try:
                return pd.read_csv(path, **kwargs)
            except Exception as e:
                st.warning(f"Failed to read {os.path.basename(path)}: {e}")
        return None

    metrics_df = _safe_read(metrics_path)
    equity_df = _safe_read(equity_path)
    lstm_df = _safe_read(lstm_eval_path, index_col=0)

    # --- Hybrid / primary model KPI summary ---
    if metrics_df is not None and not metrics_df.empty:
        # Ensure ordering (HYBRID first)
        metrics_df["model"] = metrics_df["model"].astype(str)
        metrics_df = metrics_df.set_index("model")
        hybrid_row = metrics_df.loc["HYBRID"] if "HYBRID" in metrics_df.index else None

        # Compute HYBRID max drawdown from equity curve if available
        hybrid_dd = np.nan
        if equity_df is not None and not equity_df.empty and "HYBRID" in equity_df["model"].unique():
            try:
                eq_h = equity_df[equity_df["model"] == "HYBRID"].copy()
                eq_h["peak"] = eq_h["equity"].cummax()
                eq_h["drawdown"] = eq_h["equity"] / eq_h["peak"] - 1
                hybrid_dd = float(eq_h["drawdown"].min())
            except Exception:
                pass

        c1, c2, c3, c4 = st.columns(4)
        if hybrid_row is not None:
            c1.metric("HYBRID Sharpe", f"{hybrid_row.get('sharpe', np.nan):.3f}")
            c2.metric("HYBRID Cum Return", f"{hybrid_row.get('cumulative_return', np.nan):.2%}")
            c3.metric("HYBRID Directional Acc", f"{hybrid_row.get('directional_accuracy', np.nan):.2%}")
            c4.metric("HYBRID Max Drawdown", f"{hybrid_dd:.2%}" if np.isfinite(hybrid_dd) else "N/A")
        else:
            c1.metric("HYBRID Sharpe", "N/A")
            c2.metric("HYBRID Cum Return", "N/A")
            c3.metric("HYBRID Directional Acc", "N/A")
            c4.metric("HYBRID Max Drawdown", "N/A")
    else:
        st.info("metrics_table.csv not found or empty (run performance_comparison script).")

    # --- Equity Curves & Drawdowns ---
    if equity_df is not None and not equity_df.empty:
        try:
            equity_df["date"] = pd.to_datetime(equity_df["date"])
        except Exception:
            pass
        st.subheader("Equity Curves (Backtest)")
        fig_eq = px.line(equity_df, x="date", y="equity", color="model", title="Portfolio Equity")
        st.plotly_chart(fig_eq, use_container_width=True, key="perf_equity")

        # Hybrid drawdown plot
        if "HYBRID" in equity_df["model"].unique():
            eq_h = equity_df[equity_df["model"] == "HYBRID"].copy().sort_values("date")
            eq_h["peak"] = eq_h["equity"].cummax()
            eq_h["drawdown"] = eq_h["equity"] / eq_h["peak"] - 1
            fig_dd = px.area(eq_h, x="date", y="drawdown", title="HYBRID Drawdown", color_discrete_sequence=["#d62728"])
            fig_dd.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig_dd, use_container_width=True, key="perf_hybrid_dd")
    else:
        st.info("portfolio_equity.csv not found or empty.")

    # --- Full Metrics Table ---
    if metrics_df is not None and not metrics_df.empty:
        ordered = pd.concat([
            metrics_df.loc[["HYBRID"]] if "HYBRID" in metrics_df.index else pd.DataFrame(),
            metrics_df.drop(index=["HYBRID"], errors="ignore")
        ])
        st.subheader("Model Metrics (Walk-Forward)")
        st.dataframe(
            ordered.reset_index().round({
                "mse": 6, "mae": 6, "directional_accuracy": 4,
                "avg_return": 6, "volatility": 6, "sharpe": 4, "cumulative_return": 4
            }),
            use_container_width=True
        )

    # --- LSTM Per-Ticker Error Analysis ---
    st.subheader("LSTM Per-Ticker Error Metrics")
    if lstm_df is not None and not lstm_df.empty:
        # If 'overall' row exists keep separate
        overall_row = None
        if "overall" in lstm_df.index:
            overall_row = lstm_df.loc["overall"].copy()
            lstm_core = lstm_df.drop(index=["overall"])
        else:
            lstm_core = lstm_df

        # Clean possible unnamed columns
        lstm_core = lstm_core.copy()
        # Sort by MAE ascending (better to worse)
        sort_col = "MAE" if "MAE" in lstm_core.columns else lstm_core.columns[0]
        lstm_core_sorted = lstm_core.sort_values(sort_col)

        # Display top 12
        st.write("Top 12 tickers (lowest MAE):")
        st.dataframe(lstm_core_sorted.head(12).round(4), use_container_width=True)

        # Bar chart of MAE (top 12)
        if "MAE" in lstm_core_sorted.columns:
            mae_plot = lstm_core_sorted.head(12).reset_index().rename(columns={"index": "ticker"})
            fig_mae = px.bar(mae_plot, x="ticker", y="MAE", title="LSTM MAE (Lower is Better)")
            st.plotly_chart(fig_mae, use_container_width=True, key="perf_lstm_mae")

        # Overall row metrics summary
        if overall_row is not None:
            c1, c2, c3, c4, c5 = st.columns(5)
            if "MSE" in overall_row: c1.metric("Overall MSE", f"{overall_row['MSE']:.2f}")
            if "RMSE" in overall_row: c2.metric("Overall RMSE", f"{overall_row['RMSE']:.2f}")
            if "MAE" in overall_row: c3.metric("Overall MAE", f"{overall_row['MAE']:.2f}")
            if "R²" in overall_row: c4.metric("Overall R²", f"{overall_row['R²']:.3f}")
            if "MAPE" in overall_row: c5.metric("Overall MAPE", f"{overall_row['MAPE']:.2f}")
    else:
        st.info("lstm_evaluation_results.csv not found or empty (run evaluate_lstm script).")

# Main dashboard layout
tab1, tab2, tab3, tab4 = st.tabs([
    "Portfolio Optimization",
    "Historical Analysis",
    "Model Performance",
    "Advanced Analytics"
])

with tab1:
    render_tab_portfolio(model_type, project_root)
with tab2:
    render_tab_historical()
with tab3:
    render_tab_performance()
with tab4:
    render_tab_advanced_analytics(project_root)



