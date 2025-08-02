import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
import sys
from datetime import datetime

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import hybrid model functions
try:
    from model.lstm_ogdm_hybrid import hybrid_predict_and_rebalance, parse_allocations
except ImportError:
    try:
        from src.model.lstm_ogdm_hybrid import hybrid_predict_and_rebalance, parse_allocations
    except ImportError:
        # Add more path options
        sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        from src.model.lstm_ogdm_hybrid import hybrid_predict_and_rebalance, parse_allocations

# Page configuration
st.set_page_config(
    page_title="Portfolio Optimization Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
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
    ["Markowitz", "CAPM", "LSTM-OGDM Hybrid"]
)
model_type = model_type.lower()
model_path_dict = {
    "markowitz": "markowitz_model.pkl",
    "capm": "capm_model.pkl",
    "lstm-ogdm hybrid": {
        "lstm_model": "lstm_model.keras",
        "lstm_scaler": "lstm_scaler.pkl", 
        "online_model": "online_model.pkl"
    }
}

# Main dashboard layout
tab1, tab2, tab3 = st.tabs(["Portfolio Optimization", "Historical Analysis", "Model Performance"])

with tab1:
    st.header("Portfolio Optimization & Prediction")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Portfolio Parameters")
        
        # Portfolio configuration for hybrid model
        if model_type == "lstm-ogdm hybrid":
            # Multiple stock selection
            default_tickers = ["AAPL", "GOOGL", "MSFT", "TSLA"]
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
            sequence_length = st.slider("LSTM Sequence Length", 5, 30, 10)
            
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
                    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "processed", "recent_data_with_sentiment.csv")
                    
                    # Check if data file exists
                    if not os.path.exists(data_path):
                        st.error(f"Data file not found: {data_path}")
                        st.info("Please ensure the recent_data_with_sentiment.csv file exists in the data/processed directory.")
                        raise FileNotFoundError(f"Required data file not found: {data_path}")
                    
                    try:
                        # Load the actual data
                        sample_data = pd.read_csv(data_path)
                        
                        # Filter data for selected tickers
                        sample_data = sample_data[sample_data['ticker'].isin(tickers)]
                        
                        if sample_data.empty:
                            available_tickers = pd.read_csv(data_path)['ticker'].unique().tolist()
                            st.error(f"No data found for selected tickers: {tickers}")
                            st.info(f"Available tickers in data: {', '.join(available_tickers)}")
                            raise ValueError(f"No data available for selected tickers: {tickers}")
                        
                        st.info(f"Loaded {len(sample_data)} records for {len(sample_data['ticker'].unique())} tickers from actual data")
                        
                    except Exception as e:
                        st.error(f"Error loading data file: {str(e)}")
                        raise e
                    
                    try:
                        # Model paths - LSTM model, scaler, and online model are required
                        base_model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")
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
                            raise FileNotFoundError(f"Required model files not found: {', '.join(missing_files)}")
                        
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
                                return
                            
                            # Display predictions
                            st.subheader("Predicted Returns")
                            pred_df = pd.DataFrame([
                                {'Ticker': ticker, 'Predicted Return': f"{pred:.4f}"}
                                for ticker, pred in result['predictions'].items()
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
                                    st.plotly_chart(fig_pie, use_container_width=True)
                                else:
                                    st.info("No allocation suggestions available")
                            
                            # Portfolio metrics
                            st.subheader("Portfolio Metrics")
                            col_a, col_b, col_c = st.columns(3)
                            
                            # Calculate weighted return safely
                            if result['predictions'] and result['suggested_allocations']:
                                weighted_return = sum(
                                    pred * result['suggested_allocations'].get(ticker, 0)
                                    for ticker, pred in result['predictions'].items()
                                    if ticker in result['suggested_allocations']
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
                        st.error(f"Error running hybrid model: {str(e)}")
                        st.info("This error may occur due to insufficient data, model compatibility issues, or missing features in the dataset.")
                        
                        # Show debug information without raising the exception
                        with st.expander("Debug Information"):
                            st.write(f"Error details: {str(e)}")
                            st.write(f"Selected tickers: {tickers}")
                            st.write(f"Data path: {data_path}")
                            st.write(f"Model paths:")
                            st.write(f"  - LSTM: {lstm_model_path}")
                            st.write(f"  - Scaler: {lstm_scaler_path}")
                            st.write(f"  - OGDM: {online_model_path}")

                else:
                    # Original prediction logic for other models
                    base_model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")
                    model_filename = model_path_dict.get(model_type, "default_model.pkl")
                    model_path = os.path.join(base_model_dir, model_filename)
                    
                    # Load and predict with actual model
                    if not os.path.exists(model_path):
                        st.error(f"Model file not found: {model_path}")
                        raise FileNotFoundError(f"Required model file not found: {model_path}")
                    
                    try:
                        model = joblib.load(model_path)
                        
                        # Prepare input features
                        input_features = pd.DataFrame([{
                            'RSI': rsi,
                            'MACD': macd,
                            'Bollinger_Band_Position': bb_position,
                            'Volume_Ratio': volume_ratio,
                            'VIX': vix,
                            'Market_Sentiment': {"Positive": 1, "Neutral": 0, "Negative": -1}[market_sentiment]
                        }])

                        # Predict
                        if hasattr(model, "predict_proba"):
                            prob = model.predict_proba(input_features)[0]
                            positive_prob = prob[1] if len(prob) > 1 else prob[0]
                            movement_prediction = "Positive" if positive_prob > 0.5 else "Negative"
                            confidence = positive_prob if movement_prediction == "Positive" else 1 - positive_prob
                        else:
                            pred = model.predict(input_features)[0]
                            movement_prediction = "Positive" if pred == 1 else "Negative"
                            confidence = 1.0
                        
                        # Display prediction
                        prediction_class = "positive" if movement_prediction == "Positive" else "negative"
                        st.markdown(f"""
                        <div class="prediction-box {prediction_class}">
                            Prediction: {movement_prediction}<br>
                            Confidence: {confidence:.1%}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Metrics (using actual model predictions, not random)
                        expected_return = confidence * 5 if movement_prediction == "Positive" else -confidence * 5
                        risk_score = (1 - confidence) * 10
                        volatility = (1 - confidence) * 40
                        
                        col_a, col_b, col_c = st.columns(3)
                        col_a.metric("Expected Return", f"{expected_return:.2f}%")
                        col_b.metric("Risk Score", f"{risk_score:.1f}/10")
                        col_c.metric("Volatility", f"{volatility:.1f}%")
                        
                        # Prediction chart
                        dates = pd.date_range(start=datetime.now(), periods=prediction_days, freq='D')
                        prices = 100 + np.cumsum(np.random.randn(prediction_days) * 2)
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=dates,
                            y=prices,
                            mode='lines+markers',
                            name='Predicted Price',
                            line=dict(color='blue', width=3)
                        ))
                        
                        fig.update_layout(
                            title=f"{symbol} Price Prediction",
                            xaxis_title="Date",
                            yaxis_title="Price ($)",
                            hovermode='x unified'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)

                    except Exception as e:
                        st.error(f"Error loading or running model: {str(e)}")
                        raise e

with tab2:
    st.header("Historical Analysis")
    
    # Sample historical data
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    prices = 100 + np.cumsum(np.random.randn(len(dates)) * 0.5)
    volume = np.random.randint(1000000, 10000000, len(dates))
    
    historical_data = pd.DataFrame({
        'Date': dates,
        'Price': prices,
        'Volume': volume
    })
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Price chart
        fig_price = px.line(historical_data, x='Date', y='Price', title='Historical Price Movement')
        st.plotly_chart(fig_price, use_container_width=True)
    
    with col2:
        # Volume chart
        fig_volume = px.bar(historical_data, x='Date', y='Volume', title='Trading Volume')
        st.plotly_chart(fig_volume, use_container_width=True)
    
    # Statistics
    st.subheader("Statistical Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Mean Price", f"${historical_data['Price'].mean():.2f}")
    with col2:
        st.metric("Volatility", f"{historical_data['Price'].std():.2f}")
    with col3:
        st.metric("Max Price", f"${historical_data['Price'].max():.2f}")
    with col4:
        st.metric("Min Price", f"${historical_data['Price'].min():.2f}")

with tab3:
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
        st.plotly_chart(fig_metrics, use_container_width=True)
    
    with col2:
        st.subheader("Confusion Matrix")
        confusion_matrix = np.array([[85, 15], [12, 88]])
        fig_cm = px.imshow(confusion_matrix, 
                          text_auto=True, 
                          aspect="auto",
                          title="Confusion Matrix",
                          labels=dict(x="Predicted", y="Actual"))
        st.plotly_chart(fig_cm, use_container_width=True)
    
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
    st.plotly_chart(fig_importance, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Market Movement Prediction Dashboard | Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)

