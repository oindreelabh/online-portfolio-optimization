import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
import sys
from datetime import datetime, timedelta

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Page configuration
st.set_page_config(
    page_title="Market Movement Prediction Dashboard",
    page_icon="📈",
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
    ["Markowitz", "CAPM", "LSTM", "OGDM"]
)
model_type = model_type.lower()

# Main dashboard layout
tab1, tab2, tab3 = st.tabs(["Prediction", "Historical Analysis", "Model Performance"])

with tab1:
    st.header("Market Movement Prediction")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Input Parameters")
        
        # Stock symbol input
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
            # Simulate model prediction (replace with actual model inference)
            with st.spinner("Generating predictions..."):
                # Mock prediction logic
                # Load your trained model (ensure the path and model type are correct)
                model_path = os.path.join(os.path.dirname(__file__), "your_model.pkl")
                model = joblib.load(model_path)

                # Prepare input features as a DataFrame or array as expected by your model
                input_features = pd.DataFrame([{
                    'RSI': rsi,
                    'MACD': macd,
                    'Bollinger_Band_Position': bb_position,
                    'Volume_Ratio': volume_ratio,
                    'VIX': vix,
                    'Market_Sentiment': {"Positive": 1, "Neutral": 0, "Negative": -1}[market_sentiment]
                }])

                # Predict probability and class
                if hasattr(model, "predict_proba"):
                    prob = model.predict_proba(input_features)[0]
                    positive_prob = prob[1] if len(prob) > 1 else prob[0]
                    movement_prediction = "Positive" if positive_prob > 0.5 else "Negative"
                    confidence = positive_prob if movement_prediction == "Positive" else 1 - positive_prob
                else:
                    pred = model.predict(input_features)[0]
                    movement_prediction = "Positive" if pred == 1 else "Negative"
                    confidence = 1.0  # or set to None if not available
                
                # Display prediction
                prediction_class = "positive" if movement_prediction == "Positive" else "negative"
                st.markdown(f"""
                <div class="prediction-box {prediction_class}">
                    Prediction: {movement_prediction}<br>
                    Confidence: {confidence:.1%}
                </div>
                """, unsafe_allow_html=True)
                
                # Metrics
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Expected Return", f"{np.random.uniform(-5, 5):.2f}%")
                col_b.metric("Risk Score", f"{np.random.uniform(1, 10):.1f}/10")
                col_c.metric("Volatility", f"{np.random.uniform(10, 40):.1f}%")
                
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

