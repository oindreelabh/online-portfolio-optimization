import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle
import argparse
import os
from src.utils.constants import TICKERS
from src.utils.logger import setup_logger

logger = setup_logger(os.path.basename(__file__).replace(".py", ""))

class CAPMOptimizer:
    def __init__(self, tickers, market_ticker='^GSPC', start_date=None, end_date=None, csv_file_path=None):
        self.tickers = tickers
        self.market_ticker = market_ticker
        self.start_date = start_date
        self.end_date = end_date
        self.csv_file_path = csv_file_path
        self.betas = {}
        self.alphas = {}
        self.risk_free_rate = 0.02  # 2% annual risk-free rate

    def fetch_data(self):
        """Fetch stock and market data from CSV file"""
        if self.csv_file_path:
            # Read data from CSV file
            data = pd.read_csv(self.csv_file_path, parse_dates=['date'])
            
            # Check if market ticker exists in data
            if self.market_ticker not in data['ticker'].values:
                raise ValueError(f"Market ticker {self.market_ticker} not found in data")
            
            # Filter for specified tickers if they exist in data
            available_tickers = [ticker for ticker in self.tickers if ticker in data['ticker'].values]
            if not available_tickers:
                raise ValueError(f"None of the specified tickers {self.tickers} found in data")
            
            self.tickers = available_tickers
            
            # Pivot data to get returns for each ticker
            stock_data = data[data['ticker'].isin(self.tickers)].pivot(index='date', columns='ticker', values='returns')
            stock_returns = stock_data.dropna()
            
            # Get market returns
            market_data = data[data['ticker'] == self.market_ticker].set_index('date')['returns']
            market_returns = market_data.dropna()
        else:
            # Throw error instead of downloading data
            raise ValueError("CSV file path is required. Downloading from yfinance is not supported.")

        return stock_returns, market_returns

    def calculate_beta_alpha(self, stock_returns, market_returns):
        """Calculate beta and alpha for each stock"""
        for ticker in self.tickers:
            # Align dates
            aligned_data = pd.concat([stock_returns[ticker], market_returns], axis=1).dropna()

            if len(aligned_data) > 0:
                X = aligned_data.iloc[:, 1].values.reshape(-1, 1)  # Market returns
                y = aligned_data.iloc[:, 0].values  # Stock returns

                model = LinearRegression()
                model.fit(X, y)

                self.betas[ticker] = model.coef_[0]
                self.alphas[ticker] = model.intercept_
            else:
                self.betas[ticker] = 1.0
                self.alphas[ticker] = 0.0

    def calculate_expected_returns(self, market_return):
        """Calculate expected returns using CAPM"""
        expected_returns = {}
        for ticker in self.tickers:
            expected_return = self.risk_free_rate + self.betas[ticker] * (market_return - self.risk_free_rate)
            expected_returns[ticker] = expected_return
        return expected_returns

    def optimize_portfolio(self, market_return=0.10):
        """Optimize portfolio using CAPM expected returns"""
        stock_returns, market_returns = self.fetch_data()
        self.calculate_beta_alpha(stock_returns, market_returns)

        expected_returns = self.calculate_expected_returns(market_return)

        # Simple optimization: weight proportional to expected return/beta ratio
        risk_adjusted_returns = {ticker: expected_returns[ticker] / max(self.betas[ticker], 0.1)
                               for ticker in self.tickers}

        total_score = sum(risk_adjusted_returns.values())
        weights = {ticker: score / total_score for ticker, score in risk_adjusted_returns.items()}

        return weights, expected_returns, self.betas

    def save_model(self, filepath):
        """Save the model to a pickle file"""
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self, f)
            logger.info(f"Model saved successfully to {filepath}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise

    @classmethod
    def load_model(cls, filepath):
        """Load a model from a pickle file"""
        try:
            with open(filepath, 'rb') as f:
                model = pickle.load(f)
            logger.info(f"Model loaded successfully from {filepath}")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

def main():
    """Main function to run CAPM optimization from command line"""
    parser = argparse.ArgumentParser(description='Run CAPM Portfolio Optimization')
    parser.add_argument('--data_path', help='Path to CSV file with stock price data')
    parser.add_argument('--model_save_path', required=True, help='Path to save the trained model')
    parser.add_argument('--market_ticker', default='^GSPC', help='Market index ticker (default: S&P 500)')
    parser.add_argument('--market_return', type=float, default=0.10, help='Expected market return (default: 0.10)')
    parser.add_argument('--start_date', default='2022-01-01', help='Start date for data (YYYY-MM-DD)')
    parser.add_argument('--end_date', default='2024-01-01', help='End date for data (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    # Ensure model directory exists
    os.makedirs(os.path.dirname(args.model_save_path), exist_ok=True)
    
    # Parse tickers
    tickers = TICKERS

    # Initialize optimizer
    optimizer = CAPMOptimizer(
        tickers=tickers,
        market_ticker=args.market_ticker,
        start_date=args.start_date,
        end_date=args.end_date,
        csv_file_path=args.data_path
    )
    
    # Optimize portfolio
    weights, expected_returns, betas = optimizer.optimize_portfolio(market_return=args.market_return)
    
    # Log results
    logger.info(f"Optimal weights: {weights}")
    logger.info(f"Expected returns: {expected_returns}")
    logger.info(f"Betas: {betas}")
    
    # Save model
    optimizer.save_model(args.model_save_path)
    print(f"CAPM model saved to {args.model_save_path}")

if __name__ == "__main__":
    main()