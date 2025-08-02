import numpy as np
import pandas as pd
from scipy.optimize import minimize
import pickle
import argparse
import os
from src.utils.logger import setup_logger

logger = setup_logger(os.path.basename(__file__).replace(".py", ""))

class MarkowitzOptimizer:
    def __init__(self, tickers, csv_file_path=None, start_date=None, end_date=None):
        self.tickers = tickers
        self.csv_file_path = csv_file_path
        self.start_date = start_date
        self.end_date = end_date
        self.returns = None
        self.mean_returns = None
        self.cov_matrix = None

    def fetch_data(self):
        """Fetch historical price data from CSV file and calculate returns"""
        if self.csv_file_path:
            # Read data from CSV file
            data = pd.read_csv(self.csv_file_path, index_col=0, parse_dates=True)
            
            # Filter only numeric columns (price data)
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            data = data[numeric_columns]
            
            # Remove columns with all NaN values
            data = data.dropna(axis=1, how='all')
            
            # Update tickers to only include valid numeric columns
            self.tickers = data.columns.tolist()
            
            if data.empty:
                raise ValueError("No numeric price data found in the CSV file")
                
        else:
            raise ValueError("CSV file path must be provided")
            
        # Calculate returns and handle NaN values
        self.returns = data.pct_change().dropna()
        
        # Remove any remaining NaN or infinite values
        self.returns = self.returns.replace([np.inf, -np.inf], np.nan).dropna()
        
        # Check if we have enough data after cleaning
        if self.returns.empty:
            raise ValueError("No valid return data after cleaning NaN/infinite values")
            
        self.mean_returns = self.returns.mean()
        self.cov_matrix = self.returns.cov()
        
        # Validate that mean returns and covariance matrix don't contain NaN
        if self.mean_returns.isna().any():
            logger.warning("NaN values found in mean returns, filling with 0")
            self.mean_returns = self.mean_returns.fillna(0)
            
        if self.cov_matrix.isna().any().any():
            logger.warning("NaN values found in covariance matrix, using regularization")
            # Add small regularization term to diagonal
            self.cov_matrix = self.cov_matrix.fillna(0)
            np.fill_diagonal(self.cov_matrix.values, self.cov_matrix.values.diagonal() + 1e-8)
        
        logger.info(f"Data loaded successfully. Shape: {self.returns.shape}")
        return self.returns

    def portfolio_stats(self, weights):
        """Calculate portfolio statistics"""
        portfolio_return = np.sum(self.mean_returns * weights) * 252
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix * 252, weights)))
        
        # Handle division by zero
        if portfolio_vol == 0:
            sharpe_ratio = 0
        else:
            sharpe_ratio = portfolio_return / portfolio_vol
            
        return portfolio_return, portfolio_vol, sharpe_ratio

    def negative_sharpe(self, weights):
        """Objective function to minimize (negative Sharpe ratio)"""
        return -self.portfolio_stats(weights)[2]

    def optimize(self, target_return=None):
        """Optimize portfolio weights"""
        num_assets = len(self.tickers)

        # Constraints
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        if target_return:
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                {'type': 'eq', 'fun': lambda x: self.portfolio_stats(x)[0] - target_return}
            ]

        # Bounds (no short selling)
        bounds = tuple((0, 1) for _ in range(num_assets))

        # Initial guess (equal weights)
        x0 = np.array([1/num_assets] * num_assets)

        # Optimize
        result = minimize(self.negative_sharpe, x0, method='SLSQP',
                         bounds=bounds, constraints=constraints)

        return result.x

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
    """Main function to run Markowitz optimization from command line"""
    parser = argparse.ArgumentParser(description='Run Markowitz Portfolio Optimization')
    parser.add_argument('--data_path', required=True, help='Path to CSV file with stock price data')
    parser.add_argument('--model_save_path', required=True, help='Path to save the trained model')
    parser.add_argument('--target_return', type=float, default=None, help='Target return for optimization')
    
    args = parser.parse_args()
    
    # Ensure model directory exists
    os.makedirs(os.path.dirname(args.model_save_path), exist_ok=True)
    
    # Initialize optimizer with empty tickers list (will be updated in fetch_data)
    optimizer = MarkowitzOptimizer(tickers=[], csv_file_path=args.data_path)
    
    # Fetch data and optimize
    optimizer.fetch_data()
    
    logger.info(f"Found {len(optimizer.tickers)} numeric columns: {optimizer.tickers}")
    
    optimal_weights = optimizer.optimize(target_return=args.target_return)
    
    # Calculate portfolio statistics
    portfolio_return, portfolio_vol, sharpe_ratio = optimizer.portfolio_stats(optimal_weights)
    
    # Log results
    logger.info(f"Optimal weights: {dict(zip(optimizer.tickers, optimal_weights))}")
    logger.info(f"Expected annual return: {portfolio_return:.4f}")
    logger.info(f"Annual volatility: {portfolio_vol:.4f}")
    logger.info(f"Sharpe ratio: {sharpe_ratio:.4f}")
    
    # Save model
    optimizer.save_model(args.model_save_path)
    print(f"Markowitz model saved to {args.model_save_path}")

if __name__ == "__main__":
    main()