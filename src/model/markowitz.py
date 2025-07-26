import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
import logging

logger = logging.getLogger(__name__)

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
            
            # Filter by tickers if specified
            if self.tickers:
                available_tickers = [ticker for ticker in self.tickers if ticker in data.columns]
                if not available_tickers:
                    raise ValueError(f"None of the specified tickers {self.tickers} found in CSV file")
                data = data[available_tickers]
                logger.info(f"Using tickers: {available_tickers}")
            
            # Filter by date range if specified
            if self.start_date and self.end_date:
                data = data.loc[self.start_date:self.end_date]
            
        else:
            raise ValueError("CSV file path must be provided")
        
        self.returns = data.pct_change().dropna()
        self.mean_returns = self.returns.mean()
        self.cov_matrix = self.returns.cov()
        return self.returns

    def portfolio_stats(self, weights):
        """Calculate portfolio statistics"""
        portfolio_return = np.sum(self.mean_returns * weights) * 252
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix * 252, weights)))
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