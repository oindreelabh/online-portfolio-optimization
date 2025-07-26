import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
import logging

logger = logging.getLogger(__name__)

class CAPMOptimizer:
    def __init__(self, tickers, market_ticker='^GSPC', start_date=None, end_date=None):
        self.tickers = tickers
        self.market_ticker = market_ticker
        self.start_date = start_date
        self.end_date = end_date
        self.betas = {}
        self.alphas = {}
        self.risk_free_rate = 0.02  # 2% annual risk-free rate

    def fetch_data(self):
        """Fetch stock and market data"""
        # Fetch stock data
        stock_data = yf.download(self.tickers, start=self.start_date, end=self.end_date)['Adj Close']
        stock_returns = stock_data.pct_change().dropna()

        # Fetch market data
        market_data = yf.download(self.market_ticker, start=self.start_date, end=self.end_date)['Adj Close']
        market_returns = market_data.pct_change().dropna()

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