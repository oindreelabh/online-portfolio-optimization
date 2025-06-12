from src.utils.logger import setup_logger
import os
import numpy as np

logger = setup_logger(os.path.basename(__file__).replace(".py", ""))

def mean_variance_optimization(expected_returns, cov_matrix):
    """
    expected_returns: np.array of predicted returns
    cov_matrix: np.array of covariance matrix
    Returns optimal weights.
    """
    n = len(expected_returns)
    inv_cov = np.linalg.inv(cov_matrix)
    ones = np.ones(n)
    weights = inv_cov @ expected_returns / (ones @ inv_cov @ expected_returns)
    logger.info(f"Optimal weights calculated: {weights}")
    return weights
