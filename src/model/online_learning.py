from src.utils.logger import setup_logger
import os
import pandas as pd
import numpy as np
import joblib
import argparse

logger = setup_logger(os.path.basename(__file__).replace(".py", ""))

class OnlineGradientDescentMomentum:
    def __init__(self, n_features, learning_rate=0.01, momentum=0.9, model_path=None):
        self.lr = learning_rate
        self.momentum = momentum
        self.weights = np.zeros(n_features)
        self.velocity = np.zeros(n_features)
        self.is_fitted = False
        self.model_path = model_path
        if model_path and os.path.exists(model_path):
            self.load(model_path)

    def predict(self, X):
        preds = X.dot(self.weights)
        logger.info(f"Made predictions for {X.shape[0]} samples.")
        return preds

    def partial_fit(self, X, y):
        """
        Perform one step of gradient descent with momentum.
        X: np.array shape (n_samples, n_features)
        y: np.array shape (n_samples,)
        """
        preds = self.predict(X)
        error = preds - y
        grad = X.T.dot(error) / len(y)  # gradient of MSE loss

        # Update velocity and weights
        self.velocity = self.momentum * self.velocity - self.lr * grad
        self.weights += self.velocity

        self.is_fitted = True
        logger.info("Performed partial_fit step with momentum.")

    def save(self, model_path=None):
        path = model_path if model_path else self.model_path
        if not path:
            raise ValueError("Model path must be specified to save the model.")
        joblib.dump({'weights': self.weights, 'velocity': self.velocity}, path)
        logger.info(f"Model saved to {path}.")

    def load(self, model_path=None):
        path = model_path if model_path else self.model_path
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at {path}")
        data = joblib.load(path)
        self.weights = data['weights']
        self.velocity = data['velocity']
        self.is_fitted = True
        logger.info(f"Model loaded from {path}.")


def load_data(data_path, target_col):
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    # Select only numeric columns except the target
    feature_cols = [col for col in df.select_dtypes(include=[np.number]).columns if col != target_col]
    logger.info(f"Numeric feature columns: {feature_cols}")
    X = df[feature_cols].fillna(0).to_numpy()
    y = df[target_col].fillna(0).to_numpy()
    logger.info(f"Loaded {len(df)} records with {len(feature_cols)} numeric features.")
    return X, y


def run_ogdm_training(
    data_path: str,
    target_col: str,
    model_path: str,
    learning_rate: float = 0.01,
    momentum: float = 0.9
):
    X, y = load_data(data_path, target_col)
    model = OnlineGradientDescentMomentum(n_features=X.shape[1], learning_rate=learning_rate, momentum=momentum, model_path=model_path)
    model.partial_fit(X, y)
    model.save(model_path)
    logger.info("OGDM training step completed and model saved.")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run Online Gradient Descent Momentum training and prediction")
    parser.add_argument("--data_path", type=str, required=True, help="Path to input data CSV")
    parser.add_argument("--target_col", type=str, required=True, help="Target column name")
    parser.add_argument("--model_path", type=str, required=True, help="Path to save the trained model")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate for OGDM")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum factor for OGDM")

    args = parser.parse_args()

    run_ogdm_training(
        args.data_path,
        args.target_col,
        args.model_path,
        args.learning_rate,
        args.momentum
    )