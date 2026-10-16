"""
Modeling module.
Phase 4 of CRISP-DM: Modeling.
Implements Multiple Linear Regression and regularized variants (Ridge, Lasso, ElasticNet).
"""
from typing import Dict, Any, List
import numpy as np
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV


class RegressionTrainer:
    """Trains and evaluates multiple linear regression models and regularized variants."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.trained_models: Dict[str, Any] = {}

    def train_linear_regression(
        self, X_train: np.ndarray, y_train: np.ndarray, model_name: str = "Linear Regression"
    ) -> LinearRegression:
        """Trains standard Ordinary Least Squares (OLS) Multiple Linear Regression."""
        model = LinearRegression()
        model.fit(X_train, y_train)
        self.trained_models[model_name] = model
        return model

    def train_ridge(
        self, X_train: np.ndarray, y_train: np.ndarray, model_name: str = "Ridge Regression"
    ) -> RidgeCV:
        """Trains Ridge Regression (L2 regularization) with cross-validated alpha selection."""
        alphas = np.logspace(-3, 3, 50)
        model = RidgeCV(alphas=alphas, cv=5)
        model.fit(X_train, y_train)
        self.trained_models[model_name] = model
        return model

    def train_lasso(
        self, X_train: np.ndarray, y_train: np.ndarray, model_name: str = "Lasso Regression"
    ) -> LassoCV:
        """Trains Lasso Regression (L1 regularization) with cross-validated alpha selection."""
        model = LassoCV(cv=5, random_state=self.random_state, max_iter=5000)
        model.fit(X_train, y_train)
        self.trained_models[model_name] = model
        return model

    def train_elastic_net(
        self, X_train: np.ndarray, y_train: np.ndarray, model_name: str = "ElasticNet"
    ) -> ElasticNetCV:
        """Trains ElasticNet Regression with cross-validated L1/L2 mixing."""
        l1_ratios = [0.1, 0.5, 0.7, 0.9, 0.99]
        model = ElasticNetCV(l1_ratio=l1_ratios, cv=5, random_state=self.random_state, max_iter=5000)
        model.fit(X_train, y_train)
        self.trained_models[model_name] = model
        return model

    def get_model(self, model_name: str):
        return self.trained_models.get(model_name)
