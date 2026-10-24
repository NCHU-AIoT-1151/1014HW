"""
Feature Selection module.
Part of Phase 4 of CRISP-DM: Modeling.
Implements Filter (SelectKBest), Wrapper (RFE), Embedded (LassoCV), and Multicollinearity (VIF) methods.
"""
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.linear_model import LinearRegression, LassoCV
from statsmodels.stats.outliers_influence import variance_inflation_factor


class FeatureSelector:
    """Manages multiple feature selection strategies and provides comparison tools."""

    def __init__(self, feature_names: List[str], random_state: int = 42):
        self.feature_names = feature_names
        self.random_state = random_state
        self.selection_results: Dict[str, Dict] = {}

    def select_k_best(
        self, X_train: np.ndarray, y_train: np.ndarray, k: int = 12
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Filter Method: SelectKBest using F-regression ANOVA test.
        Ranks features based on univariate linear correlation with target.
        """
        n_feats = X_train.shape[1]
        k = min(k, n_feats)
        selector = SelectKBest(score_func=f_regression, k=k)
        selector.fit(X_train, y_train)
        mask = selector.get_support()
        selected_features = [f for f, m in zip(self.feature_names, mask) if m]

        self.selection_results["SelectKBest (Filter)"] = {
            "mask": mask,
            "selected_features": selected_features,
            "scores": selector.scores_,
            "pvalues": selector.pvalues_,
            "selector": selector
        }
        return mask, selected_features

    def select_rfe(
        self, X_train: np.ndarray, y_train: np.ndarray, n_features_to_select: int = 12
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Wrapper Method: Recursive Feature Elimination (RFE) with Linear Regression.
        Recursively trains estimator and removes the lowest coefficient feature.
        """
        n_feats = X_train.shape[1]
        n_features_to_select = min(n_features_to_select, max(1, n_feats // 2 if n_feats <= 6 else n_feats - 2))
        base_estimator = LinearRegression()
        rfe = RFE(estimator=base_estimator, n_features_to_select=n_features_to_select, step=1)
        rfe.fit(X_train, y_train)
        mask = rfe.support_
        selected_features = [f for f, m in zip(self.feature_names, mask) if m]

        self.selection_results["RFE (Wrapper)"] = {
            "mask": mask,
            "selected_features": selected_features,
            "ranking": rfe.ranking_,
            "selector": rfe
        }
        return mask, selected_features

    def select_lasso(
        self, X_train: np.ndarray, y_train: np.ndarray
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Embedded Method: Lasso (L1 regularization) with Cross-Validation.
        Features with non-zero coefficients are retained.
        """
        lasso = LassoCV(cv=min(5, max(2, len(y_train) // 10)), random_state=self.random_state, max_iter=5000)
        lasso.fit(X_train, y_train)
        mask = np.abs(lasso.coef_) > 1e-4
        if not np.any(mask):
            # If all zeros, keep top non-zero or largest magnitude
            max_idx = np.argmax(np.abs(lasso.coef_))
            mask[max_idx] = True

        selected_features = [f for f, m in zip(self.feature_names, mask) if m]

        self.selection_results["Lasso (Embedded)"] = {
            "mask": mask,
            "selected_features": selected_features,
            "coef": lasso.coef_,
            "alpha": lasso.alpha_,
            "selector": lasso
        }
        return mask, selected_features

    def select_vif_filter(
        self, X_train: np.ndarray, threshold: float = 10.0
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Multicollinearity Filter: Iteratively removes features with VIF > threshold.
        Solves severe collinearity between features.
        """
        current_cols = list(range(len(self.feature_names)))
        dropped = []
        min_cols = max(1, min(5, len(self.feature_names) // 2))

        while len(current_cols) > min_cols:
            X_curr = X_train[:, current_cols]
            try:
                vifs = [variance_inflation_factor(X_curr, i) for i in range(X_curr.shape[1])]
            except Exception:
                break
            max_vif = max(vifs)
            if max_vif > threshold:
                max_idx = vifs.index(max_vif)
                dropped.append(self.feature_names[current_cols[max_idx]])
                del current_cols[max_idx]
            else:
                break

        mask = np.zeros(len(self.feature_names), dtype=bool)
        mask[current_cols] = True
        selected_features = [self.feature_names[i] for i in current_cols]

        self.selection_results["VIF Filter (Collinearity)"] = {
            "mask": mask,
            "selected_features": selected_features,
            "dropped_features": dropped
        }
        return mask, selected_features


    def get_comparison_table(self) -> pd.DataFrame:
        """Generates a binary indicator table showing feature inclusion across methods."""
        data = {"Feature": self.feature_names}
        for method_name, res in self.selection_results.items():
            data[method_name] = ["Yes" if m else "No" for m in res["mask"]]
        return pd.DataFrame(data)
