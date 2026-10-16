"""
Model Evaluation and Statistical Diagnostic module.
Phase 5 of CRISP-DM: Evaluation.
Computes MSE, RMSE, MAE, R-squared, Adjusted R-squared, and tests the 5 classical regression assumptions.
"""
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor


class ModelEvaluator:
    """Evaluates regression models with standard metrics and regression assumption diagnostics."""

    @staticmethod
    def calculate_metrics(
        y_true: np.ndarray, y_pred: np.ndarray, n_features: int, model_name: str = ""
    ) -> Dict[str, float]:
        """
        Calculates MAE, MSE, RMSE, R-squared, and Adjusted R-squared.
        """
        n = len(y_true)
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)

        # Adjusted R-squared: 1 - [(1 - R2) * (n - 1) / (n - p - 1)]
        if n - n_features - 1 > 0:
            adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - n_features - 1)
        else:
            adj_r2 = r2

        return {
            "Model": model_name,
            "Features Count": n_features,
            "MAE": round(mae, 4),
            "MSE": round(mse, 4),
            "RMSE": round(rmse, 4),
            "R2": round(r2, 4),
            "Adjusted R2": round(adj_r2, 4)
        }

    @staticmethod
    def test_regression_assumptions(
        X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, feature_names: List[str]
    ) -> Dict[str, Any]:
        """
        Tests the 5 core assumptions of Multiple Linear Regression:
        1. Linearity: Pearson correlation between actual and predicted
        2. Normality of Residuals: Shapiro-Wilk and Jarque-Bera test
        3. Homoscedasticity: Breusch-Pagan test for equal variance
        4. Independence: Durbin-Watson statistic (ideal ~ 2.0)
        5. Multicollinearity: Variance Inflation Factor (VIF < 10)
        """
        residuals = y_true - y_pred

        # 1. Linearity
        corr_pearson, _ = stats.pearsonr(y_true, y_pred)

        # 2. Normality
        # Sample up to 5000 for Shapiro
        sample_residuals = residuals if len(residuals) <= 5000 else np.random.choice(residuals, 5000, replace=False)
        shapiro_stat, shapiro_p = stats.shapiro(sample_residuals)
        jb_stat, jb_p = stats.jarque_bera(residuals)
        residual_skew = stats.skew(residuals)
        residual_kurt = stats.kurtosis(residuals)

        # 3. Homoscedasticity (Breusch-Pagan)
        X_with_const = sm.add_constant(X)
        try:
            bp_test = het_breuschpagan(residuals, X_with_const)
            bp_stat, bp_p = bp_test[0], bp_test[1]
        except Exception:
            bp_stat, bp_p = np.nan, np.nan

        # 4. Independence (Durbin-Watson)
        dw_stat = durbin_watson(residuals)

        # 5. Multicollinearity (VIF)
        vifs = []
        for i in range(X.shape[1]):
            try:
                v = variance_inflation_factor(X, i)
            except Exception:
                v = np.nan
            vifs.append(v)

        vif_df = pd.DataFrame({
            "Feature": feature_names,
            "VIF": [round(v, 2) for v in vifs]
        }).sort_values(by="VIF", ascending=False)

        return {
            "Linearity_Pearson_r": round(corr_pearson, 4),
            "Normality_Shapiro_p": round(shapiro_p, 4),
            "Normality_JarqueBera_p": round(jb_p, 4),
            "Residual_Skewness": round(residual_skew, 4),
            "Residual_Kurtosis": round(residual_kurt, 4),
            "Homoscedasticity_BreuschPagan_p": round(bp_p, 4),
            "Independence_DurbinWatson": round(dw_stat, 4),
            "Max_VIF": round(np.nanmax(vifs), 2),
            "Mean_VIF": round(np.nanmean(vifs), 2),
            "VIF_Table": vif_df
        }
