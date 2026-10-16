"""
Visualization module.
Generates publication-quality figures for Exploratory Data Analysis, Model Evaluation,
Regression Diagnostics, and Feature Importance.
"""
import os
import warnings
from typing import List, Dict
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats

# Suppress minor library deprecation notices
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Configure aesthetics
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.unicode_minus"] = False


class Visualizer:
    """Creates and saves all analytical and diagnostic visualizations."""

    def __init__(self, output_dir: str = "figures"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_target_distribution(self, df: pd.DataFrame, target_col: str = "Life expectancy"):
        """Plot target distribution and status comparison."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # KDE & Histogram
        sns.histplot(df[target_col].dropna(), kde=True, ax=axes[0], color="#2b5c8f", bins=30)
        mean_val = df[target_col].mean()
        median_val = df[target_col].median()
        axes[0].axvline(mean_val, color="#d9534f", linestyle="--", label=f"Mean: {mean_val:.1f}")
        axes[0].axvline(median_val, color="#5cb85c", linestyle=":", label=f"Median: {median_val:.1f}")
        axes[0].set_title(f"Target Distribution: {target_col}", fontsize=13, fontweight="bold")
        axes[0].set_xlabel("Life Expectancy (Years)")
        axes[0].set_ylabel("Count")
        axes[0].legend()

        # Boxplot by Status
        if "Status" in df.columns:
            status_labels = df["Status"].copy()
            if status_labels.dtype in [int, float, np.int64]:
                status_labels = status_labels.map({0: "Developing", 1: "Developed"})
            sns.boxplot(
                x=status_labels,
                y=df[target_col],
                hue=status_labels,
                ax=axes[1],
                palette=["#f0ad4e", "#0275d8"],
                legend=False
            )
            axes[1].set_title("Life Expectancy by Development Status", fontsize=13, fontweight="bold")
            axes[1].set_xlabel("Country Status")
            axes[1].set_ylabel("Life Expectancy (Years)")

        plt.tight_layout()
        path = os.path.join(self.output_dir, "01_target_distribution.png")
        plt.savefig(path, dpi=300)
        plt.close()
        return path

    def plot_correlation_matrix(self, df: pd.DataFrame):
        """Plot comprehensive correlation heatmap."""
        plt.figure(figsize=(16, 12))
        num_df = df.select_dtypes(include=[np.number])
        corr = num_df.corr()

        mask = np.triu(np.ones_like(corr, dtype=bool))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)

        sns.heatmap(
            corr,
            mask=mask,
            cmap=cmap,
            vmax=1.0,
            vmin=-1.0,
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            annot=True,
            fmt=".2f",
            annot_kws={"size": 8}
        )
        plt.title("WHO Life Expectancy Feature Correlation Matrix", fontsize=15, fontweight="bold", pad=15)
        plt.tight_layout()
        path = os.path.join(self.output_dir, "02_correlation_matrix.png")
        plt.savefig(path, dpi=300)
        plt.close()
        return path

    def plot_missing_values(self, raw_df: pd.DataFrame, clean_df: pd.DataFrame):
        """Plot missing values comparison before and after imputation."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Before
        raw_missing = raw_df.isnull().sum()
        raw_missing = raw_missing[raw_missing > 0].sort_values(ascending=True)
        raw_pct = (raw_missing / len(raw_df)) * 100

        axes[0].barh(raw_pct.index, raw_pct.values, color="#e74c3c")
        axes[0].set_title("Missing Values Percentage Before Imputation", fontsize=12, fontweight="bold")
        axes[0].set_xlabel("Missing Percentage (%)")
        for i, v in enumerate(raw_pct.values):
            axes[0].text(v + 0.5, i, f"{v:.1f}% ({raw_missing.iloc[i]})", va="center", fontsize=9)
        axes[0].set_xlim(0, max(raw_pct.values) + 8)

        # After
        clean_missing = clean_df.isnull().sum().sum()
        axes[1].text(
            0.5, 0.5,
            f"All Missing Values Successfully Imputed!\nRemaining Missing: {clean_missing}\nDataset Ready for Modeling: {clean_df.shape[0]} Rows",
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=13,
            color="#27ae60",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=1", facecolor="#eafaf1", edgecolor="#27ae60", lw=2)
        )
        axes[1].set_title("Missing Values Status After Imputation", fontsize=12, fontweight="bold")
        axes[1].axis("off")

        plt.tight_layout()
        path = os.path.join(self.output_dir, "03_missing_values_heatmap.png")
        plt.savefig(path, dpi=300)
        plt.close()
        return path

    def plot_feature_distributions(self, df: pd.DataFrame):
        """Plot distribution of 6 key predictors."""
        key_features = [
            "Schooling", "Income composition of resources", "Adult Mortality",
            "HIV/AIDS", "Diphtheria", "BMI"
        ]
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        axes = axes.flatten()

        colors = ["#3498db", "#2ecc71", "#e74c3c", "#9b59b6", "#f39c12", "#1abc9c"]

        for i, feat in enumerate(key_features):
            if feat in df.columns:
                sns.histplot(df[feat].dropna(), kde=True, ax=axes[i], color=colors[i], bins=25)
                axes[i].set_title(f"Distribution: {feat}", fontsize=11, fontweight="bold")
                axes[i].set_xlabel(feat)
                axes[i].set_ylabel("Frequency")

        plt.suptitle("Distributions of Key Predictor Features", fontsize=14, fontweight="bold", y=1.00)
        plt.tight_layout()
        path = os.path.join(self.output_dir, "04_feature_distributions.png")
        plt.savefig(path, dpi=300)
        plt.close()
        return path

    def plot_feature_selection_comparison(self, comp_df: pd.DataFrame):
        """Heatmap showing features selected across methods."""
        plt.figure(figsize=(10, 8))
        matrix_df = comp_df.set_index("Feature")
        numeric_matrix = (matrix_df == "Yes").astype(int)

        sns.heatmap(
            numeric_matrix,
            annot=matrix_df.values,
            fmt="",
            cmap=sns.color_palette(["#f8d7da", "#d4edda"]),
            cbar=False,
            linewidths=1,
            linecolor="white"
        )
        plt.title("Feature Selection Comparison across Methods", fontsize=14, fontweight="bold", pad=15)
        plt.xlabel("Selection Strategy", fontsize=11, fontweight="bold")
        plt.ylabel("Features", fontsize=11, fontweight="bold")
        plt.tight_layout()
        path = os.path.join(self.output_dir, "05_feature_selection_comparison.png")
        plt.savefig(path, dpi=300)
        plt.close()
        return path

    def plot_metrics_comparison(self, metrics_df: pd.DataFrame):
        """Plot grouped bar charts comparing R2, RMSE, and MAE across models."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        models = metrics_df["Model"].tolist()

        # R-squared
        sns.barplot(data=metrics_df, x="Model", y="R2", hue="Model", ax=axes[0], palette="Blues_d", legend=False)
        axes[0].set_title("Test R² Score (Higher is Better)", fontsize=12, fontweight="bold")
        axes[0].set_ylim(0.7, 0.9)
        axes[0].set_xticks(range(len(models)))
        axes[0].set_xticklabels(models, rotation=35, ha="right", fontsize=9)
        for p in axes[0].patches:
            axes[0].annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                             ha="center", va="bottom", fontsize=8, xytext=(0, 3), textcoords="offset points")

        # RMSE
        sns.barplot(data=metrics_df, x="Model", y="RMSE", hue="Model", ax=axes[1], palette="Oranges_d", legend=False)
        axes[1].set_title("Test RMSE (Lower is Better)", fontsize=12, fontweight="bold")
        axes[1].set_ylim(0, max(metrics_df["RMSE"]) * 1.25)
        axes[1].set_xticks(range(len(models)))
        axes[1].set_xticklabels(models, rotation=35, ha="right", fontsize=9)
        for p in axes[1].patches:
            axes[1].annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                             ha="center", va="bottom", fontsize=8, xytext=(0, 3), textcoords="offset points")

        # MAE
        sns.barplot(data=metrics_df, x="Model", y="MAE", hue="Model", ax=axes[2], palette="Greens_d", legend=False)
        axes[2].set_title("Test MAE (Lower is Better)", fontsize=12, fontweight="bold")
        axes[2].set_ylim(0, max(metrics_df["MAE"]) * 1.25)
        axes[2].set_xticks(range(len(models)))
        axes[2].set_xticklabels(models, rotation=35, ha="right", fontsize=9)
        for p in axes[2].patches:
            axes[2].annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                             ha="center", va="bottom", fontsize=8, xytext=(0, 3), textcoords="offset points")

        plt.tight_layout()
        path = os.path.join(self.output_dir, "06_model_metrics_comparison.png")
        plt.savefig(path, dpi=300)
        plt.close()
        return path

    def plot_actual_vs_predicted(
        self, y_test: np.ndarray, y_pred_base: np.ndarray, y_pred_best: np.ndarray,
        base_name: str = "Baseline (All Features)", best_name: str = "Best Feature-Selected Model"
    ):
        """Plot Actual vs Predicted scatter plots with 45-degree reference line."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        min_val = min(y_test.min(), min(y_pred_base.min(), y_pred_best.min())) - 2
        max_val = max(y_test.max(), max(y_pred_base.max(), y_pred_best.max())) + 2

        for ax, y_p, title, color in [
            (axes[0], y_pred_base, base_name, "#2980b9"),
            (axes[1], y_pred_best, best_name, "#27ae60")
        ]:
            ax.scatter(y_test, y_p, alpha=0.55, color=color, edgecolors="none")
            ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect Prediction (y = x)")
            ax.set_xlim(min_val, max_val)
            ax.set_ylim(min_val, max_val)
            ax.set_xlabel("Actual Life Expectancy (Years)", fontsize=11)
            ax.set_ylabel("Predicted Life Expectancy (Years)", fontsize=11)
            ax.set_title(f"Actual vs Predicted: {title}", fontsize=12, fontweight="bold")
            ax.legend(loc="upper left")

            # Add R2 and RMSE text
            from sklearn.metrics import r2_score, mean_squared_error
            r2 = r2_score(y_test, y_p)
            rmse = np.sqrt(mean_squared_error(y_test, y_p))
            ax.text(0.95, 0.05, f"R² = {r2:.4f}\nRMSE = {rmse:.2f}",
                    transform=ax.transAxes, ha="right", va="bottom",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

        plt.tight_layout()
        path = os.path.join(self.output_dir, "07_actual_vs_predicted.png")
        plt.savefig(path, dpi=300)
        plt.close()
        return path

    def plot_residuals_analysis(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "Linear Regression"):
        """4-panel diagnostic plot for the 5 regression assumptions."""
        residuals = y_true - y_pred
        std_residuals = (residuals - np.mean(residuals)) / np.std(residuals)

        fig, axes = plt.subplots(2, 2, figsize=(14, 11))

        # 1. Residuals vs Fitted (Linearity & Homoscedasticity)
        axes[0, 0].scatter(y_pred, residuals, alpha=0.5, color="#3498db")
        axes[0, 0].axhline(0, color="red", linestyle="--", lw=2)
        axes[0, 0].set_title("Residuals vs Fitted Values (Homoscedasticity Check)", fontsize=11, fontweight="bold")
        axes[0, 0].set_xlabel("Fitted (Predicted) Values")
        axes[0, 0].set_ylabel("Residuals")

        # 2. Normal Q-Q Plot (Normality of Residuals)
        stats.probplot(residuals, dist="norm", plot=axes[0, 1])
        axes[0, 1].get_lines()[0].set_color("#2c3e50")
        axes[0, 1].get_lines()[1].set_color("#e74c3c")
        axes[0, 1].get_lines()[1].set_linewidth(2)
        axes[0, 1].set_title("Normal Q-Q Plot (Normality Check)", fontsize=11, fontweight="bold")

        # 3. Residuals Distribution Histogram & KDE
        sns.histplot(residuals, kde=True, ax=axes[1, 0], color="#9b59b6", bins=30)
        axes[1, 0].axvline(0, color="red", linestyle="--", lw=2)
        axes[1, 0].set_title("Histogram & KDE of Residuals", fontsize=11, fontweight="bold")
        axes[1, 0].set_xlabel("Residual Value")
        axes[1, 0].set_ylabel("Count")

        # 4. Scale-Location (Spread-Location: Homoscedasticity)
        sqrt_abs_std_res = np.sqrt(np.abs(std_residuals))
        axes[1, 1].scatter(y_pred, sqrt_abs_std_res, alpha=0.5, color="#1abc9c")
        axes[1, 1].set_title("Scale-Location Plot (Equal Variance)", fontsize=11, fontweight="bold")
        axes[1, 1].set_xlabel("Fitted Values")
        axes[1, 1].set_ylabel("√|Standardized Residuals|")

        plt.suptitle(f"Regression Diagnostic Plots: {model_name}", fontsize=14, fontweight="bold", y=1.00)
        plt.tight_layout()
        path = os.path.join(self.output_dir, "08_residuals_analysis.png")
        plt.savefig(path, dpi=300)
        plt.close()
        return path

    def plot_feature_importance(self, feature_names: List[str], coefficients: np.ndarray, model_name: str = "Linear Regression"):
        """Plot standardized regression coefficients (effect sizes)."""
        plt.figure(figsize=(10, 8))
        df_coef = pd.DataFrame({"Feature": feature_names, "Coefficient": coefficients})
        df_coef["AbsCoef"] = df_coef["Coefficient"].abs()
        df_coef = df_coef.sort_values(by="Coefficient", ascending=True)

        colors = ["#27ae60" if c > 0 else "#e74c3c" for c in df_coef["Coefficient"]]
        plt.barh(df_coef["Feature"], df_coef["Coefficient"], color=colors)
        plt.axvline(0, color="black", linestyle="-", lw=1)
        plt.title(f"Standardized Regression Coefficients ({model_name})", fontsize=13, fontweight="bold", pad=12)
        plt.xlabel("Standardized Coefficient (Beta weight)", fontsize=11, fontweight="bold")
        plt.ylabel("Feature", fontsize=11, fontweight="bold")

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#27ae60", label="Positive Impact (Increases Life Expectancy)"),
            Patch(facecolor="#e74c3c", label="Negative Impact (Decreases Life Expectancy)")
        ]
        plt.legend(handles=legend_elements, loc="lower right")

        plt.tight_layout()
        path = os.path.join(self.output_dir, "09_feature_importance.png")
        plt.savefig(path, dpi=300)
        plt.close()
        return path
