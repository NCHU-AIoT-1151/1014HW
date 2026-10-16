"""
Data Loader and Exploratory Data Analysis module.
Phase 2 of CRISP-DM: Data Understanding.
"""
import os
import pandas as pd
import numpy as np


class DataLoader:
    """Handles loading and initial data inspection for WHO Life Expectancy dataset."""

    DEFAULT_DATA_PATH = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data",
        "Life_Expectancy_Data.csv"
    )

    DATA_URL = "https://raw.githubusercontent.com/EvidenceN/Life-Expectancy-Prediction/master/Life%20Expectancy/Data/Life%20Expectancy%20Data.csv"

    def __init__(self, data_path: str = None):
        self.data_path = data_path or self.DEFAULT_DATA_PATH
        self.df = None

    def load_data(self) -> pd.DataFrame:
        """Load data from local file or fallback to GitHub URL if not found."""
        if not os.path.exists(self.data_path):
            print(f"Local file {self.data_path} not found. Downloading from source...")
            os.makedirs(os.path.dirname(self.data_path), exist_ok=True)
            self.df = pd.read_csv(self.DATA_URL)
            self.df.to_csv(self.data_path, index=False)
            print(f"Saved dataset to {self.data_path}")
        else:
            self.df = pd.read_csv(self.data_path)

        # Standardize column names (strip whitespace)
        self.df.columns = self.df.columns.str.strip()
        return self.df

    def get_summary_statistics(self) -> pd.DataFrame:
        """Return numerical summary statistics."""
        if self.df is None:
            self.load_data()
        return self.df.describe().T

    def get_missing_values(self) -> pd.DataFrame:
        """Return missing values count and percentage."""
        if self.df is None:
            self.load_data()
        missing_count = self.df.isnull().sum()
        missing_pct = (missing_count / len(self.df)) * 100
        missing_df = pd.DataFrame({
            "Missing Count": missing_count,
            "Percentage (%)": missing_pct.round(2)
        })
        return missing_df[missing_df["Missing Count"] > 0].sort_values(
            by="Missing Count", ascending=False
        )

    def get_feature_correlations(self, target: str = "Life expectancy") -> pd.Series:
        """Compute Pearson correlation of numerical features with the target."""
        if self.df is None:
            self.load_data()
        numeric_df = self.df.select_dtypes(include=[np.number])
        if target not in numeric_df.columns:
            raise ValueError(f"Target '{target}' not found in numeric columns.")
        return numeric_df.corr()[target].sort_values(ascending=False)
