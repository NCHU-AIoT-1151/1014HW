"""
Data Preprocessing module.
Phase 3 of CRISP-DM: Data Preparation.
Handles missing values, categorical encoding, train/test splitting, and feature scaling.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Dict


class DataPreprocessor:
    """Preprocesses WHO Life Expectancy dataset for Multiple Linear Regression."""

    TARGET_COL = "Life expectancy"
    ID_COLS = ["Country"]
    CATEGORICAL_COLS = ["Status"]

    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self.status_mapping = {"Developing": 0, "Developed": 1}
        self.imputation_values_: Dict[str, float] = {}

    def clean_and_impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean whitespace in column names, drop rows without target,
        and perform hierarchical median imputation:
        1) Country-specific median across years
        2) Development Status median
        3) Global median
        """
        data = df.copy()
        data.columns = data.columns.str.strip()

        # Drop rows where target is missing
        data = data.dropna(subset=[self.TARGET_COL]).reset_index(drop=True)

        # Encode Status
        if "Status" in data.columns and data["Status"].dtype == object:
            data["Status"] = data["Status"].map(self.status_mapping).fillna(0).astype(int)

        # Numerical columns for imputation
        num_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        num_cols = [c for c in num_cols if c != self.TARGET_COL]

        # 1) Impute by Country median
        for col in num_cols:
            if data[col].isnull().any():
                country_medians = data.groupby("Country")[col].transform("median")
                data[col] = data[col].fillna(country_medians)

        # 2) Impute remaining nulls by Status median
        for col in num_cols:
            if data[col].isnull().any():
                status_medians = data.groupby("Status")[col].transform("median")
                data[col] = data[col].fillna(status_medians)

        # 3) Fallback to global median
        for col in num_cols:
            if data[col].isnull().any():
                global_med = data[col].median()
                data[col] = data[col].fillna(global_med)
                self.imputation_values_[col] = global_med
            else:
                self.imputation_values_[col] = data[col].median()

        return data

    def prepare_data(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Full data preparation pipeline:
        - Hierarchical imputation
        - Drop Country ID column
        - Train/Test Split (80/20)
        - Standardization with StandardScaler (fitted only on Train)
        """
        cleaned_df = self.clean_and_impute(df)

        # Feature selection: drop Country and target
        X_df = cleaned_df.drop(columns=[self.TARGET_COL] + [c for c in self.ID_COLS if c in cleaned_df.columns])
        y_series = cleaned_df[self.TARGET_COL]

        self.feature_names = X_df.columns.tolist()

        # Train / Test split
        X_train_df, X_test_df, y_train, y_test = train_test_split(
            X_df,
            y_series.values,
            test_size=self.test_size,
            random_state=self.random_state
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train_df)
        X_test_scaled = self.scaler.transform(X_test_df)

        return cleaned_df, X_train_scaled, X_test_scaled, y_train, y_test, self.feature_names

    def get_scaler(self) -> StandardScaler:
        return self.scaler
