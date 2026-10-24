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
    """Preprocesses datasets for Multiple Linear Regression."""

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
        and perform hierarchical median imputation for Life Expectancy.
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
        """Backwards-compatible wrapper for Life Expectancy preparation."""
        return self.prepare_life_expectancy(df)

    def prepare_life_expectancy(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """Full data preparation pipeline for WHO Life Expectancy (20 features)."""
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

    def prepare_steel_industry(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Data preparation for Steel Industry Energy Consumption.
        Produces exactly 14 engineered features (matching Kaggle/homework spec >13 features).
        Target: Usage_kWh
        """
        data = df.copy()
        data.columns = data.columns.str.strip()

        # Parse date
        dt_col = pd.to_datetime(data["date"], format="%d/%m/%Y %H:%M")
        data["Month"] = dt_col.dt.month
        data["Hour"] = dt_col.dt.hour
        data["Day"] = dt_col.dt.day
        data["Is_Weekday"] = (data["WeekStatus"].str.strip() == "Weekday").astype(int)
        data["Peak_Hour"] = dt_col.dt.hour.isin(range(9, 22)).astype(int)
        data["Day_Weekend"] = data["Day_of_week"].str.strip().isin(["Saturday", "Sunday"]).astype(int)

        # One-hot load type
        load_dummies = pd.get_dummies(data["Load_Type"].str.strip(), prefix="Load", drop_first=True, dtype=int)
        for col in ["Load_Maximum_Load", "Load_Medium_Load"]:
            if col in load_dummies.columns:
                data[col] = load_dummies[col]
            else:
                data[col] = 0

        # Exactly 14 clean candidate features (>13 features as required)
        candidate_features = [
            "Lagging_Current_Reactive.Power_kVarh",
            "Leading_Current_Reactive_Power_kVarh",
            "CO2(tCO2)",
            "Lagging_Current_Power_Factor",
            "Leading_Current_Power_Factor",
            "NSM",
            "Month",
            "Hour",
            "Day",
            "Is_Weekday",
            "Load_Maximum_Load",
            "Load_Medium_Load",
            "Day_Weekend",
            "Peak_Hour"
        ]

        # Handle any possible missing values
        for f in candidate_features:
            if f in data.columns:
                med = data[f].median()
                data[f] = data[f].fillna(med)
                self.imputation_values_[f] = med

        X_df = data[candidate_features]
        y_series = data["Usage_kWh"]
        self.feature_names = candidate_features

        # Split 80/20
        X_train_df, X_test_df, y_train, y_test = train_test_split(
            X_df,
            y_series.values,
            test_size=self.test_size,
            random_state=self.random_state
        )

        X_train_scaled = self.scaler.fit_transform(X_train_df)
        X_test_scaled = self.scaler.transform(X_test_df)

        return data, X_train_scaled, X_test_scaled, y_train, y_test, self.feature_names

    def prepare_arbitrary(
        self, df: pd.DataFrame, target_col: str = None
    ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], str]:
        """
        General automatic preprocessing pipeline for ANY arbitrary dataset (e.g. 50_Startups).
        - Automatically detects target if not specified.
        - Imputes missing values.
        - One-hot encodes categorical variables.
        - Scales features.
        - Train/Test split.
        """
        data = df.copy()
        data.columns = data.columns.str.strip()

        # Identify target
        if target_col is None or target_col not in data.columns:
            # Common default regression targets
            for likely_target in ["Profit", "price", "Price", "target", "Target", "label", "Usage_kWh", "Life expectancy"]:
                if likely_target in data.columns and np.issubdtype(data[likely_target].dtype, np.number):
                    target_col = likely_target
                    break
            if target_col is None:
                numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
                if not numeric_cols:
                    raise ValueError("No numeric columns found in dataset to use as regression target.")
                target_col = numeric_cols[-1]

        print(f"Target variable selected: '{target_col}'")

        # Drop rows where target is NaN
        data = data.dropna(subset=[target_col]).reset_index(drop=True)

        # Drop pure identifier columns (e.g. ID, name, high cardinality text)
        drop_cols = [target_col]
        for col in data.columns:
            if col != target_col and data[col].dtype == object:
                # If unique strings equal number of rows, likely ID
                if data[col].nunique() > 0.8 * len(data) and len(data) > 50:
                    drop_cols.append(col)

        feature_df = data.drop(columns=drop_cols)

        # Impute numeric medians and categorical modes
        for col in feature_df.columns:
            if np.issubdtype(feature_df[col].dtype, np.number):
                med = feature_df[col].median()
                if pd.isna(med):
                    med = 0.0
                feature_df[col] = feature_df[col].fillna(med)
                self.imputation_values_[col] = med
            else:
                mode_vals = feature_df[col].mode()
                mode_val = mode_vals.iloc[0] if not mode_vals.empty else "Missing"
                feature_df[col] = feature_df[col].fillna(mode_val)

        # One-hot encode categoricals
        X_df = pd.get_dummies(feature_df, drop_first=True, dtype=float)
        # Ensure all columns are numeric
        X_df = X_df.astype(float)
        y_series = data[target_col].astype(float)

        self.feature_names = X_df.columns.tolist()

        # Train / Test split
        X_train_df, X_test_df, y_train, y_test = train_test_split(
            X_df,
            y_series.values,
            test_size=self.test_size,
            random_state=self.random_state
        )

        X_train_scaled = self.scaler.fit_transform(X_train_df)
        X_test_scaled = self.scaler.transform(X_test_df)

        return data, X_train_scaled, X_test_scaled, y_train, y_test, self.feature_names, target_col

    def get_scaler(self) -> StandardScaler:
        return self.scaler

