"""
Deployment module.
Phase 6 of CRISP-DM: Deployment.
Saves model artifacts and provides an inference API and scenario simulation engine.
"""
import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, List


class ModelDeployer:
    """Handles serialization and deployment inference for Life Expectancy prediction."""

    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        os.makedirs(self.models_dir, exist_ok=True)

    def save_pipeline(
        self,
        model: Any,
        scaler: Any,
        all_features: List[str],
        selected_mask: np.ndarray,
        imputation_values: Dict[str, float],
        filename: str = "best_linear_regression_pipeline.joblib"
    ) -> str:
        """Serializes the end-to-end model artifacts."""
        filepath = os.path.join(self.models_dir, filename)
        pipeline_artifact = {
            "model": model,
            "scaler": scaler,
            "all_features": all_features,
            "selected_mask": selected_mask,
            "selected_features": [f for f, m in zip(all_features, selected_mask) if m],
            "imputation_values": imputation_values,
            "status_mapping": {"Developing": 0, "Developed": 1}
        }
        joblib.dump(pipeline_artifact, filepath)
        print(f"Pipeline artifact saved to {filepath}")
        return filepath

    @staticmethod
    def load_pipeline(filepath: str) -> Dict[str, Any]:
        """Loads serialized pipeline artifact."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Pipeline file {filepath} not found.")
        return joblib.load(filepath)


class LifeExpectancyPredictor:
    """Production inference engine for predicting life expectancy on new country profiles."""

    def __init__(self, pipeline_path: str):
        self.artifact = ModelDeployer.load_pipeline(pipeline_path)
        self.model = self.artifact["model"]
        self.scaler = self.artifact["scaler"]
        self.all_features = self.artifact["all_features"]
        self.selected_mask = self.artifact["selected_mask"]
        self.imputation_values = self.artifact["imputation_values"]
        self.status_mapping = self.artifact["status_mapping"]

    def predict_sample(self, raw_data: Dict[str, Any]) -> float:
        """
        Takes raw dictionary with feature names and values,
        imputes missing fields, encodes Status, scales features, applies mask,
        and returns predicted life expectancy in years.
        """
        # Fill missing with median
        row = []
        for feat in self.all_features:
            if feat == "Status":
                val = raw_data.get("Status", 0)
                if isinstance(val, str):
                    val = self.status_mapping.get(val, 0)
                row.append(val)
            else:
                val = raw_data.get(feat, self.imputation_values.get(feat, 0.0))
                row.append(float(val))

        X_df = pd.DataFrame([row], columns=self.all_features)
        X_scaled = self.scaler.transform(X_df)
        X_selected = X_scaled[:, self.selected_mask]
        prediction = self.model.predict(X_selected)[0]
        return float(round(prediction, 2))

    def simulate_policy_scenario(
        self, base_country_data: Dict[str, Any], policy_changes: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Simulates the effect of policy interventions on life expectancy.
        Example: {'Schooling': +2.0, 'Diphtheria': +10.0}
        """
        base_pred = self.predict_sample(base_country_data)

        modified_data = base_country_data.copy()
        for k, delta in policy_changes.items():
            if k in modified_data:
                modified_data[k] = modified_data[k] + delta

        new_pred = self.predict_sample(modified_data)
        diff = round(new_pred - base_pred, 2)

        return {
            "Base Life Expectancy": base_pred,
            "Intervention Life Expectancy": new_pred,
            "Net Change (Years)": diff,
            "Interventions Applied": policy_changes
        }
