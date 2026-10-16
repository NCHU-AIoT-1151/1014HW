"""
Main pipeline execution script.
Executes the complete CRISP-DM workflow for Multiple Linear Regression on WHO Life Expectancy Data.
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import DataLoader
from src.preprocessor import DataPreprocessor
from src.feature_selection import FeatureSelector
from src.models import RegressionTrainer
from src.evaluation import ModelEvaluator
from src.visualization import Visualizer
from src.deployment import ModelDeployer, LifeExpectancyPredictor


def run_pipeline():
    print("=" * 80)
    print("CRISP-DM PIPELINE: WHO LIFE EXPECTANCY MULTIPLE LINEAR REGRESSION")
    print("=" * 80)

    # ---------------------------------------------------------
    # PHASE 1: BUSINESS UNDERSTANDING
    # ---------------------------------------------------------
    print("\n[PHASE 1: Business Understanding]")
    print("- Project Goal: Accurately predict national Life Expectancy using public health,")
    print("  economic, immunization, and demographic indicators from WHO.")
    print("- Business Value: Guide international health organizations and policymakers to identify")
    print("  critical leverage points (e.g. schooling, immunization) to maximize longevity.")
    print("- Selected Dataset: Kaggle 'Life Expectancy (WHO)' (20 candidate features, continuous target).")

    # ---------------------------------------------------------
    # PHASE 2: DATA UNDERSTANDING
    # ---------------------------------------------------------
    print("\n[PHASE 2: Data Understanding]")
    loader = DataLoader()
    raw_df = loader.load_data()
    print(f"Dataset Shape: {raw_df.shape[0]} rows, {raw_df.shape[1]} columns")

    missing_df = loader.get_missing_values()
    print("\nTop 5 Columns with Missing Values:")
    print(missing_df.head(5))

    target_corr = loader.get_feature_correlations()
    print("\nTop 5 Features Positively Correlated with Life Expectancy:")
    print(target_corr.head(6))
    print("\nTop 5 Features Negatively Correlated with Life Expectancy:")
    print(target_corr.tail(5))

    # Visualizations for Data Understanding
    viz = Visualizer(output_dir="figures")
    viz.plot_target_distribution(raw_df)
    viz.plot_correlation_matrix(raw_df)
    viz.plot_feature_distributions(raw_df)
    print("Generated: figures/01_target_distribution.png, figures/02_correlation_matrix.png, figures/04_feature_distributions.png")

    # ---------------------------------------------------------
    # PHASE 3: DATA PREPARATION
    # ---------------------------------------------------------
    print("\n[PHASE 3: Data Preparation]")
    preprocessor = DataPreprocessor(test_size=0.2, random_state=42)
    cleaned_df, X_train, X_test, y_train, y_test, feature_names = preprocessor.prepare_data(raw_df)

    # Save cleaned dataset
    os.makedirs("data", exist_ok=True)
    cleaned_df.to_csv("data/processed_data.csv", index=False)
    print(f"Cleaned dataset saved to data/processed_data.csv ({cleaned_df.shape[0]} samples)")
    print(f"Training set: {X_train.shape[0]} samples | Testing set: {X_test.shape[0]} samples")
    print(f"Candidate Features ({len(feature_names)}): {feature_names}")

    viz.plot_missing_values(raw_df, cleaned_df)
    print("Generated: figures/03_missing_values_heatmap.png")

    # ---------------------------------------------------------
    # PHASE 4: FEATURE SELECTION & MODELING
    # ---------------------------------------------------------
    print("\n[PHASE 4: Modeling & Feature Selection Experiments]")
    selector = FeatureSelector(feature_names, random_state=42)

    # 1. SelectKBest (Filter)
    k_best_mask, k_best_feats = selector.select_k_best(X_train, y_train, k=12)
    print(f"\n1. SelectKBest (Filter, k=12) selected ({len(k_best_feats)}): {k_best_feats}")

    # 2. RFE (Wrapper)
    rfe_mask, rfe_feats = selector.select_rfe(X_train, y_train, n_features_to_select=12)
    print(f"2. RFE (Wrapper, n=12) selected ({len(rfe_feats)}): {rfe_feats}")

    # 3. LassoCV (Embedded)
    lasso_mask, lasso_feats = selector.select_lasso(X_train, y_train)
    print(f"3. Lasso (Embedded) selected ({len(lasso_feats)}): {lasso_feats}")

    # 4. VIF Filter (Multicollinearity Filter)
    vif_mask, vif_feats = selector.select_vif_filter(X_train, threshold=10.0)
    print(f"4. VIF Filter selected ({len(vif_feats)}): {vif_feats}")

    comp_table = selector.get_comparison_table()
    viz.plot_feature_selection_comparison(comp_table)
    print("Generated: figures/05_feature_selection_comparison.png")

    # Train Models across feature subsets
    trainer = RegressionTrainer(random_state=42)

    # Model 1: Baseline Linear Regression (All 20 Features)
    lr_baseline = LinearRegression()
    lr_baseline.fit(X_train, y_train)

    # Model 2: Linear Regression with SelectKBest
    lr_kbest = LinearRegression()
    lr_kbest.fit(X_train[:, k_best_mask], y_train)

    # Model 3: Linear Regression with RFE
    lr_rfe = LinearRegression()
    lr_rfe.fit(X_train[:, rfe_mask], y_train)

    # Model 4: Linear Regression with Lasso-selected features
    lr_lasso_feats = LinearRegression()
    lr_lasso_feats.fit(X_train[:, lasso_mask], y_train)

    # Model 5: Linear Regression with VIF-filtered features
    lr_vif = LinearRegression()
    lr_vif.fit(X_train[:, vif_mask], y_train)

    # Model 6: Ridge Regression (All Features)
    ridge_model = trainer.train_ridge(X_train, y_train, "Ridge Regression")

    # Model 7: Lasso Regression (All Features)
    lasso_model = trainer.train_lasso(X_train, y_train, "Lasso Regression")

    # Model 8: ElasticNet (All Features)
    elastic_model = trainer.train_elastic_net(X_train, y_train, "ElasticNet")

    # ---------------------------------------------------------
    # PHASE 5: EVALUATION
    # ---------------------------------------------------------
    print("\n[PHASE 5: Evaluation]")
    models_to_evaluate = [
        ("Baseline OLS (All 20 Feats)", lr_baseline, np.ones(len(feature_names), dtype=bool)),
        ("SelectKBest + OLS (12 Feats)", lr_kbest, k_best_mask),
        ("RFE + OLS (12 Feats)", lr_rfe, rfe_mask),
        ("Lasso Selected + OLS", lr_lasso_feats, lasso_mask),
        ("VIF Filter + OLS", lr_vif, vif_mask),
        ("Ridge Regression (L2)", ridge_model, np.ones(len(feature_names), dtype=bool)),
        ("Lasso Regression (L1)", lasso_model, np.ones(len(feature_names), dtype=bool)),
        ("ElasticNet Regression", elastic_model, np.ones(len(feature_names), dtype=bool)),
    ]

    metrics_list = []
    predictions_dict = {}

    for name, model, mask in models_to_evaluate:
        X_test_sub = X_test[:, mask]
        y_pred = model.predict(X_test_sub)
        predictions_dict[name] = y_pred
        n_feat = int(np.sum(mask))
        metric = ModelEvaluator.calculate_metrics(y_test, y_pred, n_feat, model_name=name)
        metrics_list.append(metric)

    metrics_df = pd.DataFrame(metrics_list)
    print("\nModel Performance Evaluation Table on Test Set:")
    print(metrics_df.to_string(index=False))

    viz.plot_metrics_comparison(metrics_df)
    print("\nGenerated: figures/06_model_metrics_comparison.png")

    # Visualizing Actual vs Predicted
    best_fs_name = "RFE + OLS (12 Feats)"
    viz.plot_actual_vs_predicted(
        y_test,
        predictions_dict["Baseline OLS (All 20 Feats)"],
        predictions_dict[best_fs_name],
        base_name="Baseline OLS (20 Feats)",
        best_name=best_fs_name
    )
    print("Generated: figures/07_actual_vs_predicted.png")

    # Test the 5 Regression Assumptions on Best Model (RFE + OLS)
    best_model = lr_rfe
    best_mask = rfe_mask
    best_pred = predictions_dict[best_fs_name]
    best_feat_names = [f for f, m in zip(feature_names, best_mask) if m]

    assumptions = ModelEvaluator.test_regression_assumptions(
        X_test[:, best_mask], y_test, best_pred, best_feat_names
    )

    print("\n--- Regression Assumptions Diagnostic Report (Best Feature-Selected Model) ---")
    print(f"1. Linearity (Pearson r): {assumptions['Linearity_Pearson_r']} (Target: close to 1.0)")
    print(f"2. Normality of Residuals:")
    print(f"   - Shapiro-Wilk p-value: {assumptions['Normality_Shapiro_p']}")
    print(f"   - Jarque-Bera p-value: {assumptions['Normality_JarqueBera_p']}")
    print(f"   - Residual Skewness: {assumptions['Residual_Skewness']}, Kurtosis: {assumptions['Residual_Kurtosis']}")
    print(f"3. Homoscedasticity (Breusch-Pagan p-value): {assumptions['Homoscedasticity_BreuschPagan_p']}")
    print(f"4. Independence of Errors (Durbin-Watson): {assumptions['Independence_DurbinWatson']} (Target: ~2.0)")
    print(f"5. Multicollinearity (VIF):")
    print(f"   - Max VIF: {assumptions['Max_VIF']} (Target: < 5-10)")
    print(f"   - Mean VIF: {assumptions['Mean_VIF']}")
    print("\nVIF Table for Selected Features:")
    print(assumptions["VIF_Table"].to_string(index=False))

    viz.plot_residuals_analysis(y_test, best_pred, model_name=best_fs_name)
    viz.plot_feature_importance(best_feat_names, best_model.coef_, model_name=best_fs_name)
    print("Generated: figures/08_residuals_analysis.png, figures/09_feature_importance.png")

    # ---------------------------------------------------------
    # PHASE 6: DEPLOYMENT
    # ---------------------------------------------------------
    print("\n[PHASE 6: Deployment & Scenario Simulation]")
    deployer = ModelDeployer(models_dir="models")
    pipeline_path = deployer.save_pipeline(
        model=best_model,
        scaler=preprocessor.get_scaler(),
        all_features=feature_names,
        selected_mask=best_mask,
        imputation_values=preprocessor.imputation_values_,
        filename="best_linear_regression_pipeline.joblib"
    )

    # Demonstration of Inference Engine
    predictor = LifeExpectancyPredictor(pipeline_path)

    sample_country = {
        "Status": "Developing",
        "Adult Mortality": 263.0,
        "infant deaths": 62,
        "Alcohol": 0.01,
        "percentage expenditure": 71.28,
        "Hepatitis B": 65.0,
        "Measles": 1154,
        "BMI": 19.1,
        "under-five deaths": 83,
        "Polio": 6.0,
        "Total expenditure": 8.16,
        "Diphtheria": 65.0,
        "HIV/AIDS": 0.1,
        "GDP": 584.26,
        "Population": 33736494.0,
        "thinness  1-19 years": 17.2,
        "thinness 5-9 years": 17.3,
        "Income composition of resources": 0.479,
        "Schooling": 10.1,
        "Year": 2015
    }

    base_pred = predictor.predict_sample(sample_country)
    print(f"\nPredicted Baseline Life Expectancy for sample country: {base_pred:.2f} years")

    # Policy Intervention Simulation:
    # 1. Schooling +3 years
    # 2. Diphtheria immunization +25%
    # 3. Adult Mortality -50 per 1000
    interventions = {
        "Schooling": 3.0,
        "Diphtheria": 25.0,
        "Adult Mortality": -50.0
    }
    sim_result = predictor.simulate_policy_scenario(sample_country, interventions)
    print("\nPolicy Intervention Simulation Result:")
    for k, v in sim_result.items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 80)
    print("CRISP-DM PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    return metrics_df, assumptions


if __name__ == "__main__":
    run_pipeline()
