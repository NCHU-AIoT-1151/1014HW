"""
Main pipeline execution script.
Executes complete CRISP-DM workflows for Multiple Linear Regression:
- Problem 1: WHO Life Expectancy Data (20 Features, >13 features)
- Problem 2 (Bonus 1): Steel Industry Energy Consumption (14 Features, >13 features)
- Custom / Bonus 2: Arbitrary Dataset input from URL or local file (e.g. 50_Startups.csv)
"""
import os
import sys
import argparse
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
from src.deployment import ModelDeployer, LifeExpectancyPredictor, SteelEnergyPredictor


def run_life_expectancy_pipeline():
    """Runs complete CRISP-DM workflow for Problem 1: WHO Life Expectancy (20 features)."""
    print("=" * 80)
    print("CRISP-DM PIPELINE (PROBLEM 1): WHO LIFE EXPECTANCY MULTIPLE LINEAR REGRESSION")
    print("=" * 80)

    # PHASE 1: BUSINESS UNDERSTANDING
    print("\n[PHASE 1: Business Understanding]")
    print("- Project Goal: Accurately predict national Life Expectancy using public health,")
    print("  economic, immunization, and demographic indicators from WHO.")
    print("- Business Value: Guide international health organizations and policymakers to identify")
    print("  critical leverage points (e.g. schooling, immunization) to maximize longevity.")
    print("- Selected Dataset: Kaggle 'Life Expectancy (WHO)' (20 candidate features, continuous target).")

    # PHASE 2: DATA UNDERSTANDING
    print("\n[PHASE 2: Data Understanding]")
    loader = DataLoader(dataset_type="life")
    raw_df = loader.load_data()
    print(f"Dataset Shape: {raw_df.shape[0]} rows, {raw_df.shape[1]} columns")

    missing_df = loader.get_missing_values()
    print("\nTop 5 Columns with Missing Values:")
    print(missing_df.head(5))

    target_corr = loader.get_feature_correlations(target="Life expectancy")
    print("\nTop 5 Features Positively Correlated with Life Expectancy:")
    print(target_corr.head(6))
    print("\nTop 5 Features Negatively Correlated with Life Expectancy:")
    print(target_corr.tail(5))

    viz = Visualizer(output_dir="figures")
    viz.plot_target_distribution(raw_df, target_col="Life expectancy", output_name="01_target_distribution.png", unit="Years", group_col="Status")
    viz.plot_correlation_matrix(raw_df, output_name="02_correlation_matrix.png", title="WHO Life Expectancy Feature Correlation Matrix")
    viz.plot_feature_distributions(raw_df, output_name="04_feature_distributions.png")
    print("Generated: figures/01_target_distribution.png, figures/02_correlation_matrix.png, figures/04_feature_distributions.png")

    # PHASE 3: DATA PREPARATION
    print("\n[PHASE 3: Data Preparation]")
    preprocessor = DataPreprocessor(test_size=0.2, random_state=42)
    cleaned_df, X_train, X_test, y_train, y_test, feature_names = preprocessor.prepare_life_expectancy(raw_df)

    os.makedirs("data", exist_ok=True)
    cleaned_df.to_csv("data/processed_data.csv", index=False)
    print(f"Cleaned dataset saved to data/processed_data.csv ({cleaned_df.shape[0]} samples)")
    print(f"Training set: {X_train.shape[0]} samples | Testing set: {X_test.shape[0]} samples")
    print(f"Candidate Features ({len(feature_names)}): {feature_names}")

    viz.plot_missing_values(raw_df, cleaned_df, output_name="03_missing_values_heatmap.png")
    print("Generated: figures/03_missing_values_heatmap.png")

    # PHASE 4: FEATURE SELECTION & MODELING
    print("\n[PHASE 4: Modeling & Feature Selection Experiments]")
    selector = FeatureSelector(feature_names, random_state=42)

    k_best_mask, k_best_feats = selector.select_k_best(X_train, y_train, k=12)
    print(f"\n1. SelectKBest (Filter, k=12) selected ({len(k_best_feats)}): {k_best_feats}")

    rfe_mask, rfe_feats = selector.select_rfe(X_train, y_train, n_features_to_select=12)
    print(f"2. RFE (Wrapper, n=12) selected ({len(rfe_feats)}): {rfe_feats}")

    lasso_mask, lasso_feats = selector.select_lasso(X_train, y_train)
    print(f"3. Lasso (Embedded) selected ({len(lasso_feats)}): {lasso_feats}")

    vif_mask, vif_feats = selector.select_vif_filter(X_train, threshold=10.0)
    print(f"4. VIF Filter selected ({len(vif_feats)}): {vif_feats}")

    comp_table = selector.get_comparison_table()
    viz.plot_feature_selection_comparison(comp_table, output_name="05_feature_selection_comparison.png", title="WHO Life Expectancy: Feature Selection Comparison")
    print("Generated: figures/05_feature_selection_comparison.png")

    trainer = RegressionTrainer(random_state=42)

    lr_baseline = LinearRegression()
    lr_baseline.fit(X_train, y_train)

    lr_kbest = LinearRegression()
    lr_kbest.fit(X_train[:, k_best_mask], y_train)

    lr_rfe = LinearRegression()
    lr_rfe.fit(X_train[:, rfe_mask], y_train)

    lr_lasso_feats = LinearRegression()
    lr_lasso_feats.fit(X_train[:, lasso_mask], y_train)

    lr_vif = LinearRegression()
    lr_vif.fit(X_train[:, vif_mask], y_train)

    ridge_model = trainer.train_ridge(X_train, y_train, "Ridge Regression")
    lasso_model = trainer.train_lasso(X_train, y_train, "Lasso Regression")
    elastic_model = trainer.train_elastic_net(X_train, y_train, "ElasticNet")

    # PHASE 5: EVALUATION
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

    viz.plot_metrics_comparison(metrics_df, output_name="06_model_metrics_comparison.png")
    print("\nGenerated: figures/06_model_metrics_comparison.png")

    best_fs_name = "RFE + OLS (12 Feats)"
    viz.plot_actual_vs_predicted(
        y_test,
        predictions_dict["Baseline OLS (All 20 Feats)"],
        predictions_dict[best_fs_name],
        base_name="Baseline OLS (20 Feats)",
        best_name=best_fs_name,
        target_label="Life Expectancy (Years)",
        output_name="07_actual_vs_predicted.png"
    )
    print("Generated: figures/07_actual_vs_predicted.png")

    best_model = lr_rfe
    best_mask = rfe_mask
    best_pred = predictions_dict[best_fs_name]
    best_feat_names = [f for f, m in zip(feature_names, best_mask) if m]

    assumptions = ModelEvaluator.test_regression_assumptions(
        X_test[:, best_mask], y_test, best_pred, best_feat_names
    )

    print("\n--- Regression Assumptions Diagnostic Report (Best Feature-Selected Model) ---")
    print(f"1. Linearity (Pearson r): {assumptions['Linearity_Pearson_r']}")
    print(f"2. Normality of Residuals: Shapiro p={assumptions['Normality_Shapiro_p']}, Jarque-Bera p={assumptions['Normality_JarqueBera_p']}")
    print(f"3. Homoscedasticity (Breusch-Pagan p): {assumptions['Homoscedasticity_BreuschPagan_p']}")
    print(f"4. Independence of Errors (Durbin-Watson): {assumptions['Independence_DurbinWatson']}")
    print(f"5. Multicollinearity: Max VIF = {assumptions['Max_VIF']}, Mean VIF = {assumptions['Mean_VIF']}")

    viz.plot_residuals_analysis(y_test, best_pred, model_name=best_fs_name, output_name="08_residuals_analysis.png")
    viz.plot_feature_importance(best_feat_names, best_model.coef_, model_name=best_fs_name, target_name="Life Expectancy", output_name="09_feature_importance.png")
    print("Generated: figures/08_residuals_analysis.png, figures/09_feature_importance.png")

    # PHASE 6: DEPLOYMENT
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

    interventions = {"Schooling": 3.0, "Diphtheria": 25.0, "Adult Mortality": -50.0}
    sim_result = predictor.simulate_policy_scenario(sample_country, interventions)
    print("Policy Intervention Simulation Result:")
    for k, v in sim_result.items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 80)
    print("PROBLEM 1: WHO LIFE EXPECTANCY PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    return metrics_df, assumptions


def run_steel_industry_pipeline():
    """Runs complete CRISP-DM workflow for Problem 2 (Bonus 1): Steel Industry Energy Consumption (14 features)."""
    print("\n" + "=" * 80)
    print("CRISP-DM PIPELINE (PROBLEM 2): STEEL INDUSTRY ENERGY CONSUMPTION REGRESSION")
    print("=" * 80)

    # PHASE 1: BUSINESS UNDERSTANDING
    print("\n[PHASE 1: Business Understanding]")
    print("- Project Goal: Predict industrial electricity consumption (Usage_kWh) of DAEWOO Steel Co.")
    print("- Business Value: Enable smart factory load dispatching, peak shaving, and energy conservation.")
    print("- Selected Dataset: UCI / Kaggle 'Steel Industry Energy Consumption' (14 engineered features > 13 features threshold).")

    # PHASE 2: DATA UNDERSTANDING
    print("\n[PHASE 2: Data Understanding]")
    loader = DataLoader(dataset_type="steel")
    raw_df = loader.load_data()
    print(f"Dataset Shape: {raw_df.shape[0]} rows, {raw_df.shape[1]} columns")

    missing_df = loader.get_missing_values()
    print(f"Missing Values: {len(missing_df)} columns with missing data (100% complete records).")

    target_corr = loader.get_feature_correlations(target="Usage_kWh")
    print("\nTop Correlations with Industrial Usage_kWh:")
    print(target_corr)

    viz = Visualizer(output_dir="figures")
    viz.plot_target_distribution(raw_df, target_col="Usage_kWh", output_name="steel_01_target_distribution.png", unit="kWh", group_col="Load_Type")
    viz.plot_correlation_matrix(raw_df, output_name="steel_02_correlation_matrix.png", title="Steel Industry Electrical Indicators Correlation Matrix")

    # PHASE 3: DATA PREPARATION
    print("\n[PHASE 3: Data Preparation]")
    preprocessor = DataPreprocessor(test_size=0.2, random_state=42)
    cleaned_df, X_train, X_test, y_train, y_test, feature_names = preprocessor.prepare_steel_industry(raw_df)

    print(f"Prepared Dataset: {cleaned_df.shape[0]} samples")
    print(f"Training set: {X_train.shape[0]} samples | Testing set: {X_test.shape[0]} samples")
    print(f"Candidate Engineered Features ({len(feature_names)} features, >13 compliant):")
    for idx, f in enumerate(feature_names, 1):
        print(f"  {idx:2d}. {f}")

    viz.plot_missing_values(raw_df, cleaned_df, output_name="steel_03_missing_values_heatmap.png")
    viz.plot_feature_distributions(
        cleaned_df,
        features=[
            "Lagging_Current_Reactive.Power_kVarh",
            "Leading_Current_Reactive_Power_kVarh",
            "CO2(tCO2)",
            "Lagging_Current_Power_Factor",
            "Leading_Current_Power_Factor",
            "NSM"
        ],
        output_name="steel_04_feature_distributions.png",
        title="Distributions of Key Steel Industry Electrical Features"
    )
    print("Generated: figures/steel_01_target_distribution.png, figures/steel_02_correlation_matrix.png, figures/steel_03_missing_values_heatmap.png, figures/steel_04_feature_distributions.png")

    # PHASE 4: FEATURE SELECTION & MODELING
    print("\n[PHASE 4: Modeling & Feature Selection Experiments]")
    selector = FeatureSelector(feature_names, random_state=42)

    k_best_mask, k_best_feats = selector.select_k_best(X_train, y_train, k=10)
    print(f"\n1. SelectKBest (Filter, k=10) selected ({len(k_best_feats)}): {k_best_feats}")

    rfe_mask, rfe_feats = selector.select_rfe(X_train, y_train, n_features_to_select=10)
    print(f"2. RFE (Wrapper, n=10) selected ({len(rfe_feats)}): {rfe_feats}")

    lasso_mask, lasso_feats = selector.select_lasso(X_train, y_train)
    print(f"3. Lasso (Embedded) selected ({len(lasso_feats)}): {lasso_feats}")

    vif_mask, vif_feats = selector.select_vif_filter(X_train, threshold=10.0)
    print(f"4. VIF Filter selected ({len(vif_feats)}): {vif_feats}")

    comp_table = selector.get_comparison_table()
    viz.plot_feature_selection_comparison(comp_table, output_name="steel_05_feature_selection_comparison.png", title="Steel Industry: Feature Selection Comparison")
    print("Generated: figures/steel_05_feature_selection_comparison.png")

    trainer = RegressionTrainer(random_state=42)

    lr_baseline = LinearRegression()
    lr_baseline.fit(X_train, y_train)

    lr_kbest = LinearRegression()
    lr_kbest.fit(X_train[:, k_best_mask], y_train)

    lr_rfe = LinearRegression()
    lr_rfe.fit(X_train[:, rfe_mask], y_train)

    lr_lasso_feats = LinearRegression()
    lr_lasso_feats.fit(X_train[:, lasso_mask], y_train)

    lr_vif = LinearRegression()
    lr_vif.fit(X_train[:, vif_mask], y_train)

    ridge_model = trainer.train_ridge(X_train, y_train, "Ridge Regression")
    lasso_model = trainer.train_lasso(X_train, y_train, "Lasso Regression")
    elastic_model = trainer.train_elastic_net(X_train, y_train, "ElasticNet")

    # PHASE 5: EVALUATION
    print("\n[PHASE 5: Evaluation]")
    models_to_evaluate = [
        ("Baseline OLS (All 14 Feats)", lr_baseline, np.ones(len(feature_names), dtype=bool)),
        ("SelectKBest + OLS (10 Feats)", lr_kbest, k_best_mask),
        ("RFE + OLS (10 Feats)", lr_rfe, rfe_mask),
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
    print("\nSteel Industry Model Performance Evaluation Table on Test Set:")
    print(metrics_df.to_string(index=False))

    viz.plot_metrics_comparison(metrics_df, output_name="steel_06_model_metrics_comparison.png")
    print("\nGenerated: figures/steel_06_model_metrics_comparison.png")

    best_fs_name = "RFE + OLS (10 Feats)"
    viz.plot_actual_vs_predicted(
        y_test,
        predictions_dict["Baseline OLS (All 14 Feats)"],
        predictions_dict[best_fs_name],
        base_name="Baseline OLS (14 Feats)",
        best_name=best_fs_name,
        target_label="Energy Consumption (Usage_kWh)",
        output_name="steel_07_actual_vs_predicted.png"
    )
    print("Generated: figures/steel_07_actual_vs_predicted.png")

    best_model = lr_rfe
    best_mask = rfe_mask
    best_pred = predictions_dict[best_fs_name]
    best_feat_names = [f for f, m in zip(feature_names, best_mask) if m]

    assumptions = ModelEvaluator.test_regression_assumptions(
        X_test[:, best_mask], y_test, best_pred, best_feat_names
    )

    print("\n--- Regression Assumptions Diagnostic Report (Steel Best Model) ---")
    print(f"1. Linearity (Pearson r): {assumptions['Linearity_Pearson_r']}")
    print(f"2. Normality of Residuals: Shapiro p={assumptions['Normality_Shapiro_p']}, Jarque-Bera p={assumptions['Normality_JarqueBera_p']}")
    print(f"3. Homoscedasticity (Breusch-Pagan p): {assumptions['Homoscedasticity_BreuschPagan_p']}")
    print(f"4. Independence of Errors (Durbin-Watson): {assumptions['Independence_DurbinWatson']}")
    print(f"5. Multicollinearity: Max VIF = {assumptions['Max_VIF']}, Mean VIF = {assumptions['Mean_VIF']}")

    viz.plot_residuals_analysis(y_test, best_pred, model_name=best_fs_name, output_name="steel_08_residuals_analysis.png")
    viz.plot_feature_importance(best_feat_names, best_model.coef_, model_name=best_fs_name, target_name="Usage_kWh", output_name="steel_09_feature_importance.png")
    print("Generated: figures/steel_08_residuals_analysis.png, figures/steel_09_feature_importance.png")

    # PHASE 6: DEPLOYMENT
    print("\n[PHASE 6: Deployment & Factory Energy Simulation]")
    deployer = ModelDeployer(models_dir="models")
    pipeline_path = deployer.save_pipeline(
        model=best_model,
        scaler=preprocessor.get_scaler(),
        all_features=feature_names,
        selected_mask=best_mask,
        imputation_values=preprocessor.imputation_values_,
        filename="steel_linear_regression_pipeline.joblib"
    )

    predictor = SteelEnergyPredictor(pipeline_path)
    sample_operation = {
        "Lagging_Current_Reactive.Power_kVarh": 45.2,
        "Leading_Current_Reactive_Power_kVarh": 0.05,
        "CO2(tCO2)": 0.04,
        "Lagging_Current_Power_Factor": 72.5,
        "Leading_Current_Power_Factor": 100.0,
        "NSM": 54000,
        "Month": 7,
        "Hour": 15,
        "Day": 18,
        "Is_Weekday": 1,
        "Load_Maximum_Load": 1,
        "Load_Medium_Load": 0,
        "Day_Weekend": 0,
        "Peak_Hour": 1
    }
    base_usage = predictor.predict_sample(sample_operation)
    print(f"\nPredicted Baseline Energy Consumption for sample steel shift: {base_usage:.2f} kWh")

    energy_saving_adjustments = {
        "Lagging_Current_Power_Factor": 15.0,  # +15% power factor improvement (power capacitor bank)
        "Lagging_Current_Reactive.Power_kVarh": -15.0  # -15 kVarh reactive power loss
    }
    sim_result = predictor.simulate_energy_saving(sample_operation, energy_saving_adjustments)
    print("\nSmart Factory Energy Efficiency Simulation Result:")
    for k, v in sim_result.items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 80)
    print("PROBLEM 2: STEEL INDUSTRY PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    return metrics_df, assumptions


def run_arbitrary_pipeline(csv_path_or_url: str, target_col: str = None):
    """
    Executes full CRISP-DM Multiple Linear Regression pipeline on ANY custom dataset.
    Demonstrated with 50_Startups.csv or any user provided URL/path (Bonus 2).
    """
    print("\n" + "=" * 80)
    print(f"CRISP-DM PIPELINE (BONUS 2: ARBITRARY DATASET ANALYSIS)")
    print(f"Target Source: {csv_path_or_url}")
    print("=" * 80)

    # 1. LOAD
    if csv_path_or_url.startswith("http://") or csv_path_or_url.startswith("https://"):
        df = DataLoader.load_from_url(csv_path_or_url)
    else:
        df = DataLoader.load_from_file(csv_path_or_url)

    print(f"\nLoaded dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Columns: {df.columns.tolist()}")

    # 2. PREPROCESS
    preprocessor = DataPreprocessor(test_size=0.2, random_state=42)
    cleaned_df, X_train, X_test, y_train, y_test, feature_names, detected_target = preprocessor.prepare_arbitrary(
        df, target_col=target_col
    )

    print(f"\nPreprocessed Features ({len(feature_names)}): {feature_names}")
    print(f"Target variable: '{detected_target}'")
    print(f"Train samples: {X_train.shape[0]} | Test samples: {X_test.shape[0]}")

    # 3. FEATURE SELECTION
    selector = FeatureSelector(feature_names, random_state=42)
    k_select = min(12, max(1, len(feature_names) // 2 if len(feature_names) <= 6 else len(feature_names) - 1))
    k_mask, k_feats = selector.select_k_best(X_train, y_train, k=k_select)
    rfe_mask, rfe_feats = selector.select_rfe(X_train, y_train, n_features_to_select=k_select)
    lasso_mask, lasso_feats = selector.select_lasso(X_train, y_train)

    print(f"\nFeature Selection Results:")
    print(f"- SelectKBest (k={k_select}): {k_feats}")
    print(f"- RFE (k={k_select}): {rfe_feats}")
    print(f"- Lasso (non-zero): {lasso_feats}")

    # 4. MODELING & EVALUATION
    lr_baseline = LinearRegression()
    lr_baseline.fit(X_train, y_train)

    lr_kbest = LinearRegression()
    lr_kbest.fit(X_train[:, k_mask], y_train)

    lr_rfe = LinearRegression()
    lr_rfe.fit(X_train[:, rfe_mask], y_train)

    lr_lasso = LinearRegression()
    lr_lasso.fit(X_train[:, lasso_mask], y_train)

    trainer = RegressionTrainer(random_state=42)
    ridge_model = trainer.train_ridge(X_train, y_train, "Ridge Regression")

    models_to_evaluate = [
        (f"Baseline OLS ({len(feature_names)} Feats)", lr_baseline, np.ones(len(feature_names), dtype=bool)),
        (f"SelectKBest + OLS ({len(k_feats)} Feats)", lr_kbest, k_mask),
        (f"RFE + OLS ({len(rfe_feats)} Feats)", lr_rfe, rfe_mask),
        (f"Lasso Selected + OLS ({len(lasso_feats)} Feats)", lr_lasso, lasso_mask),
        (f"Ridge Regression (All Feats)", ridge_model, np.ones(len(feature_names), dtype=bool)),
    ]

    metrics_list = []
    best_r2 = -1e9
    best_model_name = ""
    best_pred = None
    best_mask = None
    best_model = None

    for name, model, mask in models_to_evaluate:
        X_test_sub = X_test[:, mask]
        y_pred = model.predict(X_test_sub)
        n_feat = int(np.sum(mask))
        metric = ModelEvaluator.calculate_metrics(y_test, y_pred, n_feat, model_name=name)
        metrics_list.append(metric)
        if metric["R2"] > best_r2:
            best_r2 = metric["R2"]
            best_model_name = name
            best_pred = y_pred
            best_mask = mask
            best_model = model

    metrics_df = pd.DataFrame(metrics_list)
    print("\nModel Evaluation Performance Table:")
    print(metrics_df.to_string(index=False))

    # 5. REGRESSION ASSUMPTIONS
    best_feat_names = [f for f, m in zip(feature_names, best_mask) if m]
    assumptions = ModelEvaluator.test_regression_assumptions(
        X_test[:, best_mask], y_test, best_pred, best_feat_names
    )
    print(f"\nRegression Assumptions Diagnostics on Best Model ({best_model_name}):")
    print(f"- Linearity (r): {assumptions['Linearity_Pearson_r']}")
    print(f"- Residual Normality (Shapiro p): {assumptions['Normality_Shapiro_p']}")
    print(f"- Homoscedasticity (Breusch-Pagan p): {assumptions['Homoscedasticity_BreuschPagan_p']}")
    print(f"- Independence (Durbin-Watson): {assumptions['Independence_DurbinWatson']}")
    print(f"- Multicollinearity: Max VIF = {assumptions['Max_VIF']}, Mean VIF = {assumptions['Mean_VIF']}")

    print("\n" + "=" * 80)
    print("CUSTOM DATASET PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    return metrics_df, assumptions


def main():
    parser = argparse.ArgumentParser(description="HW2 Multiple Linear Regression Pipeline Runner")
    parser.add_argument(
        "--problem",
        type=str,
        default="life",
        choices=["life", "steel", "all", "custom"],
        help="Problem to execute: 'life' (WHO Life Expectancy, 20 feats), 'steel' (Steel Industry, 14 feats), 'all' (both), 'custom' (arbitrary dataset)"
    )
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="URL to custom CSV dataset (e.g. https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/50_Startups.csv)"
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Local file path to custom CSV dataset"
    )
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Target column name for custom dataset"
    )

    args = parser.parse_args()

    if args.url:
        run_arbitrary_pipeline(args.url, target_col=args.target)
    elif args.file:
        run_arbitrary_pipeline(args.file, target_col=args.target)
    elif args.problem == "steel":
        run_steel_industry_pipeline()
    elif args.problem == "all":
        run_life_expectancy_pipeline()
        run_steel_industry_pipeline()
    elif args.problem == "custom":
        # Default custom demo
        default_url = "https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/50_Startups.csv"
        run_arbitrary_pipeline(default_url, target_col="Profit")
    else:
        run_life_expectancy_pipeline()


if __name__ == "__main__":
    main()
