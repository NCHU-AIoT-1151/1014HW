"""
Standalone Deployment Script & CLI for Life Expectancy Prediction.
Allows users and instructors to predict life expectancy on custom country profiles or run scenario simulations.
"""
import os
import argparse
import json
from src.deployment import LifeExpectancyPredictor, SteelEnergyPredictor


def main():
    parser = argparse.ArgumentParser(
        description="Predict Regression Target using trained Multiple Linear Regression Pipeline."
    )
    parser.add_argument(
        "--problem",
        type=str,
        default="life",
        choices=["life", "steel"],
        help="Problem to predict: 'life' (WHO Life Expectancy) or 'steel' (Steel Industry Energy Consumption)"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to serialized joblib model pipeline. Defaults to corresponding problem pipeline."
    )
    parser.add_argument(
        "--input-json",
        type=str,
        default=None,
        help="Path to JSON file containing feature indicators."
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Run scenario simulation."
    )
    args = parser.parse_args()

    if args.problem == "steel":
        model_path = args.model_path or "models/steel_linear_regression_pipeline.joblib"
        if not os.path.exists(model_path):
            print(f"Error: Model file '{model_path}' not found. Running main.py --problem steel first...")
            from main import run_steel_industry_pipeline
            run_steel_industry_pipeline()

        predictor = SteelEnergyPredictor(model_path)
        print("=" * 70)
        print("STEEL INDUSTRY ENERGY CONSUMPTION PREDICTION ENGINE (DEPLOYED)")
        print("=" * 70)
        print(f"Model loaded: {model_path}")
        print(f"Selected Features ({len(predictor.artifact['selected_features'])}): {predictor.artifact['selected_features']}")

        if args.input_json and os.path.exists(args.input_json):
            with open(args.input_json, "r") as f:
                sample_data = json.load(f)
            print(f"\nLoaded custom operational profile from {args.input_json}")
        else:
            sample_data = {
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
            print("\nUsing standard representative steel factory maximum load shift profile:")

        for k, v in sample_data.items():
            print(f"  - {k}: {v}")

        pred_val = predictor.predict_sample(sample_data)
        print("\n" + "-" * 70)
        print(f"--> PREDICTED ELECTRICITY CONSUMPTION: {pred_val:.2f} kWh <--")
        print("-" * 70)

        print("\n[SMART FACTORY ENERGY CONSERVATION SIMULATION]")
        print("Simulating power factor optimization & reactive power reduction:")
        adjustments = {
            "Lagging_Current_Power_Factor": 15.0,  # +15% power factor improvement
            "Lagging_Current_Reactive.Power_kVarh": -15.0  # -15 kVarh reactive power loss
        }
        for k, delta in adjustments.items():
            print(f"  * {k}: {'+' if delta > 0 else ''}{delta}")

        sim_res = predictor.simulate_energy_saving(sample_data, adjustments)
        print(f"\nBaseline Consumption: {sim_res['Base Usage (kWh)']:.2f} kWh")
        print(f"Optimized Consumption: {sim_res['Optimized Usage (kWh)']:.2f} kWh")
        print(f"Conserved Electricity: {sim_res['Energy Saved (kWh)']:.2f} kWh ({sim_res['Percentage Change (%)']}%)")
        print("=" * 70)

    else:
        # Life Expectancy (Default)
        model_path = args.model_path or "models/best_linear_regression_pipeline.joblib"
        if not os.path.exists(model_path):
            print(f"Error: Model file '{model_path}' not found. Please run main.py first.")
            return

        predictor = LifeExpectancyPredictor(model_path)
        print("=" * 70)
        print("WHO LIFE EXPECTANCY PREDICTION ENGINE (DEPLOYED)")
        print("=" * 70)
        print(f"Model loaded: {model_path}")
        print(f"Selected Features ({len(predictor.artifact['selected_features'])}): {predictor.artifact['selected_features']}")

        if args.input_json and os.path.exists(args.input_json):
            with open(args.input_json, "r") as f:
                country_data = json.load(f)
            print(f"\nLoaded custom country data from {args.input_json}")
        else:
            country_data = {
                "Country": "Demonstration Country",
                "Status": "Developing",
                "Adult Mortality": 250.0,
                "infant deaths": 50,
                "Alcohol": 1.5,
                "percentage expenditure": 100.0,
                "Hepatitis B": 70.0,
                "Measles": 500,
                "BMI": 22.0,
                "under-five deaths": 65,
                "Polio": 75.0,
                "Total expenditure": 6.5,
                "Diphtheria": 75.0,
                "HIV/AIDS": 0.5,
                "GDP": 1200.0,
                "Population": 25000000.0,
                "thinness  1-19 years": 15.0,
                "thinness 5-9 years": 15.0,
                "Income composition of resources": 0.52,
                "Schooling": 10.5,
                "Year": 2015
            }
            print("\nUsing standard representative developing nation profile:")

        for k, v in country_data.items():
            print(f"  - {k}: {v}")

        prediction = predictor.predict_sample(country_data)
        print("\n" + "-" * 70)
        print(f"--> PREDICTED LIFE EXPECTANCY: {prediction:.2f} YEARS <--")
        print("-" * 70)

        print("\n[POLICY INTERVENTION SIMULATION]")
        print("Simulating the following health & education policy reforms:")
        interventions = {
            "Schooling": 2.5,
            "Diphtheria": 20.0,
            "Polio": 15.0,
            "Adult Mortality": -40.0
        }
        for k, delta in interventions.items():
            print(f"  * {k}: {'+' if delta > 0 else ''}{delta}")

        sim_result = predictor.simulate_policy_scenario(country_data, interventions)
        print(f"\nBaseline Life Expectancy:     {sim_result['Base Life Expectancy']:.2f} years")
        print(f"Post-Intervention Expectancy: {sim_result['Intervention Life Expectancy']:.2f} years")
        print(f"Expected Gain in Longevity:   +{sim_result['Net Change (Years)']:.2f} YEARS")
        print("=" * 70)



if __name__ == "__main__":
    main()
