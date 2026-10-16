"""
Standalone Deployment Script & CLI for Life Expectancy Prediction.
Allows users and instructors to predict life expectancy on custom country profiles or run scenario simulations.
"""
import os
import argparse
import json
from src.deployment import LifeExpectancyPredictor


def main():
    parser = argparse.ArgumentParser(
        description="Predict National Life Expectancy using trained Multiple Linear Regression Pipeline."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/best_linear_regression_pipeline.joblib",
        help="Path to serialized joblib model pipeline."
    )
    parser.add_argument(
        "--input-json",
        type=str,
        default=None,
        help="Path to JSON file containing country indicators."
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Run policy intervention scenario simulation."
    )
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"Error: Model file '{args.model_path}' not found. Please run main.py first.")
        return

    predictor = LifeExpectancyPredictor(args.model_path)
    print("=" * 70)
    print("WHO LIFE EXPECTANCY PREDICTION ENGINE (DEPLOYED)")
    print("=" * 70)
    print(f"Model loaded: {args.model_path}")
    print(f"Selected Features ({len(predictor.artifact['selected_features'])}): {predictor.artifact['selected_features']}")

    # Load data or use default sample
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

    if args.simulate or True:
        print("\n[POLICY INTERVENTION SIMULATION]")
        print("Simulating the following health & education policy reforms:")
        interventions = {
            "Schooling": 2.5,          # +2.5 years of education
            "Diphtheria": 20.0,        # +20% immunization coverage
            "Polio": 15.0,             # +15% polio vaccination
            "Adult Mortality": -40.0   # -40 per 1000 mortality reduction
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
