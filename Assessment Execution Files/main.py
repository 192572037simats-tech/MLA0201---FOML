"""
Customer Churn Prediction and Segmentation System
====================================================
Main entry point for the project.

Running this script will:
  1. Load the dataset
  2. Preprocess the data
  3. Train multiple churn prediction models
  4. Evaluate and compare the models
  5. Select and save the best model
  6. Perform customer segmentation using K-Means
  7. Generate all visualizations
  8. Save all outputs inside the outputs/ and models/ folders

Usage:
    python main.py
"""

import os
import sys
import pandas as pd

# Make sure the src/ folder is importable
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from data_preprocessing import preprocess_pipeline
from churn_prediction import run_churn_prediction
from customer_segmentation import run_customer_segmentation
from visualization import generate_all_visualizations

DATA_PATH = os.path.join("data", "Telco-Customer-Churn.csv")
MODELS_DIR = "models"
OUTPUTS_DIR = "outputs"


def main():
    print("=" * 60)
    print("CUSTOMER CHURN PREDICTION AND SEGMENTATION SYSTEM")
    print("=" * 60)

    # 1 & 2. Load and preprocess the data
    print("\n[STEP 1/4] Loading and preprocessing data...")
    df_raw, df_encoded, encoders, scaler, feature_columns = preprocess_pipeline(DATA_PATH)

    # 3, 4 & 5. Train, evaluate, compare, and save the best churn prediction model
    print("\n[STEP 2/4] Training and evaluating churn prediction models...")
    results, trained_models, best_name, best_model, (X_test, y_test) = run_churn_prediction(
        df_encoded, feature_columns, models_dir=MODELS_DIR
    )
    best_cm = results[best_name]["Confusion Matrix"]

    # 6. Customer segmentation
    print("\n[STEP 3/4] Performing customer segmentation with K-Means...")
    df_segmented, cluster_range, inertia_values, segment_profile, _ = run_customer_segmentation(
        df_raw, n_clusters=4, max_clusters=10
    )

    # Save the segmentation results as a CSV for reference
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    segmented_output_path = os.path.join(OUTPUTS_DIR, "segmented_customers.csv")
    df_segmented.to_csv(segmented_output_path, index=False)
    print(f"Segmented customer data saved to: {segmented_output_path}")

    segment_profile_path = os.path.join(OUTPUTS_DIR, "segment_profile.csv")
    segment_profile.to_csv(segment_profile_path)
    print(f"Segment profile summary saved to: {segment_profile_path}")

    # 7. Generate all visualizations
    print("\n[STEP 4/4] Generating visualizations...")
    generate_all_visualizations(
        df_raw, results, best_name, best_cm,
        cluster_range, inertia_values, df_segmented,
        output_dir=OUTPUTS_DIR,
    )

    # Save a summary report of model performance
    summary_path = os.path.join(OUTPUTS_DIR, "model_performance_summary.csv")
    summary_rows = []
    for name, metrics in results.items():
        summary_rows.append({
            "Model": name,
            "Accuracy": round(metrics["Accuracy"], 4),
            "Precision": round(metrics["Precision"], 4),
            "Recall": round(metrics["Recall"], 4),
            "F1 Score": round(metrics["F1 Score"], 4),
        })
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"Model performance summary saved to: {summary_path}")

    print("\n" + "=" * 60)
    print("PROJECT EXECUTION COMPLETE")
    print(f"Best Model : {best_name}")
    print(f"Models saved in   : {MODELS_DIR}/")
    print(f"Outputs saved in  : {OUTPUTS_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
