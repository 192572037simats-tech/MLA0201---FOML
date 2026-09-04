"""
Visualization Module
----------------------
Generates all charts used in the churn prediction and customer
segmentation analysis, and saves them as PNG files inside outputs/.
"""

import os
import matplotlib
matplotlib.use("Agg")  # Allows chart generation without a display (headless environments)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

sns.set_style("whitegrid")


def _save_and_close(fig, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved chart: {output_path}")


def plot_churn_distribution(df_raw: pd.DataFrame, output_path: str):
    """Bar chart showing how many customers churned vs stayed."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.countplot(x="Churn", data=df_raw, hue="Churn", palette="Set2", legend=False, ax=ax)
    ax.set_title("Customer Churn Distribution")
    ax.set_xlabel("Churn")
    ax.set_ylabel("Number of Customers")
    _save_and_close(fig, output_path)


def plot_model_comparison(results: dict, output_path: str):
    """Bar chart comparing Accuracy, Precision, Recall, and F1 Score across models."""
    metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1 Score"]
    model_names = list(results.keys())

    data = {metric: [results[m][metric] for m in model_names] for metric in metrics_to_plot}
    df_metrics = pd.DataFrame(data, index=model_names)

    fig, ax = plt.subplots(figsize=(9, 6))
    df_metrics.plot(kind="bar", ax=ax, colormap="viridis")
    ax.set_title("Model Comparison")
    ax.set_ylabel("Score")
    ax.set_xlabel("Model")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    plt.xticks(rotation=15)
    _save_and_close(fig, output_path)


def plot_confusion_matrix(cm, model_name: str, output_path: str):
    """Heatmap of the confusion matrix for the best model."""
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Churn", "Churn"],
                yticklabels=["No Churn", "Churn"], ax=ax)
    ax.set_title(f"Confusion Matrix - {model_name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    _save_and_close(fig, output_path)


def plot_elbow_curve(cluster_range, inertia_values, output_path: str):
    """Elbow method plot used to justify the chosen number of clusters."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(list(cluster_range), inertia_values, marker="o", color="teal")
    ax.set_title("Elbow Method for Optimal K")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Inertia")
    _save_and_close(fig, output_path)


def plot_customer_segments(df_segmented: pd.DataFrame, output_path: str):
    """Scatter plot showing customer segments based on Tenure vs Monthly Charges."""
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.scatterplot(
        data=df_segmented,
        x="tenure",
        y="MonthlyCharges",
        hue="Segment",
        palette="tab10",
        alpha=0.7,
        ax=ax,
    )
    ax.set_title("Customer Segments (Tenure vs Monthly Charges)")
    ax.set_xlabel("Tenure (months)")
    ax.set_ylabel("Monthly Charges")
    _save_and_close(fig, output_path)


def plot_cluster_distribution(df_segmented: pd.DataFrame, output_path: str):
    """Bar chart showing how many customers fall into each segment."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.countplot(x="Segment", data=df_segmented, hue="Segment", palette="tab10", legend=False, ax=ax)
    ax.set_title("Customer Distribution Across Segments")
    ax.set_xlabel("Segment")
    ax.set_ylabel("Number of Customers")
    _save_and_close(fig, output_path)


def generate_all_visualizations(df_raw, results, best_model_name, best_cm,
                                 cluster_range, inertia_values, df_segmented,
                                 output_dir="outputs"):
    """Generate and save every visualization required by the project."""
    plot_churn_distribution(df_raw, os.path.join(output_dir, "churn_distribution.png"))
    plot_model_comparison(results, os.path.join(output_dir, "model_comparison.png"))
    plot_confusion_matrix(best_cm, best_model_name, os.path.join(output_dir, "confusion_matrix.png"))
    plot_elbow_curve(cluster_range, inertia_values, os.path.join(output_dir, "elbow_curve.png"))
    plot_customer_segments(df_segmented, os.path.join(output_dir, "customer_segments.png"))
    plot_cluster_distribution(df_segmented, os.path.join(output_dir, "cluster_distribution.png"))
    print("\nAll visualizations generated successfully.")
