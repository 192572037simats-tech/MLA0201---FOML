"""
Customer Segmentation Module
------------------------------
Uses K-Means Clustering to group customers into meaningful segments
based on Tenure, Monthly Charges, and Total Charges.
"""

import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

SEGMENTATION_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
RANDOM_STATE = 42


def select_segmentation_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Select the relevant customer characteristics used for segmentation."""
    available = [c for c in SEGMENTATION_FEATURES if c in df_raw.columns]
    return df_raw[available].copy()


def scale_features(df_features: pd.DataFrame):
    """Scale features so K-Means is not biased towards larger-magnitude columns."""
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df_features)
    return scaled, scaler


def find_optimal_clusters(scaled_features, max_clusters: int = 10):
    """
    Run K-Means for a range of cluster counts (1..max_clusters) and
    record the inertia (within-cluster sum of squares) for each,
    used to plot the Elbow Method curve.
    """
    inertia_values = []
    cluster_range = range(1, max_clusters + 1)

    for k in cluster_range:
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        kmeans.fit(scaled_features)
        inertia_values.append(kmeans.inertia_)

    return list(cluster_range), inertia_values


def apply_kmeans(scaled_features, n_clusters: int = 4):
    """Fit the final K-Means model with the chosen number of clusters."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled_features)
    return kmeans, cluster_labels


def profile_segments(df_raw: pd.DataFrame, cluster_labels, feature_cols) -> pd.DataFrame:
    """Compute the average feature values for each segment (cluster)."""
    df_with_clusters = df_raw.copy()
    df_with_clusters["Segment"] = cluster_labels
    profile = df_with_clusters.groupby("Segment")[feature_cols].mean().round(2)
    profile["Customer Count"] = df_with_clusters.groupby("Segment").size()
    return profile


def run_customer_segmentation(df_raw: pd.DataFrame, n_clusters: int = 4, max_clusters: int = 10):
    """
    Full customer segmentation pipeline:
    select features -> scale -> elbow method -> K-Means -> profile segments.
    Returns the dataframe with an added 'Segment' column, cluster range,
    inertia values (for the elbow curve), and the segment profile summary.
    """
    df_features = select_segmentation_features(df_raw)
    scaled_features, scaler = scale_features(df_features)

    cluster_range, inertia_values = find_optimal_clusters(scaled_features, max_clusters)

    kmeans_model, cluster_labels = apply_kmeans(scaled_features, n_clusters)

    df_segmented = df_raw.copy()
    df_segmented["Segment"] = cluster_labels

    segment_profile = profile_segments(df_raw, cluster_labels, df_features.columns.tolist())

    print("\nCustomer Segment Profile (average values per segment):")
    print(segment_profile)

    return df_segmented, cluster_range, inertia_values, segment_profile, scaled_features
