"""
Data Preprocessing Module
--------------------------
Loads the raw Telco Customer Churn dataset and prepares it for
machine learning: cleans missing values, encodes categorical
columns, and scales numeric columns.
"""

import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Columns that are unique identifiers and carry no predictive value
COLUMNS_TO_DROP = ["customerID"]

# The target column we want to predict
TARGET_COLUMN = "Churn"


def load_dataset(data_path: str) -> pd.DataFrame:
    """Load the raw churn dataset from a CSV file."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset not found at '{data_path}'. "
            "Please place the Telco Customer Churn CSV file inside the data/ folder. "
            "See data/README.md for download instructions."
        )
    df = pd.read_csv(data_path)
    print(f"Loaded dataset with shape: {df.shape}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values and fix column data types."""
    df = df.copy()

    # 'TotalCharges' is sometimes stored as a string with blank spaces
    # for customers with zero tenure. Convert it to numeric and fix NaNs.
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # Drop rows that are still fully empty (safety net)
    df = df.dropna(how="all")

    # Fill any remaining missing numeric values with the column median
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    # Fill any remaining missing categorical values with the column mode
    categorical_cols = df.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])

    return df


def remove_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns that are not useful for modeling."""
    df = df.copy()
    cols_present = [c for c in COLUMNS_TO_DROP if c in df.columns]
    if cols_present:
        df = df.drop(columns=cols_present)
    return df


def encode_categorical_features(df: pd.DataFrame):
    """
    Convert categorical (text) columns into numerical format using
    Label Encoding so they can be used by machine learning models.
    Returns the encoded dataframe and a dictionary of fitted encoders
    (useful if you ever need to decode predictions back to labels).
    """
    df = df.copy()
    encoders = {}

    categorical_cols = df.select_dtypes(include=["object"]).columns

    for col in categorical_cols:
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col].astype(str))
        encoders[col] = encoder

    return df, encoders


def scale_numerical_features(df: pd.DataFrame, feature_columns):
    """Scale numerical features to a common range using StandardScaler."""
    df = df.copy()
    scaler = StandardScaler()
    df[feature_columns] = scaler.fit_transform(df[feature_columns])
    return df, scaler


def preprocess_pipeline(data_path: str):
    """
    Run the full preprocessing pipeline and return:
      - df_raw:        cleaned dataframe with original readable values (for segmentation/visuals)
      - df_encoded:     fully encoded & scaled dataframe (for ML models)
      - encoders:       dictionary of label encoders used
      - scaler:         fitted StandardScaler used on numeric columns
      - feature_columns: list of feature column names used for training
    """
    df = load_dataset(data_path)
    df = clean_data(df)
    df = remove_unnecessary_columns(df)

    # Keep a human-readable copy for segmentation & visualization
    df_raw = df.copy()

    # Encode categorical columns into numbers
    df_encoded, encoders = encode_categorical_features(df)

    # Identify feature columns (everything except the target)
    feature_columns = [c for c in df_encoded.columns if c != TARGET_COLUMN]

    # Scale numeric features (helps Logistic Regression converge better)
    df_encoded, scaler = scale_numerical_features(df_encoded, feature_columns)

    print("Preprocessing complete.")
    print(f"Final feature columns ({len(feature_columns)}): {feature_columns}")

    return df_raw, df_encoded, encoders, scaler, feature_columns
