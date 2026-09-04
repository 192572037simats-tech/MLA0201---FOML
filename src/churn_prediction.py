"""
Customer Churn Prediction Module
---------------------------------
Trains multiple classification models to predict customer churn,
evaluates them, selects the best-performing model, and saves it.
"""

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

TARGET_COLUMN = "Churn"
RANDOM_STATE = 42


def split_features_target(df_encoded: pd.DataFrame, feature_columns):
    """Split the dataframe into features (X) and target (y)."""
    X = df_encoded[feature_columns]
    y = df_encoded[TARGET_COLUMN]
    return X, y


def get_models():
    """Return a dictionary of the models to be trained and compared."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
    }


def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """
    Train each model, evaluate it on the test set, and collect
    performance metrics for comparison.
    Returns a results dictionary and a dictionary of trained model objects.
    """
    models = get_models()
    results = {}
    trained_models = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        metrics = {
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(y_test, predictions, zero_division=0),
            "Recall": recall_score(y_test, predictions, zero_division=0),
            "F1 Score": f1_score(y_test, predictions, zero_division=0),
            "Confusion Matrix": confusion_matrix(y_test, predictions),
        }

        results[name] = metrics
        trained_models[name] = model

        print(f"{name} Results:")
        print(f"  Accuracy : {metrics['Accuracy']:.4f}")
        print(f"  Precision: {metrics['Precision']:.4f}")
        print(f"  Recall   : {metrics['Recall']:.4f}")
        print(f"  F1 Score : {metrics['F1 Score']:.4f}")

    return results, trained_models


def select_best_model(results: dict, trained_models: dict):
    """
    Compare all trained models using F1 Score (a balanced metric for
    imbalanced classification problems like churn prediction) and
    return the name and object of the best-performing model.
    """
    best_name = max(results, key=lambda name: results[name]["F1 Score"])
    best_model = trained_models[best_name]
    print(f"\nBest performing model: {best_name} (F1 Score: {results[best_name]['F1 Score']:.4f})")
    return best_name, best_model


def save_model(model, model_name: str, output_dir: str = "models"):
    """Save the trained model to disk using joblib."""
    os.makedirs(output_dir, exist_ok=True)
    safe_name = model_name.lower().replace(" ", "_")
    model_path = os.path.join(output_dir, f"best_model_{safe_name}.joblib")
    joblib.dump(model, model_path)
    print(f"Best model saved to: {model_path}")
    return model_path


def run_churn_prediction(df_encoded: pd.DataFrame, feature_columns, models_dir="models"):
    """
    Full churn prediction pipeline:
    split data -> train models -> evaluate -> select best -> save best model.
    """
    X, y = split_features_target(df_encoded, feature_columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    results, trained_models = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    best_name, best_model = select_best_model(results, trained_models)
    save_model(best_model, best_name, models_dir)

    return results, trained_models, best_name, best_model, (X_test, y_test)
