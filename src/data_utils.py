import os
import pandas as pd


def load_app_data(data_dir):
    model_results = pd.read_csv(os.path.join(data_dir, "model_results.csv"))
    predictions = pd.read_csv(os.path.join(data_dir, "predictions_by_family.csv"))
    feature_importance = pd.read_csv(os.path.join(data_dir, "feature_importance_by_family.csv"))
    family_features = pd.read_csv(os.path.join(data_dir, "family_features.csv"))

    predictions["date"] = pd.to_datetime(predictions["date"], format="mixed")
    family_features["date"] = pd.to_datetime(family_features["date"], format="mixed")

    if "lag_diff_1" not in family_features.columns:
        family_features["lag_diff_1"] = family_features["lag_1"] - family_features["lag_7"]

    return model_results, predictions, feature_importance, family_features


def format_num(x):
    return f"{x:,.0f}"


def format_pct(x):
    return f"{x:.2f}%"