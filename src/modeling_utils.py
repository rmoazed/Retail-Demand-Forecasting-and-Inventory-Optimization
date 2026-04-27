
import os
import joblib
import numpy as np
import pandas as pd

PRODUCE_ALPHA = 0.01


def load_xgb_model(model_dir, family):
    safe_name = family.lower().replace(" ", "_")
    model_path = os.path.join(model_dir, f"xgb_{safe_name}.joblib")
    return joblib.load(model_path)


def predict_scenario(model, row, family, promo_count, transactions, oil_price, is_holiday):
    scenario = row.copy()

    scenario["promo_count"] = promo_count
    scenario["transactions"] = transactions
    scenario["oil_price"] = oil_price
    scenario["is_holiday"] = int(is_holiday)

    model_features = model.get_booster().feature_names
    X = pd.DataFrame([scenario[model_features]])

    try:
        if family == "PRODUCE":
            delta_pred = model.predict(X)[0]
            pred = scenario["lag_1"] + PRODUCE_ALPHA * delta_pred
        else:
            pred_log = model.predict(X)[0]
            pred = np.expm1(pred_log)

        if not np.isfinite(pred) or pred <= 0:
            pred = scenario["lag_1"]

        max_reasonable_sales = max(
            scenario["lag_1"] * 3,
            scenario["rolling_mean_14"] * 3,
            1
        )

        if pred > max_reasonable_sales:
            pred = max_reasonable_sales

    except Exception:
        pred = scenario["lag_1"]

    return max(float(pred), 0)
