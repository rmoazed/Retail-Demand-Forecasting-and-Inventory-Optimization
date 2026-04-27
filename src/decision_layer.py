def build_decision_layer(predicted_sales, rolling_mean_14, rolling_std_14, strategy):
    volatility_ratio = 0 if rolling_mean_14 == 0 else rolling_std_14 / rolling_mean_14

    if volatility_ratio >= 0.25:
        risk_level = "High"
        base_factor = 1.5
    elif volatility_ratio >= 0.12:
        risk_level = "Medium"
        base_factor = 1.0
    else:
        risk_level = "Low"
        base_factor = 0.5

    strategy_multiplier = {
        "Conservative": 1.25,
        "Balanced": 1.0,
        "Aggressive": 0.75
    }

    safety_factor = base_factor * strategy_multiplier[strategy]
    inventory_buffer = predicted_sales * volatility_ratio * safety_factor
    recommended_stock = predicted_sales + inventory_buffer

    if risk_level == "High":
        recommendation = "Increase inventory buffer due to elevated demand volatility."
    elif risk_level == "Medium":
        recommendation = "Maintain moderate safety stock to reduce stockout risk."
    else:
        recommendation = "Standard inventory level is likely sufficient."

    return recommended_stock, inventory_buffer, risk_level, recommendation, volatility_ratio