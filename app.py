import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.data_utils import load_app_data, format_num, format_pct
from src.modeling_utils import load_xgb_model, predict_scenario
from src.decision_layer import build_decision_layer


st.set_page_config(
    page_title="Retail Demand Forecasting",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")


@st.cache_data
def load_cached_data():
    return load_app_data(DATA_DIR)


@st.cache_resource
def load_cached_model(family):
    return load_xgb_model(MODEL_DIR, family)


model_results, predictions, feature_importance, family_features = load_cached_data()
families = sorted(model_results["family"].dropna().unique())


st.title("Retail Demand Forecasting Command Center")
st.caption(
    "An interactive demand forecasting and inventory decision system for major grocery product families."
)

tab1, tab2, tab3, tab4 = st.tabs([
    "Command Center",
    "Forecast Lab",
    "Inventory Simulator",
    "Category Intelligence"
])


# -----------------------------
# TAB 1: COMMAND CENTER
# -----------------------------
with tab1:
    st.markdown("## Executive Overview")

    avg_improvement = model_results["mae_improvement_pct"].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Families Modeled", len(families))
    c2.metric("Best Model", "XGBoost")
    c3.metric("Avg MAE Improvement", format_pct(avg_improvement))
    c4.metric("Decision Layer", "Inventory Buffer")

    st.markdown("### System Workflow")
    st.write(
        "This project forecasts product-family demand and converts predictions into inventory recommendations. "
        "The workflow compares a naive baseline against XGBoost models using lag features, rolling averages, "
        "calendar effects, promotions, transactions, oil prices, and holiday indicators."
    )

    st.info(
        "For highly volatile categories like PRODUCE, the system uses a conservative shrinkage correction on top "
        "of the naive baseline to avoid overreacting to noisy demand spikes."
    )

    display_results = model_results.copy()
    display_results["Naive MAE"] = display_results["naive_mae"].map(format_num)
    display_results["XGBoost MAE"] = display_results["xgb_mae"].map(format_num)
    display_results["Naive RMSE"] = display_results["naive_rmse"].map(format_num)
    display_results["XGBoost RMSE"] = display_results["xgb_rmse"].map(format_num)
    display_results["MAE Improvement"] = display_results["mae_improvement_pct"].map(format_pct)
    display_results = display_results.rename(columns={"family": "Family"})

    st.markdown("### Model Performance Summary")
    st.dataframe(
        display_results[["Family", "Naive MAE", "XGBoost MAE", "MAE Improvement", "Naive RMSE", "XGBoost RMSE"]],
        use_container_width=True,
        hide_index=True
    )

    fig = px.bar(
        model_results,
        x="family",
        y="mae_improvement_pct",
        text="mae_improvement_pct",
        title="MAE Improvement Over Naive Baseline",
        labels={"family": "Product Family", "mae_improvement_pct": "Improvement (%)"}
    )
    fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    st.plotly_chart(fig, use_container_width=True, key="command_improvement_chart")


# -----------------------------
# TAB 2: FORECAST LAB
# -----------------------------
with tab2:
    st.markdown("## Forecast Lab")

    selected_family = st.selectbox("Select product family", families, key="forecast_family")

    fam_preds = predictions[predictions["family"] == selected_family].copy().sort_values("date")
    fam_results = model_results[model_results["family"] == selected_family].iloc[0]

    c1, c2, c3 = st.columns(3)
    c1.metric("Naive MAE", format_num(fam_results["naive_mae"]))
    c2.metric("XGBoost MAE", format_num(fam_results["xgb_mae"]))
    c3.metric("MAE Improvement", format_pct(fam_results["mae_improvement_pct"]))

    actual_df = fam_preds.dropna(subset=["date", "unit_sales"])
    xgb_df = fam_preds.dropna(subset=["date", "xgb_prediction"])
    naive_df = fam_preds.dropna(subset=["date", "naive_prediction"])

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=actual_df["date"],
        y=actual_df["unit_sales"],
        mode="lines",
        name="Actual Sales"
    ))

    fig.add_trace(go.Scatter(
        x=xgb_df["date"],
        y=xgb_df["xgb_prediction"],
        mode="lines",
        name="XGBoost Prediction"
    ))

    fig.add_trace(go.Scatter(
        x=naive_df["date"],
        y=naive_df["naive_prediction"],
        mode="lines",
        name="Naive Prediction",
        line=dict(dash="dash", width=3)
    ))

    fig.update_layout(
        title=f"Actual vs Forecasted Demand: {selected_family}",
        xaxis_title="Date",
        yaxis_title="Unit Sales",
        height=540
    )

    st.plotly_chart(fig, use_container_width=True, key=f"forecast_plot_{selected_family}")

    fam_preds["absolute_error"] = abs(fam_preds["unit_sales"] - fam_preds["xgb_prediction"])

    fig_err = px.line(
        fam_preds.dropna(subset=["absolute_error"]),
        x="date",
        y="absolute_error",
        title=f"Absolute Forecast Error Over Time: {selected_family}",
        labels={"absolute_error": "Absolute Error", "date": "Date"}
    )
    st.plotly_chart(fig_err, use_container_width=True, key=f"error_plot_{selected_family}")


# -----------------------------
# TAB 3: INVENTORY SIMULATOR
# -----------------------------
with tab3:
    st.markdown("## Inventory Simulator")

    st.write(
        "Select a family and date, then adjust business inputs to see how forecasted demand and recommended "
        "inventory levels change."
    )

    selected_family = st.selectbox("Select product family", families, key="sim_family")

    fam_features = family_features[family_features["family"] == selected_family].copy()
    available_dates = sorted(fam_features["date"].dt.date.unique())

    selected_date = st.selectbox("Select forecast date", available_dates, key="sim_date")
    row = fam_features[fam_features["date"].dt.date == selected_date].iloc[0]

    model = load_cached_model(selected_family)

    left, right = st.columns([1, 1])

    with left:
        st.markdown("### Scenario Inputs")

        promo_count = st.slider(
            "Promotion Count",
            min_value=0,
            max_value=int(max(fam_features["promo_count"].max(), 1)),
            value=int(row["promo_count"]),
            key="promo_slider"
        )

        transactions = st.slider(
            "Expected Transactions",
            min_value=0,
            max_value=int(max(fam_features["transactions"].max(), 1)),
            value=int(row["transactions"]),
            key="transactions_slider"
        )

        oil_price = st.slider(
            "Oil Price",
            min_value=float(fam_features["oil_price"].min()),
            max_value=float(fam_features["oil_price"].max()),
            value=float(row["oil_price"]),
            key="oil_slider"
        )

        is_holiday = st.checkbox(
            "Holiday / Event Day",
            value=bool(row["is_holiday"]),
            key="holiday_checkbox"
        )

        strategy = st.radio(
            "Inventory Strategy",
            ["Conservative", "Balanced", "Aggressive"],
            horizontal=True,
            key="strategy_radio"
        )

    predicted_sales = predict_scenario(
        model=model,
        row=row,
        family=selected_family,
        promo_count=promo_count,
        transactions=transactions,
        oil_price=oil_price,
        is_holiday=is_holiday
    )

    recommended_stock, inventory_buffer, risk_level, recommendation, volatility_ratio = build_decision_layer(
        predicted_sales=predicted_sales,
        rolling_mean_14=row["rolling_mean_14"],
        rolling_std_14=row["rolling_std_14"],
        strategy=strategy
    )

    with right:
        st.markdown("### Forecast Output")

        c1, c2 = st.columns(2)
        c1.metric("Predicted Unit Sales", format_num(predicted_sales))
        c2.metric("Recommended Stock", format_num(recommended_stock))

        c3, c4 = st.columns(2)
        c3.metric("Inventory Buffer", format_num(inventory_buffer))
        c4.metric("Risk Level", risk_level)

        st.success(recommendation)

        decision_chart = pd.DataFrame({
            "Metric": ["Predicted Demand", "Recommended Stock"],
            "Units": [predicted_sales, recommended_stock]
        })

        fig = px.bar(
            decision_chart,
            x="Metric",
            y="Units",
            title="Predicted Demand vs Recommended Stock Level",
            text="Units"
        )
        fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
        fig.update_layout(yaxis_title="Units", xaxis_title="")
        st.plotly_chart(fig, use_container_width=True, key=f"decision_chart_{selected_family}_{selected_date}")

        st.caption(
            f"Volatility ratio used for decision layer: {volatility_ratio:.3f}. "
            "Higher volatility increases the recommended inventory buffer."
        )


# -----------------------------
# TAB 4: CATEGORY INTELLIGENCE
# -----------------------------
with tab4:
    st.markdown("## Category Intelligence")

    selected_family = st.selectbox("Select product family", families, key="insights_family")

    st.markdown("### Family Comparison")

    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(
            model_results,
            x="family",
            y=["naive_mae", "xgb_mae"],
            barmode="group",
            title="Naive vs XGBoost MAE by Product Family",
            labels={"value": "MAE", "family": "Product Family", "variable": "Model"}
        )
        st.plotly_chart(fig, use_container_width=True, key="category_mae_comparison")

    with col2:
        fig = px.bar(
            model_results,
            x="family",
            y="mae_improvement_pct",
            title="MAE Improvement Over Naive Baseline",
            labels={"mae_improvement_pct": "Improvement (%)", "family": "Product Family"},
            text="mae_improvement_pct"
        )
        fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
        st.plotly_chart(fig, use_container_width=True, key="category_improvement_chart")

    st.markdown("### Feature Importance")

    fam_importance = (
        feature_importance[feature_importance["family"] == selected_family]
        .sort_values("importance", ascending=False)
        .head(15)
    )

    fig = px.bar(
        fam_importance.sort_values("importance"),
        x="importance",
        y="feature",
        orientation="h",
        title=f"Top Feature Importances: {selected_family}",
        labels={"importance": "Importance", "feature": "Feature"}
    )
    st.plotly_chart(fig, use_container_width=True, key=f"feature_importance_{selected_family}")

    st.markdown("### How the Model Uses Past Data")
    st.info(
        """
**Lag features** allow the model to learn from past demand patterns.

For example:
- **lag_1** = sales from yesterday
- **lag_7** = sales from the same day last week
- **lag_28** = sales from roughly one month ago

These help the model capture weekly shopping cycles, recurring demand patterns, and short-term trends.

Rolling averages, such as `rolling_mean_14`, summarize recent demand behavior to smooth noise and capture trends.
"""
    )

    fam_features = family_features[family_features["family"] == selected_family].copy()

    day_pattern = (
        fam_features.groupby("day_of_week", as_index=False)["unit_sales"]
        .mean()
    )

    day_labels = {
        0: "Mon",
        1: "Tue",
        2: "Wed",
        3: "Thu",
        4: "Fri",
        5: "Sat",
        6: "Sun"
    }
    day_pattern["day"] = day_pattern["day_of_week"].map(day_labels)

    fig = px.line(
        day_pattern,
        x="day",
        y="unit_sales",
        markers=True,
        title=f"Average Sales by Day of Week: {selected_family}",
        labels={"unit_sales": "Average Unit Sales", "day": "Day"}
    )
    st.plotly_chart(fig, use_container_width=True, key=f"day_pattern_{selected_family}")

    st.markdown("### Business Interpretation")
    st.write(
        "Feature importance and day-of-week patterns help explain what drives demand. "
        "For many retail categories, weekly shopping cycles and overall transaction activity are major predictors. "
        "More volatile families may require conservative adjustments and larger inventory buffers."
    )