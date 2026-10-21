import pandas as pd

from utils.energy_forecast import (
    build_day_ahead_features,
    evaluate_forecast_model,
    load_energy_data,
)


def test_load_energy_data_returns_merged_household_and_weather_frame():
    df = load_energy_data(limit_households=2)

    assert isinstance(df, pd.DataFrame)
    assert {"Household_ID", "Timestamp", "kWh_received_Total"}.issubset(df.columns)
    assert len(df) > 0
    assert df["Timestamp"].is_monotonic_increasing


def test_build_day_ahead_features_creates_time_features_and_targets():
    df = load_energy_data(limit_households=1)
    features = build_day_ahead_features(df)

    assert {"target_24h", "hour", "dayofweek", "is_weekend"}.issubset(features.columns)
    assert features["target_24h"].notna().all()
    assert features.shape[0] > 0


def test_evaluate_forecast_model_returns_reasonable_metrics():
    df = load_energy_data(limit_households=1)
    metrics = evaluate_forecast_model(df)

    assert {"mae", "rmse", "mape", "r2"}.issubset(metrics)
    assert metrics["mae"] >= 0
    assert metrics["rmse"] >= 0
    assert metrics["r2"] <= 1.0
