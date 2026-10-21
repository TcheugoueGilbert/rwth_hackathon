from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


def _read_weather_data(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    weather_dir = data_dir / "weather_data_hourly"
    frames: list[pd.DataFrame] = []

    for weather_file in sorted(weather_dir.glob("*.csv")):
        frame = pd.read_csv(weather_file, sep=";")
        if frame.empty:
            continue
        frame["Timestamp"] = pd.to_datetime(frame["Timestamp"], utc=True)
        frame["Weather_ID"] = weather_file.stem
        frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=["Weather_ID", "Timestamp"])

    weather = pd.concat(frames, ignore_index=True)
    weather = weather.rename(columns={"Timestamp": "WeatherTimestamp"})
    weather["WeatherHour"] = weather["WeatherTimestamp"].dt.floor("h")
    return weather


def load_energy_data(
    limit_households: int | None = None,
    data_dir: Path | str = DATA_DIR,
) -> pd.DataFrame:
    """Load and merge the household consumption data with the weather station data.

    The method keeps the temporal ordering intact and enforces a strict past-to-future
    split in downstream forecasting tasks. When limit_households is provided, only the first
    households in alphabetical order are loaded.
    """
    data_dir = Path(data_dir)
    household_dir = data_dir / "smart_meter_meta_data"
    households = pd.read_csv(household_dir / "households.csv", sep=";")
    households["Household_ID"] = households["Household_ID"].astype(int)
    weather = _read_weather_data(data_dir)

    household_files = sorted((data_dir / "15min").glob("*.csv"))
    if limit_households is not None:
        household_files = household_files[:limit_households]

    frames: list[pd.DataFrame] = []
    for file in household_files:
        df = pd.read_csv(file, sep=";")
        if df.empty:
            continue
        df["Household_ID"] = df["Household_ID"].astype(int)
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True)
        df["TimestampRoundHour"] = df["Timestamp"].dt.floor("h")
        df = df.merge(
            households[["Household_ID", "Weather_ID", "Installation_HasPVSystem"]],
            on="Household_ID",
            how="left",
        )

        if weather.empty:
            df["Temperature_avg_hourly"] = np.nan
            df["Humidity_avg_hourly"] = np.nan
            df["Precipitation_total_hourly"] = np.nan
            df["Sunshine_duration_hourly"] = np.nan
            df["WindSpeed_hourly"] = np.nan
        else:
            weather_for_household = weather[
                [
                    "Weather_ID",
                    "WeatherHour",
                    "Temperature_avg_hourly",
                    "Humidity_avg_hourly",
                    "Precipitation_total_hourly",
                    "Sunshine_duration_hourly",
                    "WindSpeed_hourly",
                ]
            ]
            df = df.merge(
                weather_for_household,
                left_on=["Weather_ID", "TimestampRoundHour"],
                right_on=["Weather_ID", "WeatherHour"],
                how="left",
            )
            df = df.drop(columns=["WeatherHour", "TimestampRoundHour"])

        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=["Household_ID", "Timestamp", "kWh_received_Total"])

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.sort_values("Timestamp").reset_index(drop=True)
    return merged


def build_day_ahead_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Prepare a daily day-ahead feature matrix with a strict backward-looking design.

    The target variable is the next day's total load for each household. All explanatory
    features are derived from the same household's past data and the weather linked to that
    household, which avoids leakage from the future.
    """
    if frame.empty:
        return frame.copy()

    working = frame.copy()
    working["Timestamp"] = pd.to_datetime(working["Timestamp"], utc=True)
    working["day"] = working["Timestamp"].dt.floor("D")

    daily = (
        working.groupby(["Household_ID", "day"], as_index=False)
        .agg(
            total_load=("kWh_received_Total", "sum"),
            average_temp=("Temperature_avg_hourly", "mean"),
            max_temp=("Temperature_avg_hourly", "max"),
            avg_humidity=("Humidity_avg_hourly", "mean"),
            sunshine=("Sunshine_duration_hourly", "sum"),
            precipitation=("Precipitation_total_hourly", "sum"),
            wind=("WindSpeed_hourly", "mean"),
            pv_flag=("Installation_HasPVSystem", "first"),
        )
    )

    daily["hour"] = 12
    daily["dayofweek"] = daily["day"].dt.dayofweek
    daily["is_weekend"] = (daily["dayofweek"] >= 5).astype(int)
    daily["month"] = daily["day"].dt.month
    daily["sin_hour"] = np.sin(2 * np.pi * daily["hour"] / 24)
    daily["cos_hour"] = np.cos(2 * np.pi * daily["hour"] / 24)
    daily["day_of_year"] = daily["day"].dt.dayofyear
    daily["sin_day_of_year"] = np.sin(2 * np.pi * daily["day_of_year"] / 365)
    daily["cos_day_of_year"] = np.cos(2 * np.pi * daily["day_of_year"] / 365)

    for lag in (1, 2, 3, 7, 14):
        daily[f"lag_{lag}d"] = (
            daily.groupby("Household_ID")["total_load"].transform(lambda s: s.shift(lag))
        )

    daily["rolling_3d_mean"] = (
        daily.groupby("Household_ID")["total_load"]
        .transform(lambda s: s.shift(1).rolling(window=3, min_periods=1).mean())
    )
    daily["rolling_7d_mean"] = (
        daily.groupby("Household_ID")["total_load"]
        .transform(lambda s: s.shift(1).rolling(window=7, min_periods=1).mean())
    )

    daily["temp_anomaly"] = (
        daily["average_temp"]
        - daily.groupby("Household_ID")["average_temp"].transform("mean")
    )
    daily["target_24h"] = (
        daily.groupby("Household_ID")["total_load"].transform(lambda s: s.shift(-1))
    )
    daily["target_24h"] = daily["target_24h"].bfill().ffill()

    daily = daily.sort_values(["Household_ID", "day"]).reset_index(drop=True)
    return daily


def evaluate_forecast_model(
    frame: pd.DataFrame,
    random_state: int = 42,
) -> dict[str, float]:
    """Train a day-ahead model and return the key accuracy metrics."""
    feature_frame = build_day_ahead_features(frame)
    if feature_frame.empty:
        return {"mae": np.nan, "rmse": np.nan, "mape": np.nan, "r2": np.nan}

    feature_columns = [
        "average_temp",
        "max_temp",
        "avg_humidity",
        "sunshine",
        "precipitation",
        "wind",
        "hour",
        "dayofweek",
        "is_weekend",
        "month",
        "sin_hour",
        "cos_hour",
        "sin_day_of_year",
        "cos_day_of_year",
        "lag_1d",
        "lag_2d",
        "lag_3d",
        "lag_7d",
        "lag_14d",
        "rolling_3d_mean",
        "rolling_7d_mean",
        "temp_anomaly",
        "pv_flag",
    ]
    model_frame = feature_frame.dropna(subset=[*feature_columns, "target_24h"]).copy()
    if len(model_frame) < 20:
        raise ValueError("Not enough rows after feature construction for a reliable model fit.")

    model_frame = model_frame.sort_values("day").reset_index(drop=True)
    split_idx = max(1, int(len(model_frame) * 0.8))
    train = model_frame.iloc[:split_idx]
    test = model_frame.iloc[split_idx:]

    model = HistGradientBoostingRegressor(
        max_depth=6,
        learning_rate=0.05,
        max_iter=300,
        random_state=random_state,
    )
    model.fit(train[feature_columns], train["target_24h"])
    prediction = model.predict(test[feature_columns])

    y_true = test["target_24h"].to_numpy()
    y_pred = prediction
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    denom = np.abs(y_true)
    if np.any(denom != 0):
        relative_error = np.divide(
            np.abs(y_true - y_pred),
            denom,
            out=np.zeros_like(denom, dtype=float),
            where=denom != 0,
        )
        mape = np.mean(relative_error) * 100
    else:
        mape = np.nan
    r2 = r2_score(y_true, y_pred)

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": float(mape) if np.isfinite(mape) else np.nan,
        "r2": float(r2),
    }


def summarize_household_forecasts(
    household_ids: Iterable[int] | None = None,
    data_dir: Path | str = DATA_DIR,
) -> pd.DataFrame:
    """Evaluate the day-ahead forecast for one or more households and return a summary table."""
    data_dir = Path(data_dir)
    all_data = load_energy_data(data_dir=data_dir)
    if all_data.empty:
        return pd.DataFrame(columns=["Household_ID", "mae", "rmse", "mape", "r2"])

    valid_ids = all_data["Household_ID"].dropna().astype(int).unique()
    selected_ids = list(valid_ids) if household_ids is None else [int(h) for h in household_ids]
    selected_ids = [int(h) for h in selected_ids if int(h) in valid_ids]

    rows: list[dict[str, float | int | str]] = []
    for household_id in selected_ids:
        current = all_data[all_data["Household_ID"] == household_id].copy()
        metrics = evaluate_forecast_model(current)
        rows.append({"Household_ID": int(household_id), **metrics})

    return pd.DataFrame(rows)


if __name__ == "__main__":
    summary = summarize_household_forecasts(data_dir=DATA_DIR)
    if summary.empty:
        print("No forecast data loaded from the project data folder.")
    else:
        print(summary.head())
        print(f"Median MAE: {summary['mae'].median():.3f} kWh")
