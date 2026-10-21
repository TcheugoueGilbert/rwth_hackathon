"""
Feature Engineering Pipeline for Day-Ahead Electricity Procurement Forecasting.

Ensures strict day-ahead temporal causality:
- Smart-meter autoregressive lags are strictly >= 24 hours (e.g. t-24h, t-48h, t-168h).
- Weather variables represent Day-Ahead numerical weather forecasts (NWP).
- Physics-informed thermodynamic transformations (Heating Degree Hours, thermal inertia).
- Solar interaction terms capturing prosumer net-metering dynamics.
"""

from typing import List, Optional
import numpy as np
import pandas as pd


class FeaturePipeline:
    """Builds features for Day-Ahead electricity forecasting."""

    def __init__(self, t_base_heating: float = 15.5):
        self.t_base_heating = t_base_heating
        self.feature_columns_: List[str] = []

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform aligned raw dataframe into a feature matrix.
        
        Expects columns:
        - Timestamp (UTC datetime or string)
        - Household_ID
        - kWh_received_Total
        - Temperature_avg_hourly
        - (optional) Humidity_avg_hourly, Sunshine_duration_hourly, WindSpeed_hourly, DewPoint_hourly
        - (optional) PV_Resolved_Flag or Installation_HasPVSystem
        """
        data = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(data["Timestamp"]):
            data["Timestamp"] = pd.to_datetime(data["Timestamp"], utc=True)

        data = data.sort_values(["Household_ID", "Timestamp"]).reset_index(drop=True)

        # 1. Calendar & Cyclical Features
        dt = data["Timestamp"].dt
        data["hour"] = dt.hour
        data["dayofweek"] = dt.dayofweek
        data["month"] = dt.month
        data["is_weekend"] = (data["dayofweek"] >= 5).astype(int)

        # Trigonometric cyclical encodings
        data["hour_sin"] = np.sin(2 * np.pi * data["hour"] / 24.0)
        data["hour_cos"] = np.cos(2 * np.pi * data["hour"] / 24.0)
        data["dow_sin"] = np.sin(2 * np.pi * data["dayofweek"] / 7.0)
        data["dow_cos"] = np.cos(2 * np.pi * data["dayofweek"] / 7.0)
        data["month_sin"] = np.sin(2 * np.pi * data["month"] / 12.0)
        data["month_cos"] = np.cos(2 * np.pi * data["month"] / 12.0)

        # 2. Physics-Informed Thermodynamic Features (Heat Pump Behavior)
        temp = data["Temperature_avg_hourly"].fillna(data["Temperature_avg_hourly"].median())
        data["temp"] = temp
        data["temp_sq"] = temp ** 2
        # Heating Degree Hours (HDH) hinge function
        data["hdh"] = np.maximum(0.0, self.t_base_heating - temp)

        # Wind & Humidity interactions
        wind = data["WindSpeed_hourly"].fillna(data["WindSpeed_hourly"].median()) if "WindSpeed_hourly" in data.columns else 2.0
        humidity = data["Humidity_avg_hourly"].fillna(75.0) if "Humidity_avg_hourly" in data.columns else 75.0
        data["wind_speed"] = wind
        data["humidity"] = humidity
        data["wind_chill_interaction"] = data["hdh"] * wind

        # 3. Solar & Prosumer Interaction Features
        sunshine = data["Sunshine_duration_hourly"].fillna(0.0) if "Sunshine_duration_hourly" in data.columns else 0.0
        data["sunshine"] = sunshine

        # Daytime indicator (between 7:00 and 19:00 UTC)
        data["is_daylight"] = ((data["hour"] >= 7) & (data["hour"] <= 19)).astype(int)
        data["sunshine_daylight"] = sunshine * data["is_daylight"]

        # Prosumer flag
        if "PV_Resolved_Flag" in data.columns:
            pv_flag = data["PV_Resolved_Flag"].astype(int)
        elif "Installation_HasPVSystem" in data.columns:
            pv_flag = data["Installation_HasPVSystem"].fillna(False).astype(int)
        else:
            pv_flag = pd.Series(0, index=data.index)
        data["is_pv_prosumer"] = pv_flag

        # Prosumer solar interaction (self-consumption duck curve effect)
        data["pv_sunshine_interaction"] = pv_flag * data["sunshine"]
        data["pv_daylight_interaction"] = pv_flag * data["is_daylight"]

        # 4. Strictly Causal Day-Ahead Autoregressive Lags (per household)
        # Using groupby on Household_ID with shift >= 24
        grouped_target = data.groupby("Household_ID")["kWh_received_Total"]
        data["lag_24h"] = grouped_target.shift(24)
        data["lag_48h"] = grouped_target.shift(48)
        data["lag_168h"] = grouped_target.shift(168)  # same hour last week

        # Prior-day rolling statistics (shifted by 24h to avoid lookahead)
        rolled_mean = grouped_target.shift(24).rolling(window=24, min_periods=12).mean()
        rolled_std = grouped_target.shift(24).rolling(window=24, min_periods=12).std()
        data["prior_day_mean"] = rolled_mean.reset_index(level=0, drop=True)
        data["prior_day_std"] = rolled_std.reset_index(level=0, drop=True).fillna(0.0)

        # 5. Metadata features if present
        if "Survey_Building_LivingArea" in data.columns:
            data["living_area"] = data["Survey_Building_LivingArea"].fillna(150.0)
        else:
            data["living_area"] = 150.0

        if "Survey_Building_Residents" in data.columns:
            data["residents"] = data["Survey_Building_Residents"].fillna(2.0)
        else:
            data["residents"] = 2.0

        # Define full feature set
        self.feature_columns_ = [
            "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
            "is_weekend", "temp", "temp_sq", "hdh", "wind_speed", "humidity",
            "wind_chill_interaction", "sunshine", "is_daylight", "sunshine_daylight",
            "is_pv_prosumer", "pv_sunshine_interaction", "pv_daylight_interaction",
            "lag_24h", "lag_48h", "lag_168h", "prior_day_mean", "prior_day_std",
            "living_area", "residents"
        ]

        # Drop rows where critical 168h lags are not yet available (initial warm-up)
        # and fill remaining small NaNs with column medians
        clean_data = data.dropna(subset=["lag_168h", "kWh_received_Total"]).copy()
        for col in self.feature_columns_:
            if clean_data[col].isnull().any():
                clean_data[col] = clean_data[col].fillna(clean_data[col].median())

        return clean_data
