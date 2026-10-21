"""
Data loading and alignment utilities for the E.ON Day-Ahead Forecasting Challenge.

Handles loading of:
- 15-minute smart meter consumption records (kWh)
- Hourly weather observations from regional stations
- Household metadata and survey responses
- Resampling and alignment to standard day-ahead procurement resolution (hourly).
"""

import os
import glob
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


class DataLoader:
    """Robust data loader and alignment pipeline for smart meter and weather data."""

    def __init__(self, data_dir: str):
        self.data_dir = os.path.abspath(data_dir)
        self.sm_dir = os.path.join(self.data_dir, "15min")
        self.meta_dir = os.path.join(self.data_dir, "smart_meter_meta_data")
        self.weather_dir = os.path.join(self.data_dir, "weather_data_hourly")

        self.households_meta: Optional[pd.DataFrame] = None
        self.survey_meta: Optional[pd.DataFrame] = None
        self.weather_cache: Dict[str, pd.DataFrame] = {}

    def load_metadata(self) -> pd.DataFrame:
        """Load households overview and survey responses, merged into a unified metadata table."""
        hh_path = os.path.join(self.meta_dir, "households.csv")
        survey_path = os.path.join(self.meta_dir, "meta_data.csv")

        hh_df = pd.read_csv(hh_path, sep=";")
        hh_df["Household_ID"] = hh_df["Household_ID"].astype(str)

        if os.path.exists(survey_path):
            survey_df = pd.read_csv(survey_path, sep=";")
            survey_df["Household_ID"] = survey_df["Household_ID"].astype(str)
            merged = pd.merge(hh_df, survey_df, on="Household_ID", how="left")
        else:
            merged = hh_df

        self.households_meta = merged
        return merged

    def load_weather_station(self, weather_id: str) -> pd.DataFrame:
        """Load and prepare hourly weather records for a specific weather station."""
        if weather_id in self.weather_cache:
            return self.weather_cache[weather_id]

        w_path = os.path.join(self.weather_dir, f"{weather_id}.csv")
        if not os.path.exists(w_path):
            raise FileNotFoundError(f"Weather file not found for station {weather_id}: {w_path}")

        df = pd.read_csv(w_path, sep=";")
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True)
        df = df.sort_values("Timestamp").drop_duplicates(subset=["Timestamp"])
        df = df.set_index("Timestamp")

        # Select relevant numeric weather features
        numeric_cols = [
            "Temperature_avg_hourly",
            "DewPoint_hourly",
            "Humidity_avg_hourly",
            "Precipitation_total_hourly",
            "WindSpeed_hourly",
            "Sunshine_duration_hourly",
        ]
        available_cols = [c for c in numeric_cols if c in df.columns]
        w_numeric = df[available_cols].copy()

        # Forward/backward fill small missing spans, fill remaining sunshine with 0
        w_numeric = w_numeric.interpolate(method="time", limit=6).bfill().ffill()
        if "Sunshine_duration_hourly" in w_numeric.columns:
            w_numeric["Sunshine_duration_hourly"] = w_numeric["Sunshine_duration_hourly"].fillna(0.0)
        else:
            w_numeric["Sunshine_duration_hourly"] = 0.0

        self.weather_cache[weather_id] = w_numeric
        return w_numeric

    def load_household_timeseries(
        self, household_id: Union[str, int], resolution: str = "1h"
    ) -> pd.DataFrame:
        """
        Load smart meter readings for a household and resample to specified resolution.
        
        Args:
            household_id: Household ID string or int.
            resolution: Target resampling resolution ('1h' for hourly Day-Ahead procurement, or '15min').
            
        Returns:
            DataFrame indexed by UTC Timestamp with columns:
            ['Household_ID', 'kWh_received_Total', 'kWh_received_HeatPump', 'kWh_received_Other']
        """
        hid = str(household_id)
        fpath = os.path.join(self.sm_dir, f"{hid}.csv")
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Household smart meter file not found: {fpath}")

        df = pd.read_csv(fpath, sep=";")
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True)
        df = df.sort_values("Timestamp").drop_duplicates(subset=["Timestamp"])
        df = df.set_index("Timestamp")

        # Keep primary consumption columns
        val_cols = ["kWh_received_Total"]
        for c in ["kWh_received_HeatPump", "kWh_received_Other"]:
            if c in df.columns:
                val_cols.append(c)

        ts_data = df[val_cols].copy()

        if resolution.lower() in ["1h", "1hour", "h"]:
            # Aggregate 15-minute readings to hourly sums (total kWh per hour)
            resampled = ts_data.resample("1h").sum(min_count=3)
            # Interpolate occasional single-hour missing gaps
            resampled = resampled.interpolate(method="time", limit=3)
        elif resolution.lower() in ["15min", "15m"]:
            resampled = ts_data.resample("15min").interpolate(method="time", limit=2)
        else:
            raise ValueError(f"Unsupported resolution: {resolution}. Use '1h' or '15min'.")

        resampled["Household_ID"] = hid
        return resampled

    def load_aligned_dataset(
        self, household_ids: Optional[List[str]] = None, max_households: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Load and align multiple households with their corresponding weather stations.
        
        Returns:
            Unified pandas DataFrame with joined smart meter and weather features.
        """
        if self.households_meta is None:
            self.load_metadata()

        meta = self.households_meta.copy()
        if household_ids is not None:
            meta = meta[meta["Household_ID"].isin([str(h) for h in household_ids])]
        elif max_households is not None:
            meta = meta.head(max_households)

        records = []
        for _, row in meta.iterrows():
            hid = str(row["Household_ID"])
            weather_id = str(row["Weather_ID"])
            try:
                ts_df = self.load_household_timeseries(hid, resolution="1h")
                w_df = self.load_weather_station(weather_id)

                # Inner join on common hourly timestamps
                aligned = ts_df.join(w_df, how="inner")
                aligned["Weather_ID"] = weather_id
                aligned["Installation_HasPVSystem"] = row.get("Installation_HasPVSystem", np.nan)
                aligned["Survey_Building_LivingArea"] = row.get("Survey_Building_LivingArea", np.nan)
                aligned["Survey_Building_Residents"] = row.get("Survey_Building_Residents", np.nan)
                aligned["Survey_HeatPump_Installation_Type"] = row.get("Survey_HeatPump_Installation_Type", "unknown")
                records.append(aligned.reset_index())
            except Exception as e:
                continue

        if not records:
            raise ValueError("No household data could be successfully loaded and aligned.")

        combined = pd.concat(records, ignore_index=True)
        return combined
