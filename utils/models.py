"""
Forecasting Models for the E.ON Day-Ahead Procurement Challenge.

Includes:
- Level 0 Baselines:
    1. Seasonal Persistence (same hour last week / t-168)
    2. Regularized Ridge Regression
- Level 1 Advanced Models:
    1. Global LightGBM Regressor
    2. Stratified Prosumer Forecaster (specialized models for Heat-Pump vs PV Prosumers)
- Level 3 Uncertainty Models:
    1. Quantile LightGBM Regressors for probabilistic day-ahead bidding and risk quantification.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb


class SeasonalPersistenceBaseline:
    """Level 0 Baseline: Day-ahead seasonal persistence (t-168h with t-24h fallback)."""

    def fit(self, X: pd.DataFrame, y: pd.Series):
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if "lag_168h" in X.columns:
            preds = X["lag_168h"].values.copy()
            # If 168h is NaN, fallback to 24h
            if "lag_24h" in X.columns:
                nan_mask = np.isnan(preds)
                preds[nan_mask] = X["lag_24h"].values[nan_mask]
            return np.nan_to_num(preds, nan=0.5)
        elif "lag_24h" in X.columns:
            return np.nan_to_num(X["lag_24h"].values, nan=0.5)
        else:
            raise ValueError("Required lag features (lag_168h or lag_24h) missing from input.")


class RidgeBaseline:
    """Level 0 Baseline: Regularized Linear Model with feature scaling."""

    def __init__(self, alpha: float = 10.0):
        self.alpha = alpha
        self.scaler = StandardScaler()
        self.model = Ridge(alpha=self.alpha)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        preds = self.model.predict(X_scaled)
        # Net load cannot be negative in this smart-meter dataset
        return np.maximum(0.0, preds)


class GlobalLightGBMForecaster:
    """Level 1: Gradient Boosted Trees trained on full portfolio."""

    def __init__(
        self,
        n_estimators: int = 250,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        num_leaves: int = 31,
        random_state: int = 42,
    ):
        self.params = {
            "objective": "regression_l1",  # MAE loss aligns with procurement baseload
            "metric": "l1",
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "num_leaves": num_leaves,
            "random_state": random_state,
            "verbose": -1,
            "n_jobs": -1,
        }
        self.model = lgb.LGBMRegressor(**self.params)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = self.model.predict(X)
        return np.maximum(0.0, preds)


class StratifiedForecaster:
    """
    Level 1: Stratified Forecaster.
    Maintains separate specialized models for:
    - Pure Heat-Pump households
    - PV Prosumer households (Heat-Pump + PV)
    """

    def __init__(self, random_state: int = 42):
        self.model_heatpump = lgb.LGBMRegressor(
            objective="regression_l1",
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
            random_state=random_state,
            verbose=-1,
            n_jobs=-1,
        )
        self.model_prosumer = lgb.LGBMRegressor(
            objective="regression_l1",
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
            random_state=random_state,
            verbose=-1,
            n_jobs=-1,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series, is_prosumer: pd.Series):
        mask_pv = is_prosumer.astype(bool).values
        mask_hp = ~mask_pv

        if mask_hp.sum() > 0:
            self.model_heatpump.fit(X[mask_hp], y[mask_hp])
        if mask_pv.sum() > 0:
            self.model_prosumer.fit(X[mask_pv], y[mask_pv])
        return self

    def predict(self, X: pd.DataFrame, is_prosumer: pd.Series) -> np.ndarray:
        mask_pv = is_prosumer.astype(bool).values
        mask_hp = ~mask_pv

        preds = np.zeros(len(X), dtype=float)
        if mask_hp.sum() > 0:
            preds[mask_hp] = self.model_heatpump.predict(X[mask_hp])
        if mask_pv.sum() > 0:
            preds[mask_pv] = self.model_prosumer.predict(X[mask_pv])

        return np.maximum(0.0, preds)


class QuantileDayAheadForecaster:
    """
    Level 3: Probabilistic Forecasting via Quantile Gradient Boosted Trees.
    Models arbitrary quantiles tau in (0, 1) to quantify uncertainty and enable
    cost-optimal asymmetric bidding.
    """

    def __init__(
        self,
        quantiles: List[float] = [0.10, 0.25, 0.50, 0.70, 0.75, 0.90],
        random_state: int = 42,
    ):
        self.quantiles = sorted(quantiles)
        self.models: Dict[float, lgb.LGBMRegressor] = {}
        for q in self.quantiles:
            self.models[q] = lgb.LGBMRegressor(
                objective="quantile",
                alpha=q,
                n_estimators=180,
                learning_rate=0.06,
                max_depth=5,
                num_leaves=25,
                random_state=random_state,
                verbose=-1,
                n_jobs=-1,
            )

    def fit(self, X: pd.DataFrame, y: pd.Series):
        for q, model in self.models.items():
            model.fit(X, y)
        return self

    def predict_quantiles(self, X: pd.DataFrame) -> Dict[float, np.ndarray]:
        preds = {}
        for q, model in self.models.items():
            preds[q] = np.maximum(0.0, model.predict(X))
        return preds

    def predict_interval(self, X: pd.DataFrame, lower_q: float = 0.10, upper_q: float = 0.90) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns (lower_bound, median_forecast, upper_bound).
        """
        all_preds = self.predict_quantiles(X)
        lower = all_preds[lower_q]
        median = all_preds[0.50]
        upper = all_preds[upper_q]
        # Ensure monotonic interval
        upper = np.maximum(upper, lower)
        return lower, median, upper
