"""
Evaluation Metrics & Market Procurement Economics.

Implements:
1. Standard statistical time-series metrics: MAE, RMSE, WAPE (%), R²
2. Day-Ahead Procurement Imbalance Cost Model:
   - Asymmetric balancing market settlement (shortage penalty vs surplus discount)
   - Evaluates portfolio economic exposure under EPEX SPOT conditions
3. Uncertainty & Probabilistic Metrics:
   - Pinball (Check) Loss
   - Prediction Interval Coverage Probability (PICP)
   - Mean Prediction Interval Width (MPIW)
   - Cost-optimal bidding quantile derivation
"""

from typing import Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class ProcurementEvaluator:
    """Evaluates forecasts under both statistical and day-ahead market financial criteria."""

    def __init__(
        self,
        c_short_eur_per_mwh: float = 35.0,  # Penalty cost for under-procurement (buying balancing energy)
        c_long_eur_per_mwh: float = 15.0,   # Penalty cost for over-procurement (dumping surplus)
    ):
        self.c_short = c_short_eur_per_mwh
        self.c_long = c_long_eur_per_mwh
        # Cost-optimal procurement quantile under asymmetric loss
        self.optimal_tau = self.c_short / (self.c_short + self.c_long)

    def compute_statistical_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Compute standard statistical regression metrics."""
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()

        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        total_actual = float(np.sum(y_true))
        wape = float((np.sum(np.abs(y_true - y_pred)) / (total_actual + 1e-6)) * 100.0)
        r2 = float(r2_score(y_true, y_pred))

        return {
            "MAE_kWh": round(mae, 4),
            "RMSE_kWh": round(rmse, 4),
            "WAPE_pct": round(wape, 2),
            "R2_Score": round(r2, 4),
        }

    def compute_market_imbalance_costs(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute financial balancing market costs for a day-ahead procurement bid.
        
        Args:
            y_true: Actual load (kWh).
            y_pred: Day-ahead procured volume (kWh).
            
        Returns:
            Dictionary with total cost (EUR), specific cost (EUR/MWh delivered),
            and volumetric deficit/surplus breakdown.
        """
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()

        # Deficit (under-procurement): actual > bid -> need to buy balancing power
        deficit_kwh = np.maximum(0.0, y_true - y_pred)
        # Surplus (over-procurement): bid > actual -> selling surplus at discount
        surplus_kwh = np.maximum(0.0, y_pred - y_true)

        total_actual_mwh = np.sum(y_true) / 1000.0
        deficit_mwh = np.sum(deficit_kwh) / 1000.0
        surplus_mwh = np.sum(surplus_kwh) / 1000.0

        shortage_cost_eur = deficit_mwh * self.c_short
        surplus_cost_eur = surplus_mwh * self.c_long
        total_imbalance_cost_eur = shortage_cost_eur + surplus_cost_eur

        specific_imbalance_cost_eur_per_mwh = (
            total_imbalance_cost_eur / (total_actual_mwh + 1e-6)
        )

        return {
            "Total_Imbalance_Cost_EUR": round(total_imbalance_cost_eur, 2),
            "Specific_Imbalance_Cost_EUR_per_MWh": round(specific_imbalance_cost_eur_per_mwh, 2),
            "Deficit_MWh": round(deficit_mwh, 2),
            "Surplus_MWh": round(surplus_mwh, 2),
        }

    def compute_pinball_loss(
        self, y_true: np.ndarray, y_quantile_pred: np.ndarray, tau: float
    ) -> float:
        """Compute the pinball (check) loss for a given quantile tau in (0, 1)."""
        y_true = np.asarray(y_true).ravel()
        q_pred = np.asarray(y_quantile_pred).ravel()
        error = y_true - q_pred
        loss = np.maximum(tau * error, (tau - 1.0) * error)
        return float(np.mean(loss))

    def compute_interval_metrics(
        self, y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray
    ) -> Dict[str, float]:
        """Compute Prediction Interval Coverage Probability (PICP) and Mean Prediction Interval Width (MPIW)."""
        y_true = np.asarray(y_true).ravel()
        lower = np.asarray(lower).ravel()
        upper = np.asarray(upper).ravel()

        in_interval = (y_true >= lower) & (y_true <= upper)
        picp = float(np.mean(in_interval) * 100.0)
        mpiw = float(np.mean(upper - lower))

        return {
            "PICP_pct": round(picp, 2),
            "MPIW_kWh": round(mpiw, 4),
        }

    def evaluate_model(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Full evaluation bundle combining statistical accuracy and business cost."""
        stats = self.compute_statistical_metrics(y_true, y_pred)
        costs = self.compute_market_imbalance_costs(y_true, y_pred)
        stats.update(costs)
        return stats
