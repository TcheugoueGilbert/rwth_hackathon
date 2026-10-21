"""
Unit and integration tests for the Day-Ahead Procurement Toolkit.
"""

import os
import unittest
import numpy as np
import pandas as pd

from utils.data_loader import DataLoader
from utils.pv_detector import PVSignatureDetector
from utils.feature_engineering import FeaturePipeline
from utils.models import (
    SeasonalPersistenceBaseline,
    RidgeBaseline,
    GlobalLightGBMForecaster,
    StratifiedForecaster,
    QuantileDayAheadForecaster,
)
from utils.evaluation import ProcurementEvaluator


class TestDayAheadToolkit(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        cls.data_dir = os.path.join(cls.repo_dir, "data")

    def test_data_loader_metadata(self):
        loader = DataLoader(self.data_dir)
        meta = loader.load_metadata()
        self.assertGreaterEqual(len(meta), 410)
        self.assertIn("Household_ID", meta.columns)
        self.assertIn("Weather_ID", meta.columns)

    def test_feature_pipeline_causality(self):
        pipeline = FeaturePipeline(t_base_heating=15.5)
        # Create synthetic time series with known lag
        dates = pd.date_range("2023-01-01", periods=200, freq="1h", tz="UTC")
        syn_df = pd.DataFrame({
            "Timestamp": dates,
            "Household_ID": "test_hh",
            "kWh_received_Total": np.sin(np.arange(200)) + 2.0,
            "Temperature_avg_hourly": 10.0,
            "WindSpeed_hourly": 3.0,
            "Humidity_avg_hourly": 70.0,
            "Sunshine_duration_hourly": 0.5,
            "is_pv_prosumer": 0,
        })
        transformed = pipeline.transform(syn_df)
        self.assertIn("hdh", transformed.columns)
        self.assertIn("lag_24h", transformed.columns)
        self.assertIn("lag_168h", transformed.columns)
        # Ensure HDH is max(0, 15.5 - 10.0) = 5.5
        self.assertAlmostEqual(transformed["hdh"].iloc[0], 5.5, places=4)

    def test_procurement_evaluator_metrics(self):
        evaluator = ProcurementEvaluator(c_short_eur_per_mwh=35.0, c_long_eur_per_mwh=15.0)
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([10.0, 20.0, 30.0])
        stats = evaluator.compute_statistical_metrics(y_true, y_pred)
        self.assertEqual(stats["MAE_kWh"], 0.0)
        self.assertEqual(stats["WAPE_pct"], 0.0)
        self.assertEqual(stats["R2_Score"], 1.0)

        # Test asymmetric costs
        # True = 20 kWh, Pred = 10 kWh (deficit of 10 kWh = 0.01 MWh -> 0.01 * 35 = 0.35 EUR)
        costs = evaluator.compute_market_imbalance_costs(np.array([20.0]), np.array([10.0]))
        self.assertAlmostEqual(costs["Total_Imbalance_Cost_EUR"], 0.35, places=2)

    def test_quantile_forecaster_monotonicity(self):
        q_forecaster = QuantileDayAheadForecaster(quantiles=[0.10, 0.50, 0.90])
        X = pd.DataFrame(np.random.randn(100, 5), columns=[f"feat_{i}" for i in range(5)])
        y = pd.Series(np.abs(np.random.randn(100)) + 1.0)
        q_forecaster.fit(X, y)

        lower, median, upper = q_forecaster.predict_interval(X, 0.10, 0.90)
        self.assertEqual(len(lower), 100)
        self.assertTrue(np.all(upper >= lower))


if __name__ == "__main__":
    unittest.main()
