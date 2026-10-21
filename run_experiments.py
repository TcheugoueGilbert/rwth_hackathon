"""
Main Execution Script: End-to-End Experimental Pipeline for E.ON Day-Ahead Challenge.

Runs:
- Level 0: Baseline Forecasting Models (Seasonal Persistence, Regularized Ridge)
- Level 1: Physics-informed PV Prosumer Detection & Stratified Forecasting
- Level 2: Comprehensive Evaluation under Statistical & Day-Ahead Balancing Market Costs
- Level 3: Uncertainty Quantification & Cost-Optimal Asymmetric Procurement Optimization
- Visualization Generation: Produces publication-ready figures in results/plots/
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd

# Add repo to pythonpath
REPO_DIR = os.path.dirname(os.path.abspath(__file__))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

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
from visualize_results import (
    plot_pv_duck_curves,
    plot_forecast_comparison,
    plot_uncertainty_by_hour_and_weather,
    plot_cost_vs_quantile,
)


def run_pipeline(n_households: int = 30, results_dir: str = "results"):
    print("=" * 80)
    print("  E.ON DAY-AHEAD FORECASTING & PROSUMER PROCUREMENT ENGINE  ")
    print("=" * 80)
    t0_total = time.time()
    os.makedirs(results_dir, exist_ok=True)
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # ---------------------------------------------------------
    # 1. DATA LOADING & METADATA PREPARATION
    # ---------------------------------------------------------
    print(f"\n[Step 1/6] Loading metadata and aligning {n_households} households...")
    loader = DataLoader(data_dir=os.path.join(REPO_DIR, "data"))
    meta_df = loader.load_metadata()
    print(f"Loaded metadata for {len(meta_df)} total households.")

    # Select representative cohort with diverse weather stations and balanced PV status
    labeled_pv = meta_df[meta_df["Installation_HasPVSystem"] == True]
    labeled_non_pv = meta_df[meta_df["Installation_HasPVSystem"] == False]
    unlabeled = meta_df[meta_df["Installation_HasPVSystem"].isna()]

    n_per_group = max(5, n_households // 3)
    cohort = pd.concat([
        labeled_pv.head(n_per_group),
        labeled_non_pv.head(n_per_group),
        unlabeled.head(n_households - 2 * n_per_group),
    ]).drop_duplicates(subset=["Household_ID"])
    selected_hids = cohort["Household_ID"].tolist()
    print(f"Selected evaluation cohort of {len(selected_hids)} households across weather stations:")
    print(cohort["Weather_ID"].value_counts().to_dict())

    # ---------------------------------------------------------
    # 2. LEVEL 1: PV PROSUMER SIGNATURE DETECTION
    # ---------------------------------------------------------
    print("\n[Step 2/6] Level 1: Extracting physical solar signatures and detecting prosumers...")
    pv_detector = PVSignatureDetector(data_dir=os.path.join(REPO_DIR, "data"))
    sig_df = pv_detector.extract_all_signatures(meta_df)
    _, cv_metrics = pv_detector.train_and_evaluate(sig_df)

    print("PV Prosumer Detector 5-Fold Stratified Cross-Validation Results:")
    for metric, val in cv_metrics.items():
        print(f"  - {metric.upper()}: {val:.4f}")

    # Impute and resolve prosumer status for all households
    meta_resolved = pv_detector.predict_all(sig_df)
    resolved_pv_map = dict(zip(meta_resolved["Household_ID"], meta_resolved["PV_Resolved_Flag"]))

    # ---------------------------------------------------------
    # 3. DATASET ALIGNMENT & FEATURE PIPELINE
    # ---------------------------------------------------------
    print("\n[Step 3/6] Aligning 15-min smart meters to hourly Day-Ahead resolution with weather...")
    aligned_df = loader.load_aligned_dataset(household_ids=selected_hids)
    aligned_df["PV_Resolved_Flag"] = aligned_df["Household_ID"].map(resolved_pv_map).fillna(False)
    print(f"Aligned dataset contains {len(aligned_df):,} hourly observations.")

    print("Executing strict Day-Ahead Feature Pipeline (no lookahead, causal lags >= 24h)...")
    pipeline = FeaturePipeline(t_base_heating=15.5)
    feature_df = pipeline.transform(aligned_df)
    print(f"Engineered {len(pipeline.feature_columns_)} predictive features across {len(feature_df):,} clean observations.")

    # ---------------------------------------------------------
    # 4. CHRONOLOGICAL TRAIN / TEST SPLIT (Zero Leakage)
    # ---------------------------------------------------------
    print("\n[Step 4/6] Creating strictly chronological train/validation/test partitions...")
    timestamps = feature_df["Timestamp"].sort_values()
    t_min, t_max = timestamps.min(), timestamps.max()
    t_split_val = timestamps.quantile(0.70)
    t_split_test = timestamps.quantile(0.80)

    train_mask = feature_df["Timestamp"] < t_split_val
    val_mask = (feature_df["Timestamp"] >= t_split_val) & (feature_df["Timestamp"] < t_split_test)
    test_mask = feature_df["Timestamp"] >= t_split_test

    train_df = feature_df[train_mask].copy()
    val_df = feature_df[val_mask].copy()
    test_df = feature_df[test_mask].copy()

    X_train = train_df[pipeline.feature_columns_]
    y_train = train_df["kWh_received_Total"]
    pv_train = train_df["is_pv_prosumer"]

    X_test = test_df[pipeline.feature_columns_]
    y_test = test_df["kWh_received_Total"].values
    pv_test = test_df["is_pv_prosumer"]

    print(f"  - Train Partition: {len(train_df):,} records ({train_df['Timestamp'].min().date()} to {train_df['Timestamp'].max().date()})")
    print(f"  - Val Partition:   {len(val_df):,} records ({val_df['Timestamp'].min().date()} to {val_df['Timestamp'].max().date()})")
    print(f"  - Test Partition:  {len(test_df):,} records ({test_df['Timestamp'].min().date()} to {test_df['Timestamp'].max().date()})")

    # ---------------------------------------------------------
    # 5. MODEL TRAINING & LEVEL 2 BENCHMARK EVALUATION
    # ---------------------------------------------------------
    print("\n[Step 5/6] Training Level 0 Baselines, Level 1 Advanced Models, and Level 3 Quantile Models...")
    evaluator = ProcurementEvaluator(c_short_eur_per_mwh=35.0, c_long_eur_per_mwh=15.0)

    # 5.1 Level 0: Seasonal Persistence
    t0 = time.time()
    persistence_model = SeasonalPersistenceBaseline().fit(X_train, y_train)
    y_pred_persistence = persistence_model.predict(X_test)
    t_pers = time.time() - t0

    # 5.2 Level 0: Ridge Regression
    t0 = time.time()
    ridge_model = RidgeBaseline(alpha=10.0).fit(X_train, y_train)
    y_pred_ridge = ridge_model.predict(X_test)
    t_ridge = time.time() - t0

    # 5.3 Level 1: Global LightGBM
    t0 = time.time()
    global_lgb = GlobalLightGBMForecaster(n_estimators=250, learning_rate=0.05).fit(X_train, y_train)
    y_pred_global = global_lgb.predict(X_test)
    t_global = time.time() - t0

    # 5.4 Level 1: Stratified Forecaster (Heat-Pump vs PV Prosumer)
    t0 = time.time()
    stratified_lgb = StratifiedForecaster().fit(X_train, y_train, pv_train)
    y_pred_stratified = stratified_lgb.predict(X_test, pv_test)
    t_strat = time.time() - t0

    # Evaluate Level 0, 1, 2
    models_dict = {
        "Level 0: Seasonal Persistence (t-168h)": y_pred_persistence,
        "Level 0: Regularized Ridge Regression": y_pred_ridge,
        "Level 1: Global LightGBM Regressor": y_pred_global,
        "Level 1: Stratified Prosumer Forecaster": y_pred_stratified,
    }

    eval_results = []
    base_cost = None
    for name, preds in models_dict.items():
        res = evaluator.evaluate_model(y_test, preds)
        res["Model"] = name
        if base_cost is None:
            base_cost = res["Total_Imbalance_Cost_EUR"]
        res["Cost_Savings_vs_Baseline_EUR"] = round(base_cost - res["Total_Imbalance_Cost_EUR"], 2)
        res["Cost_Reduction_pct"] = round(
            ((base_cost - res["Total_Imbalance_Cost_EUR"]) / base_cost) * 100.0, 2
        )
        eval_results.append(res)

    results_table = pd.DataFrame(eval_results)[[
        "Model", "MAE_kWh", "RMSE_kWh", "WAPE_pct", "R2_Score",
        "Specific_Imbalance_Cost_EUR_per_MWh", "Cost_Reduction_pct", "Total_Imbalance_Cost_EUR"
    ]]

    print("\n" + "=" * 95)
    print("  DAY-AHEAD PROCUREMENT BENCHMARK EVALUATION (LEVELS 0, 1, 2)  ")
    print("=" * 95)
    print(results_table.to_string(index=False))
    print("=" * 95)

    # ---------------------------------------------------------
    # 6. LEVEL 3: UNCERTAINTY QUANTIFICATION & COST-OPTIMAL BIDDING
    # ---------------------------------------------------------
    print("\n[Step 6/6] Level 3: Quantile Regression & Probabilistic Bidding Optimization...")
    quantiles = [0.10, 0.25, 0.50, 0.70, 0.75, 0.90]
    q_forecaster = QuantileDayAheadForecaster(quantiles=quantiles).fit(X_train, y_train)
    q_preds = q_forecaster.predict_quantiles(X_test)

    # 6.1 Interval metrics (80% interval: [Q10, Q90])
    lower_80, median_pred, upper_80 = q_forecaster.predict_interval(X_test, 0.10, 0.90)
    interval_metrics = evaluator.compute_interval_metrics(y_test, lower_80, upper_80)
    print("Level 3 Uncertainty Interval Metrics (80% Nominal Target):")
    print(f"  - Empirical Coverage (PICP): {interval_metrics['PICP_pct']:.2f}% (Target: 80.0%)")
    print(f"  - Mean Interval Width (MPIW): {interval_metrics['MPIW_kWh']:.4f} kWh/h")

    # 6.2 Pinball Losses
    print("\nPinball Loss per Quantile:")
    pinball_records = []
    for q in quantiles:
        pb_loss = evaluator.compute_pinball_loss(y_test, q_preds[q], q)
        costs = evaluator.compute_market_imbalance_costs(y_test, q_preds[q])
        pinball_records.append({
            "Quantile": q,
            "Pinball_Loss": round(pb_loss, 4),
            "Specific_Cost_EUR_per_MWh": costs["Specific_Imbalance_Cost_EUR_per_MWh"],
            "Total_Cost_EUR": costs["Total_Imbalance_Cost_EUR"],
        })
        print(f"  - Tau = {q:.2f}: Pinball Loss = {pb_loss:.4f} | Specific Cost = €{costs['Specific_Imbalance_Cost_EUR_per_MWh']:.2f}/MWh")

    q_cost_df = pd.DataFrame(pinball_records)

    # Compare cost-optimal bidding tau* vs median bidding tau=0.50
    tau_opt = evaluator.optimal_tau  # 0.70
    cost_median = q_cost_df[q_cost_df["Quantile"] == 0.50]["Total_Cost_EUR"].values[0]
    cost_opt = q_cost_df[q_cost_df["Quantile"] == round(tau_opt, 2)]["Total_Cost_EUR"].values[0]
    opt_savings_eur = cost_median - cost_opt
    opt_savings_pct = (opt_savings_eur / cost_median) * 100.0

    print(f"\nCost-Optimal Bidding Optimization Result (tau* = {tau_opt:.2f}):")
    print(f"  - Deterministic / Median Bid (tau = 0.50): €{cost_median:,.2f}")
    print(f"  - Cost-Optimal Asymmetric Bid (tau* = {tau_opt:.2f}): €{cost_opt:,.2f}")
    print(f"  - Additional Financial Imbalance Reduction: €{opt_savings_eur:,.2f} ({opt_savings_pct:.2f}% additional savings)")

    # ---------------------------------------------------------
    # 7. EXPORT RESULTS & GENERATE FIGURES
    # ---------------------------------------------------------
    results_path = os.path.join(results_dir, "experiment_summary.csv")
    results_table.to_csv(results_path, index=False)
    q_cost_path = os.path.join(results_dir, "quantile_costs.csv")
    q_cost_df.to_csv(q_cost_path, index=False)
    print(f"\nSaved numerical results to {results_path} and {q_cost_path}.")

    print("Generating visual plots...")
    plot_pv_duck_curves(
        feature_df,
        os.path.join(plots_dir, "pv_duck_curve_signature.png")
    )
    plot_forecast_comparison(
        y_test=y_test,
        baseline_pred=y_pred_persistence,
        model_pred=y_pred_stratified,
        q10_pred=lower_80,
        q90_pred=upper_80,
        timestamps=test_df["Timestamp"],
        output_path=os.path.join(plots_dir, "day_ahead_forecast_comparison.png"),
        n_hours=168,
    )
    plot_uncertainty_by_hour_and_weather(
        test_df=test_df,
        q10_pred=lower_80,
        q90_pred=upper_80,
        output_path=os.path.join(plots_dir, "uncertainty_by_hour_and_temp.png"),
    )
    plot_cost_vs_quantile(
        quantile_costs=q_cost_df,
        optimal_tau=tau_opt,
        output_path=os.path.join(plots_dir, "procurement_cost_vs_quantile.png"),
    )

    t_elapsed = time.time() - t0_total
    print(f"\nPipeline successfully completed in {t_elapsed:.1f} seconds.")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run E.ON Day-Ahead Forecasting Pipeline")
    parser.add_argument("--n_households", type=int, default=30, help="Number of households to evaluate")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory to save results")
    args = parser.parse_args()

    run_pipeline(n_households=args.n_households, results_dir=args.results_dir)
