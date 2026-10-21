"""
Visualization Generator for E.ON Day-Ahead Energy Forecasting.

Generates 4 high-quality analytical figures:
1. pv_duck_curve_signature.png - Diurnal profiles demonstrating prosumer self-consumption duck curve.
2. day_ahead_forecast_comparison.png - Hold-out test week showing actual vs baseline vs LightGBM with 80% uncertainty ribbon.
3. uncertainty_by_hour_and_temp.png - Where uncertainty is highest (solar volatility & cold snap thermal thresholds).
4. procurement_cost_vs_quantile.png - Asymmetric business imbalance cost curve proving optimality of quantile bidding.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_pv_duck_curves(df_clean: pd.DataFrame, output_path: str):
    """Plot diurnal consumption profiles for PV Prosumers vs Pure Heat Pump households in summer."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(9, 5), dpi=200)

    summer = df_clean[df_clean["month"].isin([6, 7, 8])].copy()
    if len(summer) == 0:
        summer = df_clean.copy()

    hp_hourly = summer[summer["is_pv_prosumer"] == 0].groupby("hour")["kWh_received_Total"].mean()
    pv_hourly = summer[summer["is_pv_prosumer"] == 1].groupby("hour")["kWh_received_Total"].mean()

    hours = range(24)
    plt.plot(hours, hp_hourly.reindex(hours), label="Pure Heat-Pump (Non-PV)", color="#D9534F", lw=2.5, marker="o")
    plt.plot(hours, pv_hourly.reindex(hours), label="PV Prosumer (Heat-Pump + Rooftop PV)", color="#0275D8", lw=2.5, marker="s")

    # Highlight solar generation window
    plt.axvspan(10, 16, color="#F0AD4E", alpha=0.18, label="Solar Peak Window (Duck Curve)")

    plt.title("Level 1 Innovation: Prosumer 'Duck Curve' Diurnal Signature (Summer)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Hour of Day (UTC)", fontsize=11)
    plt.ylabel("Mean Hourly Net Grid Import (kWh/h)", fontsize=11)
    plt.xticks(hours)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True, facecolor="white", framealpha=0.9, loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_forecast_comparison(
    y_test: np.ndarray,
    baseline_pred: np.ndarray,
    model_pred: np.ndarray,
    q10_pred: np.ndarray,
    q90_pred: np.ndarray,
    timestamps: pd.Series,
    output_path: str,
    n_hours: int = 168,
):
    """Plot exemplary hold-out week comparison of actuals vs forecasts and uncertainty band."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(12, 5.5), dpi=200)

    sl = slice(0, n_hours)
    t = np.arange(n_hours)

    plt.plot(t, y_test[sl], label="Actual Realized Load", color="#111111", lw=2.2, zorder=5)
    plt.plot(t, baseline_pred[sl], label="Level 0 Baseline (Persistence t-168h)", color="#999999", lw=1.5, ls="--", alpha=0.8)
    plt.plot(t, model_pred[sl], label="Level 1 Stratified LightGBM", color="#0275D8", lw=2.0)

    # Level 3 Uncertainty ribbon
    plt.fill_between(
        t,
        q10_pred[sl],
        q90_pred[sl],
        color="#5BC0DE",
        alpha=0.28,
        label="Level 3 80% Uncertainty Band [Q10 - Q90]",
    )

    plt.title("Level 0 vs Level 1 vs Level 3: Day-Ahead Forecast & Uncertainty Band (Test Week)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Forecast Horizon Step (Hours)", fontsize=11)
    plt.ylabel("Net Consumption (kWh/h)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True, facecolor="white", framealpha=0.9, loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_uncertainty_by_hour_and_weather(
    test_df: pd.DataFrame,
    q10_pred: np.ndarray,
    q90_pred: np.ndarray,
    output_path: str,
):
    """Analyze where uncertainty is highest across hours of the day and temperature regimes."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = test_df.copy()
    df["interval_width"] = q90_pred - q10_pred

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=200)

    # 1. By Hour of Day (Separated by PV Prosumer status)
    hourly_uncertainty_hp = df[df["is_pv_prosumer"] == 0].groupby("hour")["interval_width"].mean()
    hourly_uncertainty_pv = df[df["is_pv_prosumer"] == 1].groupby("hour")["interval_width"].mean()

    ax1.plot(range(24), hourly_uncertainty_hp, marker="o", color="#D9534F", label="Heat-Pump Only", lw=2)
    ax1.plot(range(24), hourly_uncertainty_pv, marker="s", color="#0275D8", label="PV Prosumer", lw=2)
    ax1.axvspan(10, 16, color="#F0AD4E", alpha=0.2, label="Midday Solar Variability")
    ax1.set_title("Uncertainty Width by Hour of Day", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Hour of Day (UTC)", fontsize=10)
    ax1.set_ylabel("Mean 80% Interval Width (kWh/h)", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    # 2. By Temperature Bins (Thermodynamic heat-pump threshold effect)
    df["temp_bin"] = pd.cut(df["temp"], bins=[-20, 0, 5, 10, 15, 20, 35], labels=["<0°C", "0-5°C", "5-10°C", "10-15°C", "15-20°C", ">20°C"])
    temp_uncertainty = df.groupby("temp_bin", observed=False)["interval_width"].mean()

    colors = ["#C9302C", "#D9534F", "#F0AD4E", "#5CB85C", "#5BC0DE", "#0275D8"]
    temp_uncertainty.plot(kind="bar", ax=ax2, color=colors, edgecolor="black", alpha=0.85)
    ax2.set_title("Uncertainty Width by Ambient Temperature Regime", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Ambient Temperature Bin", fontsize=10)
    ax2.set_ylabel("Mean 80% Interval Width (kWh/h)", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5, axis="y")
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30)

    plt.suptitle("Level 3: Uncertainty Decomposition across Operational Regimes", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_cost_vs_quantile(
    quantile_costs: pd.DataFrame,
    optimal_tau: float,
    output_path: str,
):
    """Plot business imbalance cost curve across procurement bidding quantiles."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(8, 4.8), dpi=200)

    plt.plot(
        quantile_costs["Quantile"],
        quantile_costs["Specific_Cost_EUR_per_MWh"],
        marker="o",
        lw=2.5,
        color="#2E7D32",
        label="Specific Imbalance Cost (€/MWh)",
    )

    # Highlight optimal quantile
    plt.axvline(
        optimal_tau,
        color="#D9534F",
        ls="--",
        lw=2,
        label=f"Cost-Optimal Bidding Quantile (tau* = {optimal_tau:.2f})",
    )

    plt.title("Level 2 & 3: Balancing Cost Optimization under Asymmetric Market Penalties", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Procurement Bidding Quantile (tau)", fontsize=11)
    plt.ylabel("Specific Balancing Market Cost (€/MWh)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True, facecolor="white", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")
