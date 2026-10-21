from pathlib import Path

from utils.energy_forecast import load_energy_data, evaluate_forecast_model


def main() -> None:
    data_dir = Path(__file__).resolve().parent / "data"
    sample = load_energy_data(limit_households=10, data_dir=data_dir)
    metrics = evaluate_forecast_model(sample)

    print("Day-ahead forecast summary")
    print("=" * 28)
    print(f"MAE: {metrics['mae']:.3f} kWh")
    print(f"RMSE: {metrics['rmse']:.3f} kWh")
    print(f"MAPE: {metrics['mape']:.2f}%")
    print(f"R²: {metrics['r2']:.3f}")


if __name__ == "__main__":
    main()
