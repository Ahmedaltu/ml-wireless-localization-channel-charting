from pathlib import Path

from src.data_loader import load_npz_dataset
from src.preprocessing import split_dataset, standardize_features
from src.localization.wknn import WKNNLocalizer
from src.localization.metrics import rmse, localization_errors
from src.visualization.plots import plot_positions, plot_error_cdf, plot_error_heatmap
from src.config import DATA_PROCESSED_DIR, WKNN_K, WKNN_EPS


def main():
    dataset_path = DATA_PROCESSED_DIR / "real_baseline.npz"

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. "
            "Create data/processed/real_baseline.npz with arrays X and y."
        )

    X, y = load_npz_dataset(dataset_path)

    plot_positions(y, title="All UE Positions")

    X_train, X_test, y_train, y_test = split_dataset(X, y)
    X_train, X_test, _ = standardize_features(X_train, X_test)

    model = WKNNLocalizer(k=WKNN_K, eps=WKNN_EPS)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    test_rmse = rmse(y_test, y_pred)
    errors = localization_errors(y_test, y_pred)

    print(f"WKNN RMSE: {test_rmse:.4f}")

    plot_error_cdf(errors, title=f"WKNN Error CDF (RMSE={test_rmse:.3f})")
    plot_error_heatmap(y_test, errors, title="WKNN Error Heatmap")


if __name__ == "__main__":
    main()