"""
Script to generate figures for results and reports.
"""



import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt

# Ensure project root is in sys.path for src imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_npz_dataset
from src.preprocessing import split_dataset, standardize_features
from src.localization.wknn import WKNNLocalizer
from src.localization.metrics import rmse, localization_errors
from src.visualization.plots import plot_positions, plot_error_cdf, plot_error_heatmap
from src.config import DATA_PROCESSED_DIR, DEEP_MIMO_PROCESSED_FILE, WKNN_K, WKNN_EPS

def save_figure(fig_path):
    plt.savefig(fig_path, bbox_inches="tight", dpi=200)
    plt.close()

def main():
    # Ensure output directory exists
    figures_dir = Path("docs/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load processed dataset
    dataset_path = DEEP_MIMO_PROCESSED_FILE
    if not dataset_path.exists():
        raise FileNotFoundError(f"Processed DeepMIMO dataset not found at {dataset_path}.")
    X, y = load_npz_dataset(dataset_path)

    # 1. UE Positions
    plot_positions(y, title="UE Positions")
    save_figure(figures_dir / "ue_positions.png")

    # 2. Split and standardize
    X_train, X_test, y_train, y_test = split_dataset(X, y)
    X_train, X_test, _ = standardize_features(X_train, X_test)

    # 3. WKNN Localization
    model = WKNNLocalizer(k=WKNN_K, eps=WKNN_EPS)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    errors = localization_errors(y_test, y_pred)
    test_rmse = rmse(y_test, y_pred)

    # 4. WKNN Error CDF
    plot_error_cdf(errors, title=f"WKNN Error CDF (RMSE={test_rmse:.3f})")
    save_figure(figures_dir / "wknn_error_cdf.png")

    # 5. WKNN Error Heatmap
    plot_error_heatmap(y_test, errors, title="WKNN Error Heatmap")
    save_figure(figures_dir / "wknn_error_heatmap.png")

    # 6. Channel Charting (MDS/UMAP) - Placeholder for now
    # If implemented, import and use charting modules, then plot and save
    # Example:
    # from src.charting.mds_chart import mds_channel_chart
    # from src.charting.umap_chart import umap_channel_chart
    # chart_mds = mds_channel_chart(X)
    # chart_umap = umap_channel_chart(X)
    # plot_channel_chart(chart_mds, y, title="Channel Chart (MDS)")
    # save_figure(figures_dir / "channel_chart_mds.png")
    # plot_channel_chart(chart_umap, y, title="Channel Chart (UMAP)")
    # save_figure(figures_dir / "channel_chart_umap.png")

    print("All figures generated and saved to docs/figures/")

if __name__ == "__main__":
    main()
