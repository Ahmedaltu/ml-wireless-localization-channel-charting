import numpy as np

from src.config import (
    DEEP_MIMO_RAW_DIR,
    DEEP_MIMO_PROCESSED_FILE,
    MAX_REAL_SAMPLES,
    RANDOM_STATE,
)
from src.data_loader import save_npz_dataset
from src.feature_extraction import batch_csi_to_magnitude_features
from src.preprocessing import subsample_dataset


def load_deepmimo_exported_npz(npz_path):
    """
    Load a simplified DeepMIMO export file.

    Expected keys:
    - csi: complex CSI batch of shape (N, ...)
    - positions: UE positions of shape (N, 2) or (N, 3)

    If positions are 3D, only x,y are used for baseline localization.
    """
    data = np.load(npz_path, allow_pickle=True)
    csi = data["csi"]
    positions = data["positions"]

    if positions.shape[1] >= 2:
        y = positions[:, :2]
    else:
        raise ValueError("positions must have at least 2 columns")

    return csi, y


def main():
    raw_file = DEEP_MIMO_RAW_DIR / "deepmimo_export.npz"

    if not raw_file.exists():
        raise FileNotFoundError(
            f"Expected exported DeepMIMO file at {raw_file}\n"
            "Create data/raw/deepmimo/deepmimo_export.npz with keys: csi, positions"
        )

    print(f"Loading DeepMIMO export from: {raw_file}")
    csi, y = load_deepmimo_exported_npz(raw_file)

    print(f"Raw CSI shape: {csi.shape}")
    print(f"Raw positions shape: {y.shape}")

    print("Extracting CSI magnitude features...")
    X = batch_csi_to_magnitude_features(csi)

    print(f"Feature matrix shape before subsampling: {X.shape}")

    X, y = subsample_dataset(X, y, max_samples=MAX_REAL_SAMPLES, random_state=RANDOM_STATE)

    print(f"Feature matrix shape after subsampling: {X.shape}")
    print(f"Position matrix shape after subsampling: {y.shape}")

    DEEP_MIMO_PROCESSED_FILE.parent.mkdir(parents=True, exist_ok=True)
    save_npz_dataset(DEEP_MIMO_PROCESSED_FILE, X, y)

    print(f"Saved processed DeepMIMO baseline to: {DEEP_MIMO_PROCESSED_FILE}")


if __name__ == "__main__":
    main()