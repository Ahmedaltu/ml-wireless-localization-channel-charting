"""
Script to convert cached DeepMIMO export to ML-ready dataset for wireless localization.
- Loads data/raw/deepmimo/deepmimo_export.npz
- Extracts CSI and positions
- Flattens CSI to real-valued features
- Optionally subsamples for large datasets
- Saves to data/processed/real_baseline.npz
"""
import numpy as np
from pathlib import Path

def load_deepmimo_export(path):
    arr = np.load(path)
    csi = arr["csi"]
    positions = arr["positions"]
    return csi, positions

def process_features_labels(csi, positions, max_samples=None, seed=42):
    N = positions.shape[0]
    if max_samples is not None and N > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(N, max_samples, replace=False)
        csi = csi[idx]
        positions = positions[idx]
    X = np.abs(csi).reshape(csi.shape[0], -1)
    y = positions[:, :2]
    return X, y

def save_processed_dataset(X, y, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, X=X, y=y)

def main():
    in_path = Path("data/raw/deepmimo/deepmimo_export.npz")
    out_path = Path("data/processed/real_baseline.npz")
    max_samples = 5000  # Change as needed

    if not in_path.exists():
        print(f"Input file not found: {in_path}")
        return

    csi, positions = load_deepmimo_export(in_path)
    print(f"Loaded: csi shape {csi.shape}, positions shape {positions.shape}")

    X, y = process_features_labels(csi, positions, max_samples=max_samples)
    print(f"Processed: X shape {X.shape}, y shape {y.shape}")

    save_processed_dataset(X, y, out_path)
    print(f"Saved processed dataset to {out_path}")

if __name__ == "__main__":
    main()
