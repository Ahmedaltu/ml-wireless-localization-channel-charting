
# Script to generate and save a synthetic dataset for localization experiments.
# This script creates random 2D positions and corresponding feature vectors,
# then saves them in a compressed .npz file for later use.

import numpy as np

# Import configuration variables and data saving utility
from src.config import DATA_PROCESSED_DIR, RANDOM_STATE
from src.data_loader import save_npz_dataset



def main():
    # Create a random number generator with a fixed seed for reproducibility
    rng = np.random.default_rng(RANDOM_STATE)

    # Generate synthetic 2D user equipment (UE) positions
    n_samples = 2000  # Number of samples in the dataset
    y = rng.uniform(low=[0.0, 0.0], high=[50.0, 30.0], size=(n_samples, 2))

    # Generate synthetic features correlated with position, plus noise
    n_features = 128  # Number of features per sample
    W = rng.normal(size=(2, n_features))  # Random linear transformation
    X = y @ W + 0.5 * rng.normal(size=(n_samples, n_features))  # Features = linear + noise

    # Ensure the processed data directory exists
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_PROCESSED_DIR / "baseline_dataset.npz"

    # Save the generated dataset to a compressed .npz file
    save_npz_dataset(out_path, X, y)
    print(f"Saved synthetic baseline dataset to {out_path}")



# Entry point for script execution
if __name__ == "__main__":
    main()