import numpy as np
from pathlib import Path


def load_npz_dataset(file_path: str | Path):
    """
    Load a simplified processed dataset from an .npz file.

    Expected keys:
    - X: feature matrix of shape (N, D)
    - y: position matrix of shape (N, 2)
    """
    data = np.load(file_path)
    X = data["X"]
    y = data["y"]
    return X, y


def save_npz_dataset(file_path: str | Path, X: np.ndarray, y: np.ndarray):
    """
    Save processed features and positions into an .npz file.
    """
    np.savez_compressed(file_path, X=X, y=y)