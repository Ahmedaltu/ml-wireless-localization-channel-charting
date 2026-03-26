import numpy as np


def csi_to_magnitude_feature(csi: np.ndarray) -> np.ndarray:
    """
    Convert complex CSI tensor to a flattened magnitude feature.

    Example input shape:
    - (n_rx, n_tx, n_subcarriers) or any complex tensor

    Output:
    - flattened real-valued vector
    """
    mag = np.abs(csi)
    return mag.reshape(-1)


def batch_csi_to_magnitude_features(csi_batch: np.ndarray) -> np.ndarray:
    """
    Convert a batch of CSI samples into a 2D feature matrix.

    Input shape:
    - (N, ...)

    Output:
    - (N, D)
    """
    features = [csi_to_magnitude_feature(csi) for csi in csi_batch]
    return np.asarray(features)