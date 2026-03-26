import numpy as np


def localization_errors(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.linalg.norm(y_true - y_pred, axis=1)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    errors = localization_errors(y_true, y_pred)
    return float(np.sqrt(np.mean(errors ** 2)))