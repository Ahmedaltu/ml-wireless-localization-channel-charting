import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import TEST_SIZE, RANDOM_STATE


def split_dataset(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def standardize_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def subsample_dataset(X, y, max_samples=None, random_state=RANDOM_STATE):
    if max_samples is None or len(X) <= max_samples:
        return X, y

    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(X), size=max_samples, replace=False)
    return X[idx], y[idx]