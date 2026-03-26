
# Import necessary libraries
import numpy as np
from sklearn.neighbors import NearestNeighbors



class WKNNLocalizer:
    """
    Weighted k-Nearest Neighbors (WKNN) localizer for position estimation.
    Given features and known positions, this class can fit a model and predict positions for new samples.
    """

    def __init__(self, k=5, eps=1e-8):
        """
        Initialize the WKNN localizer.
        Args:
            k (int): Number of nearest neighbors to use.
            eps (float): Small value to avoid division by zero in weights.
        """
        self.k = k
        self.eps = eps
        self.nn = None  # NearestNeighbors model
        self.y_train = None  # Ground truth positions for training data


    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Fit the WKNN model to training data.
        Args:
            X_train (np.ndarray): Training features (samples x features).
            y_train (np.ndarray): Training positions (samples x coordinates).
        Returns:
            self
        """
        # Fit a k-NN model on the training features
        self.nn = NearestNeighbors(n_neighbors=self.k, metric="euclidean")
        self.nn.fit(X_train)
        self.y_train = y_train
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict positions for test samples using weighted k-nearest neighbors.
        Args:
            X_test (np.ndarray): Test features (samples x features).
        Returns:
            np.ndarray: Predicted positions (samples x coordinates).
        """
        # Find k nearest neighbors for each test sample
        distances, indices = self.nn.kneighbors(X_test)

        preds = []
        for dists, idxs in zip(distances, indices):
            # Compute weights inversely proportional to distance (add eps for stability)
            weights = 1.0 / (dists + self.eps)
            weights = weights / np.sum(weights)  # Normalize weights
            # Weighted sum of neighbor positions
            pred = np.sum(self.y_train[idxs] * weights[:, None], axis=0)
            preds.append(pred)

        return np.asarray(preds)