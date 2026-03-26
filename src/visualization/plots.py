import numpy as np
import matplotlib.pyplot as plt


def plot_positions(y, title="UE Positions"):
    plt.figure(figsize=(6, 5))
    plt.scatter(y[:, 0], y[:, 1], s=8, alpha=0.7)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_error_cdf(errors, title="Localization Error CDF"):
    errors_sorted = np.sort(errors)
    cdf = np.arange(1, len(errors_sorted) + 1) / len(errors_sorted)

    plt.figure(figsize=(6, 5))
    plt.plot(errors_sorted, cdf)
    plt.xlabel("Localization Error")
    plt.ylabel("CDF")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_error_heatmap(y_true, errors, title="Localization Error Heatmap"):
    plt.figure(figsize=(6, 5))
    sc = plt.scatter(y_true[:, 0], y_true[:, 1], c=errors, s=12, alpha=0.8)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.colorbar(sc, label="Error")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()