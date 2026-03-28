"""
Streamlit App: CSI-Based Fingerprint Localization & Channel Charting
Presentation-ready demo for real-data wireless ML (5G/6G, RAN, Digital Twin)
Focus: Real CSI-based fingerprint localization and channel charting
"""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.manifold import MDS

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

# --- Helper Functions ---
def load_dataset(name):
    path = Path("data/processed/real_baseline.npz")
    if not path.exists():
        return None, None, None, path
    arr = np.load(path)
    X = arr["csi"] if "csi" in arr else arr["X"] if "X" in arr else None
    y = arr["positions"] if "positions" in arr else arr["y"] if "y" in arr else None
    bs_pos = None
    for key in ["bs_pos", "bs_position", "bs", "BS", "bs_coords"]:
        if key in arr:
            bs_pos = arr[key]
            break
    return X, y, bs_pos, path

def split_and_standardize(X, y, seed=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)
    return X_train_std, X_test_std, y_train, y_test

def run_wknn(X_train, y_train, X_test, k=5):
    knn = KNeighborsRegressor(n_neighbors=k, weights="distance")
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    return y_pred

def plot_positions(y, bs_pos=None, title="UE Positions"):
    fig, ax = plt.subplots()
    ax.scatter(y[:,0], y[:,1], s=10, alpha=0.7, label="UE")
    if bs_pos is not None:
        ax.scatter(bs_pos[0], bs_pos[1], c='red', marker='*', s=100, label="BS")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.legend()
    st.pyplot(fig)

def plot_error_cdf(errors):
    sorted_err = np.sort(errors)
    cdf = np.arange(1, len(errors)+1) / len(errors)
    fig, ax = plt.subplots()
    ax.plot(sorted_err, cdf)
    ax.set_xlabel("Localization Error (m)")
    ax.set_ylabel("CDF")
    ax.set_title("Localization Error CDF")
    st.pyplot(fig)

def plot_error_heatmap(y_true, errors):
    fig, ax = plt.subplots()
    sc = ax.scatter(y_true[:,0], y_true[:,1], c=errors, cmap='viridis', s=20)
    plt.colorbar(sc, ax=ax, label="Error (m)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Error Heatmap")
    st.pyplot(fig)

def run_mds(X, n=2, seed=42):
    mds = MDS(n_components=n, random_state=seed)
    return mds.fit_transform(X)

def run_umap(X, n=2, seed=42):
    if not HAS_UMAP:
        return None
    reducer = umap.UMAP(n_components=n, random_state=seed)
    return reducer.fit_transform(X)

def plot_channel_chart(embedding, y, color_by="x"):
    fig, ax = plt.subplots()
    if color_by == "x":
        c = y[:,0]
        label = "True x"
    else:
        c = y[:,1]
        label = "True y"
    sc = ax.scatter(embedding[:,0], embedding[:,1], c=c, cmap='plasma', s=15)
    plt.colorbar(sc, ax=ax, label=label)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.set_title(f"Channel Chart colored by {label}")
    st.pyplot(fig)

# --- Streamlit UI ---

st.set_page_config(page_title="CSI-Based Fingerprint Localization & Channel Charting", layout="wide")
st.title("CSI-Based Fingerprint Localization & Channel Charting")
st.caption("A 5G/6G-oriented wireless ML project for RAN/Digital Twin analysis using real CSI data.")

with st.sidebar:
    st.header("Controls")
    st.write("Dataset: Real processed dataset (required)")
    chart_method = st.selectbox("Channel Charting Method", ["MDS", "UMAP"])
    n_samples = st.slider("# Samples to visualize (charting)", 100, 5000, 1000, step=100)
    seed = st.number_input("Random Seed", value=42, step=1)
    run_btn = st.button("Run analysis")

# --- Main Tabs ---
tabs = st.tabs(["Dataset Overview", "Fingerprint Localization", "Channel Charting", "Future Work"])

# --- Dataset Loading ---
X, y, bs_pos, path = load_dataset(None)
if X is None or y is None:
    with tabs[0]:
        st.warning(f"Real dataset file not found: {path}. Please create data/processed/real_baseline.npz first using the provided scripts.")
        st.caption("This app requires a real processed dataset (not synthetic). Please run the DeepMIMO export and processing pipeline.")
    st.stop()

# Subsample for charting speed
if X.shape[0] > n_samples:
    np.random.seed(seed)
    idx = np.random.choice(X.shape[0], n_samples, replace=False)
    X_vis = X[idx]
    y_vis = y[idx]
else:
    X_vis, y_vis = X, y

# --- Dataset Overview Tab ---
with tabs[0]:
    st.subheader("Dataset Overview")
    st.write(f"**Number of samples:** {X.shape[0]}")
    st.write(f"**Feature dimension:** {X.shape[1]}")
    st.write(f"**Coordinate dimension:** {y.shape[1]}")
    plot_positions(y, bs_pos=bs_pos, title="UE Positions")
    if bs_pos is not None:
        st.info(f"BS location: {bs_pos}")
    else:
        st.info("BS location not available in dataset.")
    st.caption("This dataset uses CSI-derived feature vectors and ground-truth UE coordinates.")

# --- Fingerprint Localization Tab ---
with tabs[1]:
    st.subheader("Fingerprint Localization (WKNN)")
    st.caption("Fingerprint localization is a supervised task: it predicts actual physical UE coordinates from CSI features. Evaluated using RMSE and error distribution.")
    X_train, X_test, y_train, y_test = split_and_standardize(X, y, seed=seed)
    y_pred = run_wknn(X_train, y_train, X_test, k=5)
    errors = np.linalg.norm(y_pred - y_test, axis=1)
    rmse = np.sqrt(np.mean(errors**2))
    st.metric("Test RMSE (m)", f"{rmse:.2f}")
    plot_error_cdf(errors)
    plot_error_heatmap(y_test, errors)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test[:,0], y_test[:,1], c='blue', s=10, label='True')
    ax.scatter(y_pred[:,0], y_pred[:,1], c='orange', s=10, label='Predicted', alpha=0.6)
    ax.set_title("Predicted vs True Positions")
    ax.legend()
    st.pyplot(fig)

# --- Channel Charting Tab ---
with tabs[2]:
    st.subheader(f"Channel Charting ({chart_method})")
    st.caption("Channel charting is usually unsupervised: it aims to preserve local geometric relationships in the radio domain. Chart axes are latent embedding axes, not real x-y coordinates.")
    if chart_method == "MDS":
        emb = run_mds(X_vis, n=2, seed=seed)
    elif chart_method == "UMAP":
        if not HAS_UMAP:
            st.warning("UMAP is not installed. Please install umap-learn to use this feature.")
            emb = None
        else:
            emb = run_umap(X_vis, n=2, seed=seed)
    if emb is not None:
        st.info("Note: Chart axes are latent embedding axes, not real x-y coordinates.")
        plot_channel_chart(emb, y_vis, color_by="x")
        plot_channel_chart(emb, y_vis, color_by="y")

# --- Future Work Tab ---
with tabs[3]:
    st.subheader("Future Work")
    st.write("""
    - MLP-based localization
    - Channel charting quality metrics (Trustworthiness, Continuity, Kruskal Stress)
    - Beam management application
    - Robustness to noisy CSI
    - Multi-point / multi-BS extension
    """)
