# Spatially Consistent CSI-Based Fingerprint Localization and Channel Charting with DeepMIMO

## Executive Summary

This system implements a modular, research-oriented machine learning pipeline for wireless communications, focused on CSI-based fingerprint localization and channel charting. Using realistic wireless datasets (DeepMIMO or exported real data), the project demonstrates how channel state information (CSI) can be leveraged for both supervised localization and unsupervised radio geometry learning. The system is directly relevant to 5G/6G, RAN simulation, and Digital Twin analysis, and includes a professional Streamlit UI for demonstration and exploration.

---

## 1. System Purpose

The system addresses the challenge of learning spatial and geometric properties of wireless radio environments from channel measurements. By extracting features from CSI, it enables:
- Accurate fingerprint localization (UE positioning)
- Channel charting (latent radio geometry learning)

This is relevant for:
- Localization and context awareness
- Beam management and mobility management
- Radio resource management
- Digital Twin abstractions for RAN

CSI is a rich, high-dimensional representation of the wireless channel, making it a powerful input for learning radio geometry and supporting intelligent RAN workflows.

---

## 2. System Scope

**Implemented:**
- Dataset loading and preprocessing
- CSI feature extraction (magnitude-based)
- WKNN fingerprint localization
- Streamlit visualization UI
- MDS and UMAP channel charting (2D)

**Planned / Optional:**
- MLP/CNN-based localization
- Charting metrics (Trustworthiness, Continuity, Kruskal Stress)
- Robustness to noisy/estimated CSI
- Multi-BS / multi-point charting
- Beam management use case

---

## 3. High-Level Architecture

```
Raw Wireless Data (DeepMIMO / Exported Dataset)
    ↓
Data Extraction
    ↓
CSI Feature Construction
    ↓
Processed Dataset (X, y)
    ↓
+-----------------------------+
|  Fingerprint Localization   |
|  (WKNN, MLP)               |
+-----------------------------+
    ↓
+-----------------------------+
|  Channel Charting           |
|  (MDS, UMAP)                |
+-----------------------------+
    ↓
Visualization / Streamlit UI
    ↓
Figures / Results / Report
```

**Stage Explanations:**
- **Raw Data**: DeepMIMO or exported real wireless datasets
- **Extraction**: UE positions and CSI are extracted
- **Feature Construction**: CSI is converted to ML-ready features
- **Processed Dataset**: Saved as (X, y) for downstream tasks
- **Localization**: Supervised learning to predict UE coordinates
- **Charting**: Unsupervised learning to build a latent radio map
- **Visualization**: Results and figures for analysis and presentation

---

## 4. Data Flow

1. **Scenario Generation/Loading**: DeepMIMO or real dataset is loaded
2. **Extraction**: UE positions and channel/CSI are extracted
3. **Raw Export**: Saved as `data/raw/deepmimo/deepmimo_export.npz`
4. **Preprocessing**: Complex CSI → magnitude-based feature vectors
5. **Processed Dataset**: Saved as `data/processed/real_baseline.npz`
6. **Analysis**: WKNN/MLP localization, MDS/UMAP charting
7. **Visualization**: CDF, heatmap, channel chart, UI

- **X**: Feature matrix (flattened, magnitude-based CSI)
- **y**: 2D UE positions (ground-truth coordinates)

---

## 5. Core Modules

- **scripts/**: Data preparation and experiment runners
- **src/data_loader.py**: Loading/saving datasets
- **src/preprocessing.py**: Data cleaning, normalization, feature engineering
- **src/feature_extraction.py**: CSI feature construction
- **src/localization/wknn.py**: WKNN localization implementation
- **src/localization/metrics.py**: Evaluation metrics (RMSE, error CDF, etc.)
- **src/visualization/plots.py**: Plotting utilities for results/figures
- **app.py**: Streamlit UI for interactive demo and visualization

---

## 6. Fingerprint Localization Module

- **Task**: Supervised learning to predict UE coordinates from CSI features
- **Input**: CSI-derived feature vectors (X)
- **Output**: Predicted 2D UE coordinates (ŷ)
- **Baseline**: WKNN (Weighted K-Nearest Neighbors)
- **Extension**: MLP regression (planned)

**Evaluation Metrics:**
- RMSE (Root Mean Squared Error)
- Error CDF
- Spatial error heatmap

> WKNN is a strong classical baseline due to its simplicity, interpretability, and effectiveness in spatial ML tasks.

---

## 7. Channel Charting Module

- **Task**: Unsupervised representation learning to build a 2D latent radio map
- **Input**: CSI-derived feature vectors (X)
- **Output**: 2D embedding (chart axes are latent, not physical coordinates)
- **Baselines**: MDS (classical), UMAP (manifold learning)

**Evaluation Goals:**
- Preserve local geometric relationships
- Neighborhood preservation
- Visual spatial continuity

**Planned Metrics:**
- Trustworthiness (TW)
- Continuity (CT)
- Kruskal Stress (KS)

> Chart axes are latent embedding axes, not real x-y coordinates.

---

## 8. Streamlit UI / Presentation Layer

- **Dataset Overview tab**: Dataset dimensions, UE positions
- **Fingerprint Localization tab**: RMSE, Error CDF, Error heatmap
- **Channel Charting tab**: MDS/UMAP visualization, color-coded structure
- **Future Work tab**: Planned extensions

The UI is a lightweight research demo for rapid inspection, project presentation, and portfolio demonstration.

---

## 9. Repository / File Structure

```
app.py                  # Streamlit UI
scripts/                # Data prep and experiment scripts
src/                    # Core source code modules
  data_loader.py
  preprocessing.py
  feature_extraction.py
  localization/
  visualization/
data/
  raw/                  # Raw/cached data
  processed/            # ML-ready processed data
docs/
  figures/              # Project figures and screenshots
results/                # Generated results, metrics, models
slides/                 # Presentation slides
report/                 # Project report
archive/                # (if synthetic baseline scripts were archived)
```

**Folder Explanations:**
- **app.py**: Main UI
- **scripts/**: Data prep, runners
- **src/**: Modular codebase
- **data/**: All data (raw, processed)
- **docs/**: Documentation, figures
- **results/**: Output results
- **slides/**: Presentations
- **report/**: Written report
- **archive/**: Deprecated/archived scripts

---

## 10. Design Decisions and Rationale

- **DeepMIMO/realistic data**: Ensures relevance to real-world wireless systems
- **Magnitude-based CSI features**: Simple, robust baseline for initial experiments
- **WKNN baseline**: Strong, interpretable classical method for localization
- **MDS vs UMAP**: Compare classical and modern manifold learning for charting
- **Streamlit UI**: Lightweight, rapid prototyping and demo
- **Raw vs processed data separation**: Ensures reproducibility and clarity

---

## 11. Assumptions and Limitations

- Baseline features may ignore phase information
- Single-BS baseline may limit generality
- Channel charting metrics may be partially implemented
- Synthetic baseline (if archived) was only for early validation
- System prioritizes clarity and reproducibility over maximum performance

---

## 12. Future Work

- MLP/CNN-based localization
- Robust features under noisy CSI
- Multi-BS / multi-point charting
- Out-of-sample channel charting
- Beam management / handover application
- Formal charting metrics (TW, CT, KS)
- Comparison across scenarios/frequencies

---

## 13. Conclusion

This system demonstrates a modular, research-oriented wireless ML pipeline, connecting realistic CSI data to both localization and representation learning. It is directly relevant for 5G/6G intelligent RAN workflows and Digital Twin style abstractions, and is suitable for university coursework, technical portfolios, and telecom industry applications.
