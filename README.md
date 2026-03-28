# CSI-Based Fingerprint Localization and Channel Charting for Wireless Networks

A research-grade machine learning project for wireless communications, developed for a Master's-level course and technical portfolio. This project leverages wireless channel features (CSI) for fingerprint localization and channel charting, with a focus on realistic data (DeepMIMO or exported real/simulated datasets). The work is directly relevant to 5G/6G, RAN simulation, and Digital Twin analysis, and includes a professional Streamlit UI for demonstration and exploration.

---

## Key Features

- CSI feature extraction from realistic wireless channel data
- Fingerprint localization using Weighted K-Nearest Neighbors (WKNN)
- Planned: MLP-based localization for learning-based comparison
- Channel charting with MDS (classical) and UMAP (manifold learning)
- Error visualization: RMSE, CDF, and spatial heatmaps
- Interactive, presentation-ready Streamlit dashboard
- Modular, research-oriented codebase for easy extension

---

## Project Demo / UI Preview

The included Streamlit app provides an interactive, professional interface for exploring the project pipeline:

- **Dataset Overview tab**: Visualizes UE spatial distribution, sample count, and feature dimension.
- **Fingerprint Localization tab**: Presents RMSE, error CDF, error heatmap, and predicted vs. true position scatter.
- **Channel Charting tab**: Compares MDS and UMAP embeddings, with color-coded radio geometry visualization.
- **Future Work tab**: Outlines planned extensions, including MLP localization, advanced charting metrics, and multi-BS/beam management.

> The UI is designed for technical presentations, coursework, and portfolio demonstration.

---

## Results / Figures

> Replace these placeholders with your own results and screenshots as you progress.

**Example Figures:**

- ![UE Positions](docs/figures/ue_positions.png)
  - *UE spatial distribution*: Scatter plot of user equipment positions in the dataset.
- ![WKNN Error CDF](docs/figures/wknn_error_cdf.png)
  - *WKNN Error CDF*: Cumulative distribution of localization errors (meters).
- ![WKNN Error Heatmap](docs/figures/wknn_error_heatmap.png)
  - *WKNN Error Heatmap*: Spatial heatmap of localization errors across the test set.
- ![Channel Chart MDS](docs/figures/channel_chart_mds.png)
  - *Channel Chart (MDS)*: 2D embedding of CSI features using MDS, colored by true x/y.
- ![Channel Chart UMAP](docs/figures/channel_chart_umap.png)
  - *Channel Chart (UMAP)*: 2D embedding of CSI features using UMAP, colored by true x/y.
- ![Streamlit UI Overview](docs/figures/streamlit_ui_overview.png)
  - *Streamlit UI*: Overview of the interactive dashboard for dataset exploration and analysis.

---

## Methodology

- **Wireless dataset generation/loading**: Uses DeepMIMO or exported realistic wireless datasets.
- **CSI preprocessing**: Converts complex channel matrices to magnitude-based feature vectors.
- **Fingerprint localization**: Supervised learning with WKNN (baseline) and planned MLP extension.
- **Channel charting**: Unsupervised representation learning with MDS (baseline) and UMAP (comparison).

> Localization predicts true physical coordinates. Channel charting learns a latent radio map—chart axes are not physical coordinates.

---

## Repository Structure

```
app.py
scripts/
src/
data/
docs/figures/
archive/                # (if synthetic baseline scripts were archived)
```

---

## How to Run

1. **Environment setup**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # or .venv\\Scripts\\activate on Windows
   pip install -r requirements.txt
   ```

2. **Prepare real dataset**
   - Run DeepMIMO export and processing scripts as described in scripts/README or project docs.

3. **Launch the Streamlit app**
   ```bash
   python -m streamlit run app.py
   ```

4. **Run localization experiments**
   ```bash
   python -m scripts.run_wknn_real
   ```

5. **Prepare real data (if needed)**
   ```bash
   python -m scripts.prepare_real_data
   ```

---

## Example Output / Interpretation

- **RMSE**: Root Mean Squared Error of localization (meters); lower is better.
- **Error CDF**: Shows the proportion of test samples below each error threshold.
- **Error Heatmap**: Visualizes spatial distribution of localization errors.
- **Channel Chart**: 2D embedding of CSI features; axes are latent, not physical. Color encodes true x or y for geometric interpretation.

> **Note:** Channel chart axes are latent embedding axes, not real-world coordinates.

---

## Why This Project Matters

- **Fingerprint localization**: Enables accurate UE positioning and context awareness in wireless networks.
- **Channel charting**: Learns the underlying radio geometry, supporting advanced RAN analytics.
- **Relevance**: Directly applicable to beam management, mobility management, handover, radio resource management, and Digital Twin abstractions for RAN.
- **Industry impact**: Techniques like these are foundational for 5G/6G, smart RAN, and next-generation telecom systems.

---

## Future Work

- MLP/CNN-based localization
- Trustworthiness, Continuity, Kruskal Stress metrics for charting
- Multi-BS and multi-point channel charting
- Robustness to noisy/estimated CSI
- Beam management and handover use cases
- Out-of-sample extension for channel charting

---

## Tech Stack

- Python
- NumPy
- scikit-learn
- matplotlib
- UMAP (umap-learn)
- Streamlit
- DeepMIMO
- (Optional) MATLAB / ray tracing interoperability

---

> This project is a serious, research-oriented effort in wireless ML, suitable for university coursework, technical portfolios, and internship applications in the telecom industry.