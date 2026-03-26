# Spatially Consistent CSI-Based Fingerprint Localization and Channel Charting with DeepMIMO

A research-oriented machine learning project for wireless communications focused on:

- **CSI-based fingerprint localization**
- **Channel charting (unsupervised radio geometry learning)**
- **Evaluation of classical and learning-based methods**
- **5G/6G-relevant radio environment analysis using DeepMIMO**

## Motivation

Future wireless systems (5G-Advanced / 6G) increasingly rely on data-driven representations of the radio environment for localization, beam management, and radio resource management. This project explores how **channel state information (CSI)** can be used to:

1. Estimate user position using **fingerprint localization**
2. Learn a low-dimensional **channel chart** that preserves radio geometry

This project is inspired by practical telecom applications in **RAN simulation**, **AI/ML for wireless**, and **digital-twin-style radio environment modeling**.

## Project Objectives

### 1) Dataset and Feature Pipeline
- Load a DeepMIMO scenario
- Extract user positions and CSI
- Build compact CSI features suitable for ML

### 2) Fingerprint Localization
- Implement **WKNN** as a classical baseline
- Implement **MLP regression** as a learning-based baseline
- Evaluate using:
  - RMSE
  - Error CDF
  - Spatial error heatmaps

### 3) Channel Charting
- Build 2D channel charts from CSI features using:
  - **MDS** (classical dimensionality reduction)
  - **UMAP** (modern manifold learning)
- Evaluate using:
  - Trustworthiness (TW)
  - Continuity (CT)
  - Kruskal Stress (KS)

## Repository Structure

```text
data/            # raw and processed data
notebooks/       # exploratory and experiment notebooks
src/             # reusable source code
scripts/         # runnable experiment scripts
results/         # generated figures, metrics, models
report/          # final report draft
slides/          # presentation draft