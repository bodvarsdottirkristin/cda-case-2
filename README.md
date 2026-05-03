# CDA Case 2 — EmoPairCompete Clustering Analysis

**Team:** Kristín Böðvarsdóttir, Alessandra Carrara, Kyle Nathan Yahya, NAME4, NAME5, NAME6

---

## Overview

This project analyses the **EmoPairCompete** dataset — physiological biosignals (EDA, HR, TEMP) collected from participants wearing Empatica E4 wristbands during a competitive puzzle game. The goal is to discover latent physiological arousal states through unsupervised clustering and assess whether they align with known experimental conditions (game phase, cohort, role).

The project is split into two parts:
- **Main analysis** — clustering on statistical summaries (`final/`)
- **Advanced topic** — clustering on raw time-series via autoencoders (`advanced/`)

---

## Repository Structure

The repository contains both exploratory development work and the final clean pipeline.
**The submission code lives in `final/` and `advanced/`.**
All other top-level folders (`clustering/`, `dim_reduction/`, `gmm/`) are exploratory and not part of the submission.

```
cda-case-2/
├── final/                        ← MAIN ANALYSIS
│   ├── clustering/
│   │   ├── best_combination.py   ← K-Means, K-Medoids, GMM leaderboard
│   │   └── clustering_reduction.py
│   ├── dendrogram/
│   │   └── dendrogram_biosignals.py  ← Hierarchical clustering
│   ├── gmm/
│   │   └── gmm_biosignals.py     ← GMM full analysis
│   ├── pipeline_diagram/
│   │   └── generate_diagram.py   ← Figure 1
│   ├── cluster_profiles.py       ← Cluster z-score summary (SparsePCA GMM)
│   └── full_leaderboard.py       ← All 15 combinations + signal rankings
│
├── advanced/                     ← ADVANCED TOPIC
│   ├── v1_autoencoding.py        ← Conv1D AE (global normalisation)
│   ├── v2_autoencoding.py        ← Conv1D AE (cohort normalisation)
│   ├── autoencoding_lstm.py      ← LSTM AE
│   ├── utils/                    ← Shared model and data loading code
│   └── outputs/                  ← Experiment results (ARI/NMI summaries)
│
├── data/
│   ├── raw/                      ← Original CSVs (not committed)
│   └── processed/                ← HR_data_2.csv and reduced representations
│
├── docs/                         ← Report versions
├── requirements.txt
└── README.md
```

---

## Setup

```bash
git clone https://github.com/bodvarsdottirkristin/cda-case-2.git
cd cda-case-2
pip install -r requirements.txt
```

Place the raw EmoPairCompete data in `data/raw/` following the original dataset structure.

---

## Running the Main Analysis

All scripts are run from the repository root.

```bash
# DBSCAN — grid search across all three reductions
python final/dbscan/dbscan_biosignals.py

# K-Means, K-Medoids, GMM — corrected leaderboard
python final/clustering/best_combination.py

# Hierarchical clustering (Ward linkage)
python final/dendrogram/dendrogram_biosignals.py

# GMM — full BIC grid search across covariance types
python final/gmm/gmm_biosignals.py

# All 15 combinations leaderboard with signal-type rankings
python final/full_leaderboard.py

# Cluster z-score profiles (SparsePCA GMM k=3)
python final/cluster_profiles.py

# Pipeline diagram (requires graphviz system package)
python final/pipeline_diagram/generate_diagram.py
```

---

## Running the Advanced Analysis

```bash
# Conv1D autoencoder with cohort normalisation (main advanced result)
python advanced/v2_autoencoding.py

# LSTM autoencoder
python advanced/autoencoding_lstm.py
```

Results are saved to `advanced/outputs/`. Each output directory contains a
`eval/summary.txt` with full ARI/NMI alignment metrics.

---

## Dependencies

See `requirements.txt`. Key packages: `scikit-learn`, `scikit-learn-extra`
(K-Medoids), `umap-learn`, `graphviz`, `torch` (autoencoders).
