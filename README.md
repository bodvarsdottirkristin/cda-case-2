# Creators:
Kristín Böðvarsdóttir

Alessandra  Carrara

Kyle Nathan Yahya

NAME4

NAME5

# EmoPairCompete Physiological Data Analysis

## Overview
This project analyses the **EmoPairCompete** dataset — physiological biosignals (EDA, HR, TEMP, BVP) collected from participants wearing **Empatica E4** wristbands during a stress-inducing tangram-puzzle competition.

---

## Dataset Description
Data are organised in a nested directory hierarchy:

```
data/raw/
└── <cohort>/          # D11, D12, D13–D16
    └── <participant>/ # e.g. P01, P02 …
        └── <round>/   # 1, 2, 3, 4
            └── <phase>/  # pre, puzzle, post
                ├── BVP.csv
                └── EDA.csv
                └── HR.csv
                └── response.csv
                └── TEMP.csv
```

## Project Structure
```
cda-case-2/
├── data/
│   ├── raw/           # original CSVs (not committed to git)
│   └── processed/     # cleaned & merged outputs
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_analysis.ipynb
│   └── 03_modelling.ipynb
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── features.py
│   └── model.py
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py
│   ├── test_features.py
│   └── test_model.py
├── results/
│   ├── figures/
│   └── tables/
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/bodvarsdottirkristin/cda-case-2.git
cd cda-case-2
```

### 2. Create and activate a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Place the raw data
Copy the EmoPairCompete CSV files into `data/raw/` following the directory structure shown above.

---

## Running the Tests
```bash
pytest tests/
```

---

## Notebooks
| Notebook | Purpose |
|----------|---------|
| `01_eda.ipynb` | Exploratory data analysis — distributions, correlations |
| `02_feature_analysis.ipynb` | Feature importance, dimensionality reduction |
| `03_modelling.ipynb` | Predictive models for stress classification |
