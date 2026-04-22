import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA, SparsePCA

# =========================
# 0. Paths and imports
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = Path(__file__).resolve().parent / 'figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
sys.path.append(str(PROJECT_ROOT))

from dim_reduction.utils.high_corr import highly_corr


# =========================
# 1. Load already processed data
# =========================
input_path = PROJECT_ROOT / 'data' / 'processed' / 'HR_data_2.csv'
processed_dir = PROJECT_ROOT / 'data' / 'processed'
processed_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(input_path)

# Metadata columns
meta_cols = ['Round', 'Phase', 'Individual', 'Puzzler', 'Cohort']

# Questionnaire columns to exclude from biosignals
questionnaire_cols = [
    'Frustrated', 'upset', 'hostile', 'alert', 'ashamed', 'inspired',
    'nervous', 'attentive', 'afraid', 'active', 'determined'
]

# Numeric biosignal columns only
numeric_cols = df.select_dtypes(include='number').columns.tolist()
biosignal_cols = [c for c in numeric_cols if c not in meta_cols + questionnaire_cols]


# =========================
# 2. Drop highly correlated features
# =========================
redundant = highly_corr(df[biosignal_cols], perf=0.95)
remaining_biosignals = [c for c in biosignal_cols if c not in redundant]

print(f"Original features: {len(biosignal_cols)}")
print(f"Features after correlation drop: {len(remaining_biosignals)}")


# =========================
# 3. Scale features phase-wise
# =========================
phase_df = df.copy()

def safe_standardize(series):
    std = series.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - series.mean()) / std

for col in remaining_biosignals:
    phase_df[col] = phase_df.groupby('Phase')[col].transform(safe_standardize)

X = phase_df[remaining_biosignals]

print("Remaining NaNs before PCA:", X.isna().sum().sum())
if X.isna().sum().sum() > 0:
    raise ValueError("There are still NaNs in X before PCA.")


# =========================
# 4. Standard PCA to choose number of components
# =========================
pca = PCA()
X_pca = pca.fit_transform(X)

explained_var = pca.explained_variance_ratio_
cum_explained_var = np.cumsum(explained_var)

variance_threshold = 0.80
n_components_selected = np.argmax(cum_explained_var >= variance_threshold) + 1

print(f"Number of PCA components to explain {variance_threshold:.0%} variance: {n_components_selected}")


# =========================
# 5. Plot explained variance
# =========================
plt.figure(figsize=(8, 5))
plt.plot(
    range(1, len(cum_explained_var) + 1),
    cum_explained_var,
    marker='o'
)
plt.axhline(
    y=variance_threshold,
    linestyle='--',
    label=f'{variance_threshold:.0%} variance threshold'
)
plt.axvline(
    x=n_components_selected,
    linestyle='--',
    label=f'{n_components_selected} components'
)
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.title('PCA cumulative explained variance')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'pca_cumulative_variance.png', dpi=300)
plt.show()

plt.figure(figsize=(8, 5))
plt.bar(range(1, len(explained_var) + 1), explained_var)
plt.xlabel('Principal component')
plt.ylabel('Explained variance ratio')
plt.title('PCA explained variance by component')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'pca_scree_plot.png', dpi=300)
plt.show()


# =========================
# 6. Fit SparsePCA using chosen number of components
# =========================
spca = SparsePCA(
    n_components=n_components_selected,
    alpha=1,
    random_state=42
)

X_spca = spca.fit_transform(X)


# =========================
# 7. Inspect SparsePCA components
# =========================
components = pd.DataFrame(
    spca.components_,
    columns=remaining_biosignals,
    index=[f'PC{i+1}' for i in range(n_components_selected)]
)

loading_threshold = 1e-5

for pc in components.index:
    print(f"\n{pc}:")
    selected = components.loc[pc][abs(components.loc[pc]) > loading_threshold]
    selected = selected.sort_values(key=abs, ascending=False)
    print(selected)


# =========================
# 8. Save SparsePCA scores with metadata
# =========================
df_spca = pd.DataFrame(
    X_spca,
    columns=[f'PC{i+1}' for i in range(n_components_selected)],
    index=phase_df.index
)

df_spca = pd.concat([df_spca, phase_df[meta_cols]], axis=1)

df_spca.to_csv(processed_dir / 'HR_data_sparse_pca.csv', index=False)

print(f"Final dataframe shape after SparsePCA + metadata: {df_spca.shape}")