import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, SparsePCA

from utils.high_corr import highly_corr


# =========================
# 0. Load data
# =========================
df = pd.read_csv(Path('../data/raw/data/HR_data_2.csv'))

meta_cols = ['Round', 'Phase', 'Individual', 'Puzzler', 'Cohort']
numeric_cols = df.select_dtypes(include='number').columns.tolist()
biosignal_cols = [c for c in numeric_cols if c not in meta_cols]


# =========================
# 1. Drop highly correlated features
# =========================
redundant = highly_corr(df[biosignal_cols], perf=0.95)
df_reduced = df.drop(columns=redundant)
remaining_biosignals = [c for c in biosignal_cols if c not in redundant]


# =========================
# 2. Scale features phase-wise
# =========================
scaler = StandardScaler()

phase_df = df_reduced.copy()
phase_df[remaining_biosignals] = phase_df.groupby('Phase')[remaining_biosignals].transform(
    lambda x: scaler.fit_transform(x.values.reshape(-1, 1)).flatten()
)

# Matrix for dimensionality reduction
X = phase_df[remaining_biosignals]


# =========================
# 3. Standard PCA to choose number of components
# =========================
pca = PCA()
X_pca = pca.fit_transform(X)

explained_var = pca.explained_variance_ratio_
cum_explained_var = np.cumsum(explained_var)

# Choose threshold
threshold = 0.90
n_components_selected = np.argmax(cum_explained_var >= threshold) + 1

print(f"Original features: {len(biosignal_cols)}")
print(f"Features after correlation drop: {len(remaining_biosignals)}")
print(f"Number of PCA components to explain {threshold:.0%} variance: {n_components_selected}")


# =========================
# 4. Plot explained variance
# =========================
plt.figure(figsize=(8, 5))
plt.plot(
    range(1, len(cum_explained_var) + 1),
    cum_explained_var,
    marker='o'
)
plt.axhline(y=threshold, linestyle='--', label=f'{threshold:.0%} variance threshold')
plt.axvline(x=n_components_selected, linestyle='--', label=f'{n_components_selected} components')

plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.title('PCA cumulative explained variance')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# Optional: scree plot
plt.figure(figsize=(8, 5))
plt.bar(range(1, len(explained_var) + 1), explained_var)
plt.xlabel('Principal component')
plt.ylabel('Explained variance ratio')
plt.title('PCA explained variance by component')
plt.tight_layout()
plt.show()


# =========================
# 5. Fit SparsePCA using chosen number of components
# =========================
spca = SparsePCA(
    n_components=n_components_selected,
    alpha=1,
    random_state=42
)

X_spca = spca.fit_transform(X)

df_spca = pd.DataFrame(
    X_spca,
    columns=[f'PC{i+1}' for i in range(n_components_selected)],
    index=phase_df.index
)

df_spca = pd.concat([df_spca, phase_df[meta_cols]], axis=1)

print(f"Final dataframe shape after SparsePCA + metadata: {df_spca.shape}")