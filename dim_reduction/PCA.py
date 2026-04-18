"""
Script to reduce the dimensionality of processed dataset HR_data_2 (no NaN values), which comes with 307 rows, 70 columns.
Strategy:

1. drop high correlation - some features are very higly correlated, like HR_min, HR_max, HR_std, (first top-left red square in the correlation plot)

2. scaling: we scale data wrt to the phase they're part of. We want the mean to be referred to the average of that particular phase, and not to 
all phases simoultaneously - this way we avoid the clustering to simply detect different phases afterwards!!
With global scaling, we risk to induce the clustering to group only wrt to phases, bcs these show a significant difference in terms of stress / 
heart rate...
Insteas, if we scale wrt to each phase, cluster will focus more on "latent" emotional aspects (I hope)

3. sparse PCA or standard PCA ? I believe the first one is better in our case, bcs it allows to set some features to 0
I use the function from sklearn, but in week8 exercises it is done explicitely

"""
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import SparsePCA, PCA
from sklearn.metrics import mean_squared_error

from src.features import highly_corr

# original dataset
# This finds the directory where PCA.py is, then goes up to the root, then into data
base_path = Path(__file__).resolve().parent.parent
df = pd.read_csv(base_path / 'data' / 'raw' / 'data' / 'HR_data_2.csv')

# Biosignal feature distributions — numeric columns, excluding metadata
meta_cols = ['Round', 'Phase', 'Individual', 'Puzzler', 'Cohort']
numeric_cols = df.select_dtypes(include='number').columns.tolist()
biosignal_cols = [c for c in numeric_cols if c not in meta_cols]

#0. we have only 9 missing values, so we can easily impute them in one line
# we use the mean, but to be coherent we look at the average within each group (otherwise weird results)
df[biosignal_cols] = df.groupby('Phase')[biosignal_cols].transform(lambda x: x.fillna(x.mean()))


#1. reduce number of features
redundant = highly_corr(df[biosignal_cols], perf=0.90)
df_reduced = df.drop(columns=redundant)
remaining_biosignals = [c for c in biosignal_cols if c not in redundant]
df_reduced.to_csv(Path('data/processed/HR_data_reduced.csv'), index=False)


# 2. scale features on entire df
scaler = StandardScaler()

# we can have 2 different approaches
# GLOBAL SCALING
abs_df = df.copy()  # full df
abs_df[biosignal_cols] = abs_df[biosignal_cols].transform(
    lambda x: scaler.fit_transform(x.values.reshape(-1, 1)).flatten()
)
# PHASE-WISE SCALING depending on the phase; phases can be phase1, phase2 or phase3
phase_df = df.copy()
phase_df[biosignal_cols] = phase_df.groupby('Phase')[biosignal_cols].transform(
    lambda x: scaler.fit_transform(x.values.reshape(-1, 1)).flatten()
)



# 3. sparse PCA on reduced dataset
spca = SparsePCA(n_components=5, alpha=1, random_state=42) 
pca_results = spca.fit_transform(phase_df[remaining_biosignals])

# Convert to DataFrame for easy clustering
df_pca = pd.DataFrame(
    pca_results, 
    columns=[f'PC{i+1}' for i in range(5)],
    index=df_reduced.index
)

df_pca = pd.DataFrame(
    pca_results, 
    columns=[f'PC{i+1}' for i in range(5)],
    index=phase_df.index
)

df_pca = pd.concat([df_pca, phase_df[meta_cols]], axis=1)
df_pca.to_csv(Path('data/processed/HR_data_PCA.csv'), index=False)


print(f"Original features: {len(biosignal_cols)}")
print(f"Features after correlation drop: {len(remaining_biosignals)}")
print(f"Dimensions after Sparse PCA: {df_pca.shape[1]}")

#----------------------------------------------------------
# test our results

# Test standard PCA - is sparse PCA good enough? Is the difference between the 2 negligible?

# Standard PCA
X = phase_df[remaining_biosignals]
pca_5 = PCA(n_components=5, random_state=42).fit(X)

# Calculate Variance Explained - built in funciton
# Standard PCA (built-in)
var_pca = np.sum(pca_5.explained_variance_ratio_)

# Sparse PCA (Manual) - using the projection of our already fitted 'spca'
def calc_var(original, components):
    reconstructed = (original @ components.T) @ np.linalg.pinv(components).T
    return 1 - np.var(original.values - reconstructed.values) / np.var(original.values)

var_spca = calc_var(X, spca.components_)

# 3. Calculate RMSE (Root Mean Squared Error)
# We use RMSE because it has the same units as our scaled data
pca_rmse = np.sqrt(mean_squared_error(X, pca_5.inverse_transform(pca_5.transform(X))))
spca_rmse = np.sqrt(mean_squared_error(X, (X @ spca.components_.T) @ np.linalg.pinv(spca.components_).T))

# --- OUTPUTS ---

print("\n--- Rigorous Comparison (5 Components) ---")
print(f"Explained Var | PCA: {var_pca:.4f} vs SPCA: {var_spca:.4f}")
print(f"RMSE          | PCA: {pca_rmse:.4f} vs SPCA: {spca_rmse:.4f}")

# Check Sparsity vs Interpretability
print("\n--- Sparsity Check ---")
for i, comp in enumerate(spca.components_):
    print(f"PC{i+1} utilizes {np.sum(comp != 0)}/{len(remaining_biosignals)} features")

# Final Phase-Overlap Check
sns.pairplot(df_pca, vars=['PC1', 'PC2', 'PC3'], hue='Phase', palette='viridis')
plt.show()

# SPCA seems to perform very well, and doesn't sacrifice anything compared to PCA