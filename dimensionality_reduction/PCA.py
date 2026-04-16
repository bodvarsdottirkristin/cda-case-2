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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import SparsePCA

from ..src.features import highly_corr

# preprocess dataset and save in data/processed using functions from src.preprocessing
#TODO

# path to PROCESSED data
df = pd.read_csv(Path('../data/processed/HR_data_2_processed.csv'))

# Biosignal feature distributions — numeric columns, excluding metadata
meta_cols = ['Round', 'Phase', 'Individual', 'Puzzler', 'Cohort']
numeric_cols = df.select_dtypes(include='number').columns.tolist()
biosignal_cols = [c for c in numeric_cols if c not in meta_cols]


#1. reduce number of features
redundant = highly_corr(df[biosignal_cols], perf=0.95)
df_reduced = df.drop(columns=redundant)
remaining_biosignals = [c for c in biosignal_cols if c not in redundant]



# 2. scale features 
scaler = StandardScaler()

# we can have 2 different approaches
# GLOBAL SCALING
abs_df = df.copy()
abs_df[biosignal_cols] = abs_df[biosignal_cols].transform(
    lambda x: scaler.fit_transform(x.values.reshape(-1, 1)).flatten()
)
# PHASE-WISE SCALING depending on the phase; phases can be phase1, phase2 or phase3
phase_df = df.copy()
phase_df[biosignal_cols] = phase_df.groupby('Phase')[biosignal_cols].transform(
    lambda x: scaler.fit_transform(x.values.reshape(-1, 1)).flatten()
)



# 3. sparse PCA
spca = SparsePCA(n_components=5, alpha=1, random_state=42) 
pca_results = spca.fit_transform(df_reduced[remaining_biosignals])

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

# Optional: Add metadata back for group analysis [cite: 69-71, 75-78]
df_pca = pd.concat([df_pca, phase_df[meta_cols]], axis=1)


print(f"Original features: {len(biosignal_cols)}")
print(f"Features after correlation drop: {len(remaining_biosignals)}")
print(f"Dimensions after Sparse PCA: {df_pca.shape[1]}")