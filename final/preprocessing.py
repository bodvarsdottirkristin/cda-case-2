import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, SparsePCA
import umap

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from dim_reduction.utils.high_corr import highly_corr

OUTPUT_DIR = PROJECT_ROOT / 'data' / 'processed' / 'final'

META_COLS = ['Round', 'Phase', 'Individual', 'Puzzler', 'Cohort']
QUEST_COLS = ['Frustrated', 'upset', 'hostile', 'alert', 'ashamed',
              'inspired', 'nervous', 'attentive', 'afraid', 'active', 'determined']


def load_and_clean(path):
    """Load CSV data and clean biosignal features.

    Args:
        path: Path to HR_data_2.csv

    Returns:
        tuple: (df, biosignal_cols, meta_present, quest_present)
            - df: DataFrame with redundant biosignals dropped
            - biosignal_cols: List of remaining biosignal column names
            - meta_present: List of metadata columns present in df
            - quest_present: List of questionnaire columns present in df
    """
    df = pd.read_csv(path)

    # Identify numeric columns and biosignals
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    biosignal_cols = [c for c in numeric_cols if c not in META_COLS + QUEST_COLS]

    # Fill missing values within each Phase group, fall back to global mean
    df[biosignal_cols] = df.groupby('Phase')[biosignal_cols].transform(
        lambda x: x.fillna(x.mean())
    )
    df[biosignal_cols] = df[biosignal_cols].fillna(df[biosignal_cols].mean())

    # Drop highly correlated biosignal columns
    redundant = highly_corr(df[biosignal_cols], perf=0.95)
    remaining = [c for c in biosignal_cols if c not in redundant]
    df = df.drop(columns=redundant)

    # Return present metadata and questionnaire columns
    meta_present = [c for c in META_COLS if c in df.columns]
    quest_present = [c for c in QUEST_COLS if c in df.columns]

    return df, remaining, meta_present, quest_present


def _safe_standardize(series):
    """Standardize a series to zero mean and unit variance, handling zero std.

    Args:
        series: pd.Series to standardize

    Returns:
        pd.Series: Standardized series (or zeros if std is 0 or NaN)
    """
    std = series.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - series.mean()) / std


def normalize_by_individual(df, biosignal_cols):
    """Normalize biosignal columns within each individual.

    Standardizes each biosignal column to zero mean and unit variance
    separately for each individual, preserving all other columns.

    Args:
        df: DataFrame containing biosignal data and 'Individual' column
        biosignal_cols: List of column names to normalize

    Returns:
        pd.DataFrame: Copy of df with normalized biosignal columns
    """
    df_norm = df.copy()
    for col in biosignal_cols:
        df_norm[col] = df_norm.groupby('Individual')[col].transform(_safe_standardize)
    return df_norm


def run_pca(df, biosignal_cols):
    """Apply PCA to biosignal data, retaining components up to 80% variance.

    Fits a PCA model to the biosignal columns and selects the minimum
    number of components needed to explain at least 80% of the variance.

    Args:
        df: DataFrame containing the biosignal data
        biosignal_cols: List of biosignal column names to apply PCA on

    Returns:
        tuple: (df_pca, n_components)
            - df_pca: DataFrame with PCA-transformed columns named PC1, PC2, ...
            - n_components: Number of components selected
    """
    X = df[biosignal_cols]
    pca_full = PCA()
    pca_full.fit(X)
    cum_var = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = int(np.argmax(cum_var >= 0.80)) + 1
    print(f"PCA: selected {n_components} components ({cum_var[n_components-1]:.1%} variance explained)")

    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X)

    df_pca = pd.DataFrame(
        X_pca,
        columns=[f'PC{i+1}' for i in range(n_components)],
        index=df.index
    )
    return df_pca, n_components


def run_spca(df, biosignal_cols, n_components):
    """Apply SparsePCA to biosignal data with a fixed number of components.

    Fits a SparsePCA model to the biosignal columns with the specified number
    of components. SparsePCA introduces sparsity to the components, making
    them more interpretable by using only a subset of features.

    Args:
        df: DataFrame containing the biosignal data
        biosignal_cols: List of biosignal column names to apply SparsePCA on
        n_components: Number of sparse components to extract

    Returns:
        pd.DataFrame: DataFrame with SparsePCA-transformed columns named PC1, PC2, ...
    """
    X = df[biosignal_cols]
    spca = SparsePCA(n_components=n_components, alpha=1, random_state=42)
    X_spca = spca.fit_transform(X)

    df_spca = pd.DataFrame(
        X_spca,
        columns=[f'PC{i+1}' for i in range(n_components)],
        index=df.index
    )
    return df_spca


def run_umap(df, biosignal_cols):
    """Apply UMAP to biosignal data, reducing to 10 components.

    Fits a UMAP (Uniform Manifold Approximation and Projection) model to the
    biosignal columns and transforms the data into a 10-dimensional space.
    UMAP preserves both local and global structure in the data.

    Args:
        df: DataFrame containing the biosignal data
        biosignal_cols: List of biosignal column names to apply UMAP on

    Returns:
        pd.DataFrame: DataFrame with UMAP-transformed columns named UMAP1, UMAP2, ...
    """
    X = df[biosignal_cols]
    reducer = umap.UMAP(
        n_components=10,
        n_neighbors=50,
        min_dist=0.0,
        metric='euclidean',
        random_state=42
    )
    X_umap = reducer.fit_transform(X)

    df_umap = pd.DataFrame(
        X_umap,
        columns=[f'UMAP{i+1}' for i in range(10)],
        index=df.index
    )
    return df_umap


def save_output(reduced_df, df_source, meta_cols, questionnaire_cols, path):
    """Save dimensionality-reduced data with metadata and questionnaire columns.

    Concatenates the reduced-dimensionality features with metadata and
    questionnaire columns, then writes to a CSV file.

    Args:
        reduced_df: DataFrame with reduced-dimensionality features (e.g., PCA components)
        df_source: Original DataFrame containing metadata and questionnaire columns
        meta_cols: List of metadata column names to include
        questionnaire_cols: List of questionnaire column names to include
        path: Path object where the output CSV will be written
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df_out = pd.concat([
        reduced_df.reset_index(drop=True),
        df_source[meta_cols].reset_index(drop=True),
        df_source[questionnaire_cols].reset_index(drop=True),
    ], axis=1)
    df_out.to_csv(path, index=False)
    print(f"Saved: {path}  ({df_out.shape[0]} rows x {df_out.shape[1]} cols)")


def main():
    """Execute the full biosignal preprocessing pipeline.

    Loads raw data, normalizes by individual, and applies three dimensionality
    reduction techniques (PCA, SparsePCA, UMAP), saving the results to separate CSV files.
    """
    data_path = PROJECT_ROOT / 'data' / 'processed' / 'HR_data_2.csv'

    print("=== Loading and cleaning ===")
    df, biosig, meta, quest = load_and_clean(data_path)
    print(f"Features after correlation drop: {len(biosig)}")

    print("\n=== Normalizing by individual ===")
    df_norm = normalize_by_individual(df, biosig)
    print(f"NaNs after normalization: {df_norm[biosig].isna().sum().sum()}")

    print("\n=== Running PCA ===")
    df_pca_reduced, n_components = run_pca(df_norm, biosig)
    save_output(df_pca_reduced, df_norm, meta, quest, OUTPUT_DIR / 'HR_data_pca.csv')

    print("\n=== Running SparsePCA ===")
    df_spca_reduced = run_spca(df_norm, biosig, n_components)
    save_output(df_spca_reduced, df_norm, meta, quest, OUTPUT_DIR / 'HR_data_spca.csv')

    print("\n=== Running UMAP ===")
    df_umap_reduced = run_umap(df_norm, biosig)
    save_output(df_umap_reduced, df_norm, meta, quest, OUTPUT_DIR / 'HR_data_umap.csv')

    print("\n=== Done ===")


if __name__ == '__main__':
    main()
