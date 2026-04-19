import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

# =========================
# 0. Paths and imports
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = Path(__file__).resolve().parent / 'figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
sys.path.append(str(PROJECT_ROOT))

from dim_reduction.utils.high_corr import highly_corr


# =========================
# 1. Load data
# =========================
input_path = PROJECT_ROOT / 'data' / 'processed' / 'HR_data_2.csv'
processed_dir = PROJECT_ROOT / 'data' / 'processed'
processed_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(input_path)

meta_cols = ['Round', 'Phase', 'Individual', 'Puzzler', 'Cohort']
questionnaire_cols = ['Frustrated', 'upset', 'hostile', 'alert', 'ashamed', 'inspired',
                      'nervous', 'attentive', 'afraid', 'active', 'determined']

numeric_cols = df.select_dtypes(include='number').columns.tolist()
biosignal_cols = [c for c in numeric_cols if c not in meta_cols + questionnaire_cols]

# =========================
# 3. Drop highly correlated features
# =========================
redundant = highly_corr(df[biosignal_cols], perf=0.95)
remaining_biosignals = [c for c in biosignal_cols if c not in redundant]

print(f"Original features: {len(biosignal_cols)}")
print(f"Features after correlation drop: {len(remaining_biosignals)}")


# =========================
# 4. Scale features phase-wise
# =========================
phase_df = df.copy()

def safe_standardize(series):
    std = series.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - series.mean()) / std

for col in remaining_biosignals:
    phase_df[col] = phase_df.groupby('Phase')[col].transform(safe_standardize)

X = phase_df[remaining_biosignals].values
y = StandardScaler().fit_transform(phase_df[questionnaire_cols])


print("NaNs in X:", np.isnan(X).sum())
print("NaNs in y:", np.isnan(y).sum())
print("X shape:", X.shape)
print("y shape:", y.shape)

print("X min:", X.min(), "X max:", X.max())
print("X mean:", X.mean(), "X std:", X.std())


# =========================
# 5. Cross-validation to choose n_components
# =========================
pls = PLSRegression()
n_components = range(1, min(21, len(remaining_biosignals) + 1))
param_grid = {'n_components': n_components}

pls_grid = GridSearchCV(
    estimator=pls,
    param_grid=param_grid,
    cv=5,
    verbose=2,
    n_jobs=-1,
    scoring='neg_mean_squared_error',
    refit='neg_mean_squared_error'
)
pls_grid.fit(X, y)



n_components_selected = pls_grid.best_params_['n_components']
print(f"Best n_components: {n_components_selected}")

# =========================
# 6. Plot MSE vs n_components
# =========================
pls_mse = pls_grid.cv_results_['mean_test_score'] * -1

for i, (n, mse) in enumerate(zip(n_components, pls_mse)):
    print(f"n_components={n}: MSE={mse:.4f}")


print(f"Best MSE: {min(pls_mse):.4f} at n_components={n_components_selected}")

plt.figure(figsize=(8, 5))
plt.plot(n_components, pls_mse, marker='o', label='MSE')
plt.axvline(x=n_components_selected, linestyle='--', label=f'{n_components_selected} components')
plt.xlabel('Number of components')
plt.ylabel('Mean CV score')
plt.title('PLS cross-validation scores')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'pls_cv_scores.png', dpi=300)


# =========================
# 7. Fit PLS with chosen n_components and inspect coefficients
# =========================
pls_final = PLSRegression(n_components=n_components_selected)
pls_final.fit(X, y)

coef = pls_final.coef_.mean(axis=0) / np.std(X, axis=0)


# =========================
# 8. Save PLS scores with metadata
# =========================
X_scores, _ = pls_final.transform(X, y)

df_pls = pd.DataFrame(
    X_scores,
    columns=[f'PLS{i+1}' for i in range(n_components_selected)],
    index=phase_df.index
)
df_pls = pd.concat([df_pls, phase_df[meta_cols]], axis=1)


print(f"Final dataframe shape after PLS + metadata: {df_pls.shape}")

plt.figure(figsize=(15, 6))
plt.stem(range(len(remaining_biosignals)), coef)
plt.xticks(range(len(remaining_biosignals)), remaining_biosignals, rotation=90)
plt.xlabel('Biosignal feature')
plt.ylabel('Relative importance in PLS model')
plt.title('PLS feature coefficients')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'pls_coefficients.png', dpi=300)

# Fit PLS with max components to see all at once
pls_full = PLSRegression(n_components=20)
pls_full.fit(X, y)

# Variance in X explained by each component
x_var = np.var(pls_full.x_scores_, axis=0)
x_var_ratio = x_var / np.sum(np.var(X, axis=0))

plt.figure(figsize=(8, 5))
plt.bar(range(1, 21), x_var_ratio)
plt.xlabel('PLS component')
plt.ylabel('Variance ratio (X)')
plt.title('PLS variance explained in X by component')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'pls_scree_plot.png', dpi=300)
plt.show()


df_pls.to_csv(processed_dir / 'HR_data_pls.csv', index=False)

