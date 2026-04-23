import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# =========================
# 0. Paths and imports
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = Path(__file__).resolve().parent / 'figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
sys.path.append(str(PROJECT_ROOT))

DATASET_DIR = PROJECT_ROOT / 'data' / 'raw' / 'data' / 'dataset'
processed_dir = PROJECT_ROOT / 'data' / 'processed'
processed_dir.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =========================
# 1. Load raw signals
# =========================
# BVP is excluded: at 64Hz it aliases badly when downsampled to 1Hz
SIGNALS = ['HR', 'EDA', 'TEMP']


def load_phase(phase_dir: Path, cohort: str, individual: str, round_: str, phase: str) -> pd.DataFrame:
    """Load and merge signals for one phase, resampled to 1-second intervals."""
    dfs = {}
    for sig in SIGNALS:
        csv = phase_dir / f'{sig}.csv'
        if csv.exists():
            s = pd.read_csv(csv, index_col=0)
            s['time'] = pd.to_datetime(s['time'])
            s = s.set_index('time').sort_index()
            dfs[sig] = s[sig]

    if not dfs:
        return pd.DataFrame()

    resampled = {sig: s.resample('1s').mean() for sig, s in dfs.items()}
    merged = pd.concat(resampled, axis=1)
    merged.index.name = 'time'
    merged = merged.reset_index()

    merged['Cohort'] = cohort
    merged['Individual'] = individual
    merged['Round'] = round_
    merged['Phase'] = phase

    response_csv = phase_dir / 'response.csv'
    if response_csv.exists():
        resp = pd.read_csv(response_csv, index_col=0)
        meta_cols = ('particpant_ID', 'participant_ID', 'puzzler', 'team_ID', 'E4_nr')
        questionnaire_cols = [c for c in resp.columns if c not in meta_cols]
        for col in questionnaire_cols:
            merged[col] = resp[col].iloc[0] if len(resp) > 0 else np.nan
        puzzler_col = 'puzzler' if 'puzzler' in resp.columns else None
        merged['Puzzler'] = resp[puzzler_col].iloc[0] if (puzzler_col and len(resp) > 0) else np.nan

    return merged


print("Loading raw signals...")
records = []
for cohort_dir in sorted(DATASET_DIR.iterdir()):
    if not cohort_dir.is_dir():
        continue
    cohort = cohort_dir.name
    for id_dir in sorted(cohort_dir.iterdir()):
        if not id_dir.is_dir() or not id_dir.name.startswith('ID_'):
            continue
        individual = id_dir.name
        for round_dir in sorted(id_dir.iterdir()):
            if not round_dir.is_dir():
                continue
            round_ = round_dir.name
            for phase_dir in sorted(round_dir.iterdir()):
                if not phase_dir.is_dir():
                    continue
                phase = phase_dir.name
                phase_df = load_phase(phase_dir, cohort, individual, round_, phase)
                if not phase_df.empty:
                    records.append(phase_df)

df = pd.concat(records, ignore_index=True)
print(f"Loaded {len(df):,} rows from {len(records)} phases")


# =========================
# 2. Windowing
# =========================
WINDOW_SIZE = 60   # seconds
STEP_SIZE   = 30   # 50% overlap

signal_cols = SIGNALS
meta_keys   = ['Cohort', 'Individual', 'Round', 'Phase']

windows, window_meta = [], []

for (cohort, individual, round_, phase), grp in df.groupby(meta_keys):
    sig = grp[signal_cols].values.astype(np.float32)
    sig = sig[~np.all(np.isnan(sig), axis=1)]
    n = len(sig)
    for start in range(0, n - WINDOW_SIZE + 1, STEP_SIZE):
        w = sig[start:start + WINDOW_SIZE]
        if not np.any(np.isnan(w)):
            windows.append(w)
            window_meta.append({
                'Cohort': cohort, 'Individual': individual,
                'Round': round_, 'Phase': phase
            })

X_raw = np.stack(windows)           # (N, window_size, n_signals)
meta_df = pd.DataFrame(window_meta)
print(f"Windows: {X_raw.shape}  -  {X_raw.shape[0]} windows x {WINDOW_SIZE}s x {len(signal_cols)} signals")


# =========================
# 3. Normalize per signal channel
# =========================
N, T, C = X_raw.shape
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw.reshape(-1, C)).reshape(N, T, C)

# LSTM expects (batch, seq_len, features)
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
dataset   = TensorDataset(X_tensor)
loader    = DataLoader(dataset, batch_size=64, shuffle=True)


# =========================
# 4. LSTM Autoencoder
# =========================
LATENT_DIM  = 32
HIDDEN_SIZE = 64
NUM_LAYERS  = 2


class LSTMEncoder(nn.Module):
    def __init__(self, n_features: int, hidden_size: int, num_layers: int, latent_dim: int):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_size, num_layers,
                            batch_first=True, dropout=0.1 if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_size, latent_dim)

    def forward(self, x):
        # x: (batch, seq_len, features)
        _, (h_n, _) = self.lstm(x)
        # h_n: (num_layers, batch, hidden_size) - take last layer's hidden state
        z = self.fc(h_n[-1])
        return z


class LSTMDecoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_size: int, num_layers: int,
                 n_features: int, seq_len: int):
        super().__init__()
        self.seq_len = seq_len
        self.fc = nn.Linear(latent_dim, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers,
                            batch_first=True, dropout=0.1 if num_layers > 1 else 0.0)
        self.out = nn.Linear(hidden_size, n_features)

    def forward(self, z):
        # Broadcast latent vector as input at every timestep
        x = self.fc(z).unsqueeze(1).repeat(1, self.seq_len, 1)
        out, _ = self.lstm(x)
        return self.out(out)   # (batch, seq_len, n_features)


class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features: int, hidden_size: int, num_layers: int,
                 latent_dim: int, seq_len: int):
        super().__init__()
        self.encoder = LSTMEncoder(n_features, hidden_size, num_layers, latent_dim)
        self.decoder = LSTMDecoder(latent_dim, hidden_size, num_layers, n_features, seq_len)

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z

    def encode(self, x):
        return self.encoder(x)


model     = LSTMAutoencoder(C, HIDDEN_SIZE, NUM_LAYERS, LATENT_DIM, T).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

print(f"\nTraining LSTM autoencoder on {DEVICE}...")
EPOCHS = 50
train_losses = []

for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    for (batch,) in loader:
        batch = batch.to(DEVICE)
        recon, _ = model(batch)
        loss = criterion(recon, batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * len(batch)
    avg = epoch_loss / len(dataset)
    train_losses.append(avg)
    if epoch % 5 == 0:
        print(f"  Epoch {epoch:3d}/{EPOCHS}  loss={avg:.5f}")


# =========================
# 5. Extract latent vectors
# =========================
model.eval()
latents = []
with torch.no_grad():
    for (batch,) in DataLoader(dataset, batch_size=256):
        z = model.encode(batch.to(DEVICE))
        latents.append(z.cpu().numpy())

Z = np.concatenate(latents)   # (N_windows, LATENT_DIM)
print(f"\nLatent matrix: {Z.shape}")


# =========================
# 6. Aggregate latent vectors per phase
# =========================
meta_df['latent_idx'] = np.arange(len(meta_df))
phase_latents, phase_meta = [], []

for keys, grp in meta_df.groupby(meta_keys):
    phase_latents.append(Z[grp.index].mean(axis=0))
    phase_meta.append(dict(zip(meta_keys, keys)))

Z_phase    = np.stack(phase_latents)   # (n_phases, LATENT_DIM)
phase_meta = pd.DataFrame(phase_meta)
print(f"Phase-level latent matrix: {Z_phase.shape}")


# =========================
# 7. t-SNE
# =========================
print("\nRunning t-SNE...")
tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
Z_2d = tsne.fit_transform(Z_phase)


# =========================
# 8. K-Means clustering on latent space
# =========================
sil_scores = {}
for k in range(2, 9):
    km  = KMeans(n_clusters=k, random_state=42, n_init=10)
    lbl = km.fit_predict(Z_phase)
    sil_scores[k] = silhouette_score(Z_phase, lbl)

best_k = max(sil_scores, key=sil_scores.get)
print(f"Silhouette scores: { {k: f'{v:.3f}' for k, v in sil_scores.items()} }")
print(f"Best k: {best_k}")

km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
phase_meta['Cluster'] = km_final.fit_predict(Z_phase)


# =========================
# 9. Figures
# =========================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# training loss
axes[0].plot(range(1, EPOCHS + 1), train_losses, color='steelblue')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('MSE Loss')
axes[0].set_title('LSTM Autoencoder Training Loss')

# silhouette scores
axes[1].bar(list(sil_scores.keys()), list(sil_scores.values()), color='steelblue')
axes[1].axvline(best_k, color='tomato', linestyle='--', label=f'Best k={best_k}')
axes[1].set_xlabel('k')
axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('K-Means Silhouette Scores')
axes[1].legend()

# t-SNE coloured by cluster
scatter = axes[2].scatter(
    Z_2d[:, 0], Z_2d[:, 1],
    c=phase_meta['Cluster'], cmap='tab10', s=40, alpha=0.8
)
plt.colorbar(scatter, ax=axes[2], label='Cluster')
axes[2].set_title(f't-SNE of Phase Latents  (k={best_k})')
axes[2].set_xlabel('t-SNE 1')
axes[2].set_ylabel('t-SNE 2')

plt.tight_layout()
out = FIGURES_DIR / 'lstm_autoencoder_tsne.png'
plt.savefig(out, dpi=150)
print(f"\nFigure saved -> {out}")

# t-SNE coloured by cohort
fig2, ax2 = plt.subplots(figsize=(8, 6))
cohorts = sorted(phase_meta['Cohort'].unique())
cmap    = plt.get_cmap('tab10')
for i, cohort in enumerate(cohorts):
    mask = phase_meta['Cohort'] == cohort
    ax2.scatter(Z_2d[mask, 0], Z_2d[mask, 1], label=cohort,
                color=cmap(i), s=40, alpha=0.8)
ax2.legend(title='Cohort')
ax2.set_title('t-SNE coloured by Cohort')
ax2.set_xlabel('t-SNE 1')
ax2.set_ylabel('t-SNE 2')
plt.tight_layout()
out2 = FIGURES_DIR / 'lstm_autoencoder_tsne_cohort.png'
plt.savefig(out2, dpi=150)
print(f"Figure saved -> {out2}")
