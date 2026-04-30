"""
Train and evaluate an LSTM autoencoder on processed EmoPairCompete windows.

This script reuses the same processed files as the ConvAE pipeline.

Input processed file examples:
    data/processed/autoencoder/autoencoder_windows.npz
    data/processed/autoencoder/autoencoder_windows_cohort_norm.npz
    data/processed/autoencoder/autoencoder_windows_cohort_individual_norm.npz

Pipeline:
    load processed windows
    → train LSTM autoencoder
    → extract window-level latents
    → average latents per phase
    → K-Means clustering
    → save outputs

Example:
    python advanced/v9_lstm_autoencoder.py \
        --processed-file data/processed/autoencoder/autoencoder_windows_cohort_norm.npz \
        --output-dir advanced/outputs/lstm_ae_32_cohort_norm
"""

from __future__ import annotations

import argparse
import json
import sys
from io import StringIO
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------
# Paths and imports
# ---------------------------------------------------------------------
ADVANCED_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ADVANCED_DIR.parent

sys.path.insert(0, str(ADVANCED_DIR))

from utils.data_processing import META_KEYS  # noqa: E402
from utils.lstm_autoencoder import build_model  # noqa: E402


# ---------------------------------------------------------------------
# Loading processed data
# ---------------------------------------------------------------------
def load_processed_file(
    processed_file: Path,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Load processed autoencoder windows.

    Returns:
        X_scaled:
            shape N x T x C

        X_lstm:
            shape N x T x C, same as X_scaled

        window_meta:
            one row per window
    """
    if not processed_file.exists():
        raise FileNotFoundError(f"Could not find processed file: {processed_file}")

    arrays = np.load(processed_file, allow_pickle=True)

    if "X_scaled" not in arrays:
        raise ValueError(f"{processed_file} does not contain X_scaled.")

    X_scaled = arrays["X_scaled"].astype(np.float32)

    if "window_meta_json" in arrays:
        window_meta_json = str(arrays["window_meta_json"])
        window_meta = pd.read_json(StringIO(window_meta_json), orient="split")
    else:
        metadata_csv = processed_file.parent / "window_metadata.csv"
        if not metadata_csv.exists():
            raise ValueError(
                f"No window_meta_json in {processed_file} and no "
                f"window_metadata.csv found in {processed_file.parent}."
            )
        window_meta = pd.read_csv(metadata_csv)

    return X_scaled, X_scaled, window_meta


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------
def train_autoencoder(
    model: torch.nn.Module,
    loader: DataLoader,
    dataset_size: int,
    epochs: int,
    learning_rate: float,
    device: torch.device,
) -> list[float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    train_losses: list[float] = []

    print(f"\nTraining LSTM autoencoder on {device}...")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0

        for (batch,) in loader:
            batch = batch.to(device)

            reconstruction, _ = model(batch)
            loss = criterion(reconstruction, batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(batch)

        average_loss = epoch_loss / dataset_size
        train_losses.append(average_loss)

        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"  Epoch {epoch:3d}/{epochs}  loss={average_loss:.6f}")

    return train_losses


def extract_latents(
    model: torch.nn.Module,
    dataset: TensorDataset,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    model.eval()
    latents: list[np.ndarray] = []

    with torch.no_grad():
        for (batch,) in DataLoader(dataset, batch_size=batch_size, shuffle=False):
            z = model.encode(batch.to(device))
            latents.append(z.cpu().numpy())

    return np.concatenate(latents, axis=0)


# ---------------------------------------------------------------------
# Aggregation and clustering
# ---------------------------------------------------------------------
def aggregate_latents_per_phase(
    window_latents: np.ndarray,
    window_meta: pd.DataFrame,
    meta_keys: Iterable[str] = META_KEYS,
) -> tuple[np.ndarray, pd.DataFrame]:
    meta_keys = list(meta_keys)

    phase_latents: list[np.ndarray] = []
    phase_meta_rows: list[dict[str, object]] = []

    for keys, group in window_meta.groupby(meta_keys, sort=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        phase_latents.append(window_latents[group.index].mean(axis=0))

        row = dict(zip(meta_keys, keys))

        for col in window_meta.columns:
            if col in meta_keys or col.startswith("Window"):
                continue

            values = group[col].dropna()
            row[col] = values.iloc[0] if len(values) else np.nan

        phase_meta_rows.append(row)

    return np.stack(phase_latents).astype(np.float32), pd.DataFrame(phase_meta_rows)


def run_kmeans_sweep(
    phase_latents: np.ndarray,
    k_min: int,
    k_max: int,
    random_state: int,
) -> tuple[dict[int, float], int, np.ndarray]:
    n_samples = len(phase_latents)

    if n_samples < 3:
        raise ValueError("Need at least 3 phase-level samples for clustering.")

    k_max = min(k_max, n_samples - 1)

    if k_min > k_max:
        raise ValueError(f"Invalid k range: k_min={k_min}, k_max={k_max}")

    silhouette_scores: dict[int, float] = {}

    for k in range(k_min, k_max + 1):
        model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = model.fit_predict(phase_latents)
        silhouette_scores[k] = float(silhouette_score(phase_latents, labels))

    best_k = max(silhouette_scores, key=silhouette_scores.get)

    final_model = KMeans(n_clusters=best_k, random_state=random_state, n_init=10)
    final_labels = final_model.fit_predict(phase_latents)

    return silhouette_scores, best_k, final_labels


# ---------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------
def save_outputs(
    output_dir: Path,
    train_losses: list[float],
    window_latents: np.ndarray,
    phase_latents: np.ndarray,
    phase_meta: pd.DataFrame,
    silhouette_scores: dict[int, float],
    best_k: int,
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "window_latents.npy", window_latents.astype(np.float32))
    np.save(output_dir / "phase_latents.npy", phase_latents.astype(np.float32))

    phase_meta.to_csv(output_dir / "phase_metadata_with_clusters.csv", index=False)

    pd.DataFrame(
        {
            "epoch": range(1, len(train_losses) + 1),
            "loss": train_losses,
        }
    ).to_csv(output_dir / "training_loss.csv", index=False)

    pd.DataFrame(
        {
            "k": list(silhouette_scores.keys()),
            "silhouette": list(silhouette_scores.values()),
        }
    ).to_csv(output_dir / "kmeans_silhouette_scores.csv", index=False)

    metrics = {
        "model": "LSTMAutoencoder",
        "processed_file": str(args.processed_file),
        "output_dir": str(output_dir),
        "latent_dim": int(args.latent_dim),
        "hidden_dim": int(args.hidden_dim),
        "num_layers": int(args.num_layers),
        "dropout": float(args.dropout),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "k_min": int(args.k_min),
        "k_max": int(args.k_max),
        "best_k": int(best_k),
        "best_silhouette": float(silhouette_scores[best_k]),
        "final_training_loss": float(train_losses[-1]),
        "note": "K-Means was fitted on phase-level LSTM autoencoder latents.",
    }

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Outputs saved to: {output_dir}")


def plot_training_and_silhouette(
    train_losses: list[float],
    silhouette_scores: dict[int, float],
    best_k: int,
    figures_dir: Path,
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(range(1, len(train_losses) + 1), train_losses)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].set_title("LSTM Autoencoder Training Loss")

    axes[1].bar(list(silhouette_scores.keys()), list(silhouette_scores.values()))
    axes[1].axvline(best_k, linestyle="--", label=f"Best k={best_k}")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].set_title("K-Means Silhouette Scores")
    axes[1].legend()

    plt.tight_layout()

    out = figures_dir / "lstm_training_and_silhouette.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)

    print(f"Figure saved -> {out}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    default_processed_file = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "autoencoder"
        / "autoencoder_windows_cohort_norm.npz"
    )

    default_output_dir = ADVANCED_DIR / "outputs" / "lstm_ae_32_cohort_norm"

    parser = argparse.ArgumentParser(
        description="Train LSTM autoencoder on processed physiological windows."
    )

    parser.add_argument(
        "--processed-file",
        type=Path,
        default=default_processed_file,
        help="Processed .npz file containing X_scaled and metadata.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help="Experiment output folder.",
    )

    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)

    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=8)
    parser.add_argument("--random-state", type=int, default=42)

    parser.add_argument(
        "--save-model",
        action="store_true",
        help="Save trained LSTM autoencoder weights.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    torch.manual_seed(args.random_state)
    np.random.seed(args.random_state)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_scaled, X_lstm, window_meta = load_processed_file(args.processed_file)

    print(
        f"Loaded windows: {X_lstm.shape[0]} windows x "
        f"{X_lstm.shape[1]} seconds x {X_lstm.shape[2]} signals"
    )

    X_tensor = torch.tensor(X_lstm, dtype=torch.float32)
    dataset = TensorDataset(X_tensor)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = build_model(
        n_channels=X_lstm.shape[2],
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        device=device,
    )

    train_losses = train_autoencoder(
        model=model,
        loader=loader,
        dataset_size=len(dataset),
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=device,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.save_model:
        model_path = args.output_dir / "lstm_autoencoder.pt"
        torch.save(model.state_dict(), model_path)
        print(f"Model weights saved to: {model_path}")

    Z_window = extract_latents(
        model=model,
        dataset=dataset,
        device=device,
    )

    print(f"Window-level latent matrix: {Z_window.shape}")

    Z_phase, phase_meta = aggregate_latents_per_phase(
        window_latents=Z_window,
        window_meta=window_meta,
    )

    print(f"Phase-level latent matrix: {Z_phase.shape}")

    silhouette_scores, best_k, labels = run_kmeans_sweep(
        phase_latents=Z_phase,
        k_min=args.k_min,
        k_max=args.k_max,
        random_state=args.random_state,
    )

    phase_meta["Cluster"] = labels

    print(f"Silhouette scores: { {k: f'{v:.3f}' for k, v in silhouette_scores.items()} }")
    print(f"Best k: {best_k}")

    save_outputs(
        output_dir=args.output_dir,
        train_losses=train_losses,
        window_latents=Z_window,
        phase_latents=Z_phase,
        phase_meta=phase_meta,
        silhouette_scores=silhouette_scores,
        best_k=best_k,
        args=args,
    )

    plot_training_and_silhouette(
        train_losses=train_losses,
        silhouette_scores=silhouette_scores,
        best_k=best_k,
        figures_dir=args.output_dir / "figures",
    )

    print("\nDone.")
    print(f"Output folder: {args.output_dir}")


if __name__ == "__main__":
    main()