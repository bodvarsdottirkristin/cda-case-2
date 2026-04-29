"""
Train and evaluate the Conv1D autoencoder on raw EmoPairCompete signals.

Reusable processed data is saved in:
    data/processed/autoencoder/autoencoder_windows.npz

Experiment-specific outputs are saved in:
    advanced/outputs/conv_ae_<latent_dim>/

Expected project layout:

    project_root/
    ├── advanced/
    │   ├── v1_autoencoding.py
    │   ├── outputs/
    │   └── utils/
    │       ├── conv_autoencoder.py
    │       └── data_processing.py
    └── data/
        ├── raw/
        │   └── data/
        │       └── dataset/
        └── processed/
            └── autoencoder/
                └── autoencoder_windows.npz
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
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------
# Paths and local imports
# ---------------------------------------------------------------------
ADVANCED_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ADVANCED_DIR.parent

sys.path.insert(0, str(ADVANCED_DIR))

from utils.conv_autoencoder import build_model  # noqa: E402
from utils.data_processing import (  # noqa: E402
    META_KEYS,
    SIGNALS,
    build_autoencoder_input,
    to_conv1d_format,
)


# ---------------------------------------------------------------------
# Processed-data cache: one reusable .npz file
# ---------------------------------------------------------------------
def processed_file_is_compatible(
    processed_file: Path,
    signals: Iterable[str],
    window_size: int,
    step_size: int,
) -> bool:
    """
    Check whether the reusable processed file exists and matches
    the current preprocessing settings.
    """
    if not processed_file.exists():
        return False

    try:
        data = np.load(processed_file, allow_pickle=True)

        cached_signals = list(data["signals"])
        cached_window_size = int(data["window_size"])
        cached_step_size = int(data["step_size"])

    except Exception:
        return False

    return (
        cached_signals == list(signals)
        and cached_window_size == int(window_size)
        and cached_step_size == int(step_size)
    )


def save_processed_file(
    processed_file: Path,
    X_raw: np.ndarray,
    X_scaled: np.ndarray,
    X_conv1d: np.ndarray,
    window_meta: pd.DataFrame,
    signals: Iterable[str],
    window_size: int,
    step_size: int,
) -> None:
    """
    Save one reusable processed-data file.

    This file can be reused by other autoencoder models.
    """
    processed_file.parent.mkdir(parents=True, exist_ok=True)

    window_meta_json = window_meta.to_json(orient="split")

    np.savez_compressed(
        processed_file,
        X_raw=X_raw.astype(np.float32),
        X_scaled=X_scaled.astype(np.float32),
        X_conv1d=X_conv1d.astype(np.float32),
        window_meta_json=np.array(window_meta_json),
        signals=np.array(list(signals)),
        window_size=np.array(window_size),
        step_size=np.array(step_size),
    )

    print(f"Processed data saved to: {processed_file}")


def load_processed_file(
    processed_file: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Load reusable processed data.

    Returns:
        X_raw:       shape (n_windows, window_size, n_signals)
        X_scaled:    shape (n_windows, window_size, n_signals)
        X_conv1d:    shape (n_windows, n_signals, window_size)
        window_meta: one row per window
    """
    data = np.load(processed_file, allow_pickle=True)

    X_raw = data["X_raw"]
    X_scaled = data["X_scaled"]

    if "X_conv1d" in data:
        X_conv1d = data["X_conv1d"]
    else:
        X_conv1d = to_conv1d_format(X_scaled)

    window_meta_json = str(data["window_meta_json"])
    window_meta = pd.read_json(StringIO(window_meta_json), orient="split")

    return X_raw, X_scaled, X_conv1d, window_meta


def get_or_create_processed_data(
    dataset_dir: Path,
    processed_file: Path,
    signals: Iterable[str],
    window_size: int,
    step_size: int,
    force_reprocess: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Load processed data if available.

    Otherwise, process raw data and save one reusable .npz file.
    """
    if not force_reprocess and processed_file_is_compatible(
        processed_file=processed_file,
        signals=signals,
        window_size=window_size,
        step_size=step_size,
    ):
        print(f"Loading processed data from: {processed_file}")
        return load_processed_file(processed_file)

    print("Processed data not found or incompatible.")
    print(f"Processing raw data from: {dataset_dir}")

    X_raw, X_scaled, window_meta, _, _ = build_autoencoder_input(
        dataset_dir=dataset_dir,
        signals=signals,
        window_size=window_size,
        step_size=step_size,
    )

    X_conv1d = to_conv1d_format(X_scaled)

    save_processed_file(
        processed_file=processed_file,
        X_raw=X_raw,
        X_scaled=X_scaled,
        X_conv1d=X_conv1d,
        window_meta=window_meta,
        signals=signals,
        window_size=window_size,
        step_size=step_size,
    )

    return X_raw, X_scaled, X_conv1d, window_meta


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
    """Train the autoencoder using reconstruction MSE."""
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    train_losses: list[float] = []

    print(f"\nTraining Conv1D autoencoder on {device}...")

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
            print(f"  Epoch {epoch:3d}/{epochs}  loss={average_loss:.5f}")

    return train_losses


def extract_latents(
    model: torch.nn.Module,
    dataset: TensorDataset,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    """Encode all windows into latent vectors."""
    model.eval()
    latents: list[np.ndarray] = []

    with torch.no_grad():
        for (batch,) in DataLoader(dataset, batch_size=batch_size, shuffle=False):
            z = model.encode(batch.to(device))
            latents.append(z.cpu().numpy())

    return np.concatenate(latents, axis=0)


def aggregate_latents_per_phase(
    window_latents: np.ndarray,
    window_meta: pd.DataFrame,
    meta_keys: Iterable[str] = META_KEYS,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Average window-level latent vectors per Cohort/Individual/Round/Phase.

    The final clustering is performed at phase level.
    """
    meta_keys = list(meta_keys)

    phase_latents: list[np.ndarray] = []
    phase_meta_rows: list[dict[str, object]] = []

    for keys, group in window_meta.groupby(meta_keys, sort=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        phase_latents.append(window_latents[group.index].mean(axis=0))

        row = dict(zip(meta_keys, keys))

        # Preserve useful extra metadata, for example questionnaire columns.
        for col in window_meta.columns:
            if col in meta_keys or col.startswith("Window"):
                continue

            values = group[col].dropna()
            row[col] = values.iloc[0] if len(values) else np.nan

        phase_meta_rows.append(row)

    return np.stack(phase_latents), pd.DataFrame(phase_meta_rows)


# ---------------------------------------------------------------------
# Clustering and optional 2D visualization
# ---------------------------------------------------------------------
def run_kmeans_sweep(
    phase_latents: np.ndarray,
    k_min: int,
    k_max: int,
    random_state: int,
) -> tuple[dict[int, float], int, np.ndarray]:
    """
    Run K-Means for k_min...k_max and select the best k by silhouette score.

    K-Means is fitted on the original latent space, not on PCA/t-SNE.
    """
    n_samples = len(phase_latents)

    if n_samples < 3:
        raise ValueError("Need at least 3 phase-level samples for clustering.")

    k_max = min(k_max, n_samples - 1)

    if k_min > k_max:
        raise ValueError(
            f"Invalid k range after sample-size check: k_min={k_min}, k_max={k_max}"
        )

    silhouette_scores: dict[int, float] = {}

    for k in range(k_min, k_max + 1):
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(phase_latents)
        silhouette_scores[k] = float(silhouette_score(phase_latents, labels))

    best_k = max(silhouette_scores, key=silhouette_scores.get)

    final_kmeans = KMeans(n_clusters=best_k, random_state=random_state, n_init=10)
    final_labels = final_kmeans.fit_predict(phase_latents)

    return silhouette_scores, best_k, final_labels


def make_2d_projection(
    phase_latents: np.ndarray,
    method: str,
    random_state: int,
) -> np.ndarray | None:
    """
    Optional 2D projection for visualization only.

    method:
        none -> no projection
        pca  -> linear projection
        tsne -> non-linear diagnostic visualization
    """
    if method == "none":
        return None

    if method == "pca":
        return PCA(n_components=2, random_state=random_state).fit_transform(
            phase_latents
        )

    if method == "tsne":
        perplexity = min(30, max(2, (len(phase_latents) - 1) // 3))

        try:
            return TSNE(
                n_components=2,
                perplexity=perplexity,
                random_state=random_state,
                max_iter=1000,
                init="pca",
                learning_rate="auto",
            ).fit_transform(phase_latents)
        except TypeError:
            # Older scikit-learn versions use n_iter instead of max_iter.
            return TSNE(
                n_components=2,
                perplexity=perplexity,
                random_state=random_state,
                n_iter=1000,
                init="pca",
                learning_rate="auto",
            ).fit_transform(phase_latents)

    raise ValueError(f"Unknown embedding method: {method}")


# ---------------------------------------------------------------------
# Saving useful experiment outputs only
# ---------------------------------------------------------------------
def save_experiment_outputs(
    output_dir: Path,
    train_losses: list[float],
    phase_latents: np.ndarray,
    phase_meta: pd.DataFrame,
    silhouette_scores: dict[int, float],
    best_k: int,
    args: argparse.Namespace,
) -> None:
    """
    Save only useful results for the project.

    Saved:
        - phase_latents.npy
        - phase_metadata_with_clusters.csv
        - training_loss.csv
        - kmeans_silhouette_scores.csv
        - metrics.json

    Not saved by default:
        - window_latents.npy
        - large manifest files
    """
    output_dir.mkdir(parents=True, exist_ok=True)

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
        "model": "Conv1DAutoencoder",
        "signals": list(SIGNALS),
        "latent_dim": int(args.latent_dim),
        "window_size": int(args.window_size),
        "step_size": int(args.step_size),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "k_min": int(args.k_min),
        "k_max": int(args.k_max),
        "best_k": int(best_k),
        "best_silhouette": float(silhouette_scores[best_k]),
        "final_training_loss": float(train_losses[-1]),
        "embedding_method": args.embedding_method,
        "processed_file": str(args.processed_file),
        "output_dir": str(args.output_dir),
        "note": "K-Means was fitted on phase-level latent vectors, not on PCA/t-SNE.",
    }

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Experiment outputs saved to: {output_dir}")


def plot_training_and_silhouette(
    train_losses: list[float],
    silhouette_scores: dict[int, float],
    best_k: int,
    figures_dir: Path,
) -> None:
    """Save the main training/clustering diagnostic figure."""
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(range(1, len(train_losses) + 1), train_losses)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].set_title("Autoencoder Training Loss")

    axes[1].bar(list(silhouette_scores.keys()), list(silhouette_scores.values()))
    axes[1].axvline(best_k, linestyle="--", label=f"Best k={best_k}")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].set_title("K-Means Silhouette Scores")
    axes[1].legend()

    plt.tight_layout()
    out = figures_dir / "training_and_silhouette.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)

    print(f"Figure saved -> {out}")


def plot_2d_embedding(
    Z_2d: np.ndarray,
    phase_meta: pd.DataFrame,
    method: str,
    best_k: int,
    figures_dir: Path,
) -> None:
    """
    Save optional 2D diagnostic plots.

    These are visual diagnostics only.
    Clustering is not fitted on these 2D projections.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)

    method_label = method.upper()

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        Z_2d[:, 0],
        Z_2d[:, 1],
        c=phase_meta["Cluster"],
        cmap="tab10",
        s=40,
        alpha=0.8,
    )
    plt.colorbar(scatter, ax=ax, label="Cluster")
    ax.set_title(f"{method_label} of Phase Latents, colored by cluster (k={best_k})")
    ax.set_xlabel(f"{method_label} 1")
    ax.set_ylabel(f"{method_label} 2")
    plt.tight_layout()

    out = figures_dir / f"{method}_clusters.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Figure saved -> {out}")

    if "Cohort" in phase_meta.columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        cohorts = sorted(phase_meta["Cohort"].dropna().unique())
        cmap = plt.get_cmap("tab10")

        for i, cohort in enumerate(cohorts):
            mask = phase_meta["Cohort"] == cohort
            ax.scatter(
                Z_2d[mask, 0],
                Z_2d[mask, 1],
                label=cohort,
                color=cmap(i % 10),
                s=40,
                alpha=0.8,
            )

        ax.legend(title="Cohort")
        ax.set_title(f"{method_label} of Phase Latents, colored by cohort")
        ax.set_xlabel(f"{method_label} 1")
        ax.set_ylabel(f"{method_label} 2")
        plt.tight_layout()

        out = figures_dir / f"{method}_cohort.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Figure saved -> {out}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    default_dataset_dir = PROJECT_ROOT / "data" / "raw" / "data" / "dataset"
    default_processed_file = (
        PROJECT_ROOT / "data" / "processed" / "autoencoder" / "autoencoder_windows.npz"
    )

    parser = argparse.ArgumentParser(
        description="Train Conv1D autoencoder on raw HR/EDA/TEMP windows."
    )

    parser.add_argument("--dataset-dir", type=Path, default=default_dataset_dir)

    parser.add_argument(
        "--processed-file",
        type=Path,
        default=default_processed_file,
        help="Reusable processed autoencoder input file.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Experiment-specific output directory. "
            "Default: advanced/outputs/conv_ae_<latent_dim>."
        ),
    )

    parser.add_argument("--window-size", type=int, default=60)
    parser.add_argument("--step-size", type=int, default=30)
    parser.add_argument("--latent-dim", type=int, default=32)

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)

    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=8)
    parser.add_argument("--random-state", type=int, default=42)

    parser.add_argument(
        "--embedding-method",
        choices=["none", "pca", "tsne"],
        default="none",
        help=(
            "Optional 2D projection for visualization only. "
            "K-Means is always run on the original latent space."
        ),
    )

    parser.add_argument(
        "--force-reprocess",
        action="store_true",
        help="Ignore existing processed data and rebuild from raw files.",
    )

    parser.add_argument(
        "--save-model",
        action="store_true",
        help="Save trained model weights to output-dir/model.pt.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.output_dir is None:
        args.output_dir = ADVANCED_DIR / "outputs" / f"conv_ae_{args.latent_dim}"

    torch.manual_seed(args.random_state)
    np.random.seed(args.random_state)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_raw, X_scaled, X_conv1d, window_meta = get_or_create_processed_data(
        dataset_dir=args.dataset_dir,
        processed_file=args.processed_file,
        signals=SIGNALS,
        window_size=args.window_size,
        step_size=args.step_size,
        force_reprocess=args.force_reprocess,
    )

    print(
        f"Windows: {X_scaled.shape} - "
        f"{X_scaled.shape[0]} windows x "
        f"{X_scaled.shape[1]} seconds x "
        f"{X_scaled.shape[2]} signals"
    )

    X_tensor = torch.tensor(X_conv1d, dtype=torch.float32)
    dataset = TensorDataset(X_tensor)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = build_model(
        n_channels=X_conv1d.shape[1],
        latent_dim=args.latent_dim,
        window=args.window_size,
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
        model_path = args.output_dir / "model.pt"
        torch.save(model.state_dict(), model_path)
        print(f"Model weights saved to: {model_path}")

    Z_window = extract_latents(
        model=model,
        dataset=dataset,
        device=device,
    )
    print(f"\nWindow-level latent matrix: {Z_window.shape}")

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

    save_experiment_outputs(
        output_dir=args.output_dir,
        train_losses=train_losses,
        phase_latents=Z_phase,
        phase_meta=phase_meta,
        silhouette_scores=silhouette_scores,
        best_k=best_k,
        args=args,
    )

    figures_dir = args.output_dir / "figures"

    plot_training_and_silhouette(
        train_losses=train_losses,
        silhouette_scores=silhouette_scores,
        best_k=best_k,
        figures_dir=figures_dir,
    )

    if args.embedding_method != "none":
        Z_2d = make_2d_projection(
            phase_latents=Z_phase,
            method=args.embedding_method,
            random_state=args.random_state,
        )

        plot_2d_embedding(
            Z_2d=Z_2d,
            phase_meta=phase_meta,
            method=args.embedding_method,
            best_k=best_k,
            figures_dir=figures_dir,
        )
    else:
        print(
            "No 2D projection generated. "
            "K-Means was run on the original phase-level latent space."
        )


if __name__ == "__main__":
    main()