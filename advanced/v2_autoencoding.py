"""
Train and evaluate the Conv1D autoencoder using normalized raw time-series data.

This is the normalized-data version of v1_autoencoding.py.

It creates or reuses processed files such as:

    data/processed/autoencoder/autoencoder_windows_individual_norm.npz
    data/processed/autoencoder/autoencoder_windows_cohort_norm.npz
    data/processed/autoencoder/autoencoder_windows_individual_round_norm.npz

Experiment outputs are saved in:

    advanced/outputs/conv_ae_<latent_dim>_<normalization>_norm/

Example:

    python advanced/v2_autoencoding.py --normalization individual
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------
# Paths and local imports
# ---------------------------------------------------------------------
ADVANCED_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ADVANCED_DIR.parent

sys.path.insert(0, str(ADVANCED_DIR))

from utils.conv_autoencoder import build_model  # noqa: E402
from utils.data_processing import SIGNALS  # noqa: E402
from utils.data_processing_norm import build_normalized_autoencoder_file  # noqa: E402

from v1_autoencoding import (  # noqa: E402
    load_processed_file,
    train_autoencoder,
    extract_latents,
    aggregate_latents_per_phase,
    run_kmeans_sweep,
    make_2d_projection,
    save_experiment_outputs,
    plot_training_and_silhouette,
    plot_2d_embedding,
)


def default_processed_file(normalization: str) -> Path:
    return (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "autoencoder"
        / f"autoencoder_windows_{normalization}_norm.npz"
    )


def default_output_dir(latent_dim: int, normalization: str) -> Path:
    return (
        ADVANCED_DIR
        / "outputs"
        / f"conv_ae_{latent_dim}_{normalization}_norm"
    )


def parse_args() -> argparse.Namespace:
    default_dataset_dir = PROJECT_ROOT / "data" / "raw" / "data" / "dataset"

    parser = argparse.ArgumentParser(
        description="Train Conv1D autoencoder on normalized HR/EDA/TEMP windows."
    )

    parser.add_argument("--dataset-dir", type=Path, default=default_dataset_dir)

    parser.add_argument(
        "--normalization",
        choices=["individual", "cohort", "individual_round", "cohort_round"],
        default="individual",
        help="Baseline normalization strategy.",
    )

    parser.add_argument(
        "--processed-file",
        type=Path,
        default=None,
        help=(
            "Optional normalized processed .npz file. "
            "If omitted, a default file is chosen based on normalization."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Experiment-specific output directory. "
            "If omitted, a default folder is chosen based on latent_dim and normalization."
        ),
    )

    parser.add_argument("--window-size", type=int, default=60)
    parser.add_argument("--step-size", type=int, default=30)
    parser.add_argument("--resample-rule", type=str, default="1s")
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
        help="Rebuild the normalized processed data even if the file already exists.",
    )

    parser.add_argument(
        "--save-model",
        action="store_true",
        help="Save trained model weights to output-dir/model.pt.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.processed_file is None:
        args.processed_file = default_processed_file(args.normalization)

    if args.output_dir is None:
        args.output_dir = default_output_dir(args.latent_dim, args.normalization)

    torch.manual_seed(args.random_state)
    np.random.seed(args.random_state)

    # -----------------------------------------------------------------
    # 1. Create or reuse normalized processed data
    # -----------------------------------------------------------------
    if args.force_reprocess or not args.processed_file.exists():
        print("Normalized processed data not found or force_reprocess=True.")
        print(f"Creating normalized processed data: {args.processed_file}")

        build_normalized_autoencoder_file(
            dataset_dir=args.dataset_dir,
            processed_file=args.processed_file,
            signals=SIGNALS,
            window_size=args.window_size,
            step_size=args.step_size,
            normalization=args.normalization,
            resample_rule=args.resample_rule,
        )
    else:
        print(f"Using existing normalized processed data: {args.processed_file}")

    # -----------------------------------------------------------------
    # 2. Load normalized processed data
    # -----------------------------------------------------------------
    X_raw, X_scaled, X_conv1d, window_meta = load_processed_file(
        args.processed_file
    )

    print(
        f"Windows: {X_scaled.shape} - "
        f"{X_scaled.shape[0]} windows x "
        f"{X_scaled.shape[1]} seconds x "
        f"{X_scaled.shape[2]} signals"
    )

    # -----------------------------------------------------------------
    # 3. Train autoencoder
    # -----------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    # -----------------------------------------------------------------
    # 4. Extract and aggregate latents
    # -----------------------------------------------------------------
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

    # -----------------------------------------------------------------
    # 5. K-Means clustering
    # -----------------------------------------------------------------
    silhouette_scores, best_k, labels = run_kmeans_sweep(
        phase_latents=Z_phase,
        k_min=args.k_min,
        k_max=args.k_max,
        random_state=args.random_state,
    )

    phase_meta["Cluster"] = labels

    print(f"Silhouette scores: { {k: f'{v:.3f}' for k, v in silhouette_scores.items()} }")
    print(f"Best k: {best_k}")

    # -----------------------------------------------------------------
    # 6. Save useful experiment outputs
    # -----------------------------------------------------------------
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

    print("\nDone.")
    print(f"Processed file: {args.processed_file}")
    print(f"Output folder:  {args.output_dir}")


if __name__ == "__main__":
    main()