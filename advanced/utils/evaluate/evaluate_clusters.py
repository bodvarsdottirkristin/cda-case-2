"""
Evaluate clustering results from an autoencoder or any other model.

This script is model-agnostic.

It expects an experiment output folder containing:
    phase_metadata_with_clusters.csv

It saves evaluation results into:
    <output-dir>/eval/

Saved files:
    cluster_alignment_metrics.csv
    questionnaire_cluster_profiles.csv
    summary.txt

Example from project root:
    python advanced/utils/evaluate/evaluate_clusters.py \
        --output-dir advanced/outputs/conv_ae_32
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def infer_advanced_dir() -> Path:
    here = Path(__file__).resolve()

    for parent in here.parents:
        if parent.name == "advanced":
            return parent

    cwd = Path.cwd()
    if (cwd / "advanced").exists():
        return cwd / "advanced"

    return here.parents[1]


ADVANCED_DIR = infer_advanced_dir()


# Variables used to interpret clusters.
# Phase is the main experimental-state variable.
# Puzzler and parent are role variables.
# Cohort, Round, and Individual are possible confounds.
ALIGNMENT_COLUMNS = [
    "Phase",
    "Puzzler",
    "Cohort",
    "Round",
    "Individual",
]


def safe_alignment_metrics(df: pd.DataFrame, label_col: str) -> dict[str, object]:
    """Compute ARI and NMI between one metadata variable and cluster labels."""
    usable = df[[label_col, "Cluster"]].dropna()

    if (
        usable.empty
        or usable[label_col].nunique() < 2
        or usable["Cluster"].nunique() < 2
    ):
        return {
            "variable": label_col,
            "ari": np.nan,
            "nmi": np.nan,
            "n_unique_values": usable[label_col].nunique() if not usable.empty else 0,
            "n_samples_used": len(usable),
        }

    return {
        "variable": label_col,
        "ari": adjusted_rand_score(usable[label_col], usable["Cluster"]),
        "nmi": normalized_mutual_info_score(usable[label_col], usable["Cluster"]),
        "n_unique_values": usable[label_col].nunique(),
        "n_samples_used": len(usable),
    }


def get_questionnaire_columns(df: pd.DataFrame) -> list[str]:
    """
    Return numeric columns that may represent questionnaire/emotion metadata.

    Role/metadata columns such as Puzzler and parent are excluded because
    they are useful for role-alignment checks, but they are not emotion scores.
    """
    excluded_cols = {
        "Cohort",
        "Individual",
        "Round",
        "Phase",
        "Cluster",
        "WindowStartIndex",
        "WindowEndIndex",
        "WindowStartTime",
        "WindowEndTime",
        "Puzzler",
        "parent",
    }

    numeric_cols = df.select_dtypes(include="number").columns
    return [col for col in numeric_cols if col not in excluded_cols]


def compute_alignment_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute ARI/NMI against available metadata and role labels."""
    label_cols = [col for col in ALIGNMENT_COLUMNS if col in df.columns]
    rows = [safe_alignment_metrics(df, col) for col in label_cols]
    return pd.DataFrame(rows)


def compute_questionnaire_profiles(df: pd.DataFrame) -> pd.DataFrame | None:
    """Compute questionnaire/numeric profiles by cluster."""
    questionnaire_cols = get_questionnaire_columns(df)

    if not questionnaire_cols:
        return None

    return df.groupby("Cluster")[questionnaire_cols].agg(["mean", "std", "count"])


def make_crosstab_text(df: pd.DataFrame, col: str) -> str:
    """Create a readable crosstab string for summary.txt."""
    if col not in df.columns:
        return f"{col} not found.\n"

    table = pd.crosstab(df[col], df["Cluster"])
    return table.to_string()


def load_metrics_json(output_dir: Path) -> dict:
    """Load model-run metrics from <output-dir>/metrics.json if available."""
    metrics_path = output_dir / "metrics.json"

    if not metrics_path.exists():
        return {}

    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def update_metrics_json(output_dir: Path, alignment_metrics: pd.DataFrame) -> None:
    """
    Optionally add evaluation metrics into <output-dir>/metrics.json.

    metrics.json remains in the main experiment folder because it describes
    the whole model run, while detailed evaluation files live in eval/.
    """
    metrics_path = output_dir / "metrics.json"
    metrics = load_metrics_json(output_dir)

    evaluation = {}

    for _, row in alignment_metrics.iterrows():
        variable = row["variable"]
        evaluation[f"{variable}_ari"] = (
            None if pd.isna(row["ari"]) else float(row["ari"])
        )
        evaluation[f"{variable}_nmi"] = (
            None if pd.isna(row["nmi"]) else float(row["nmi"])
        )

    metrics["evaluation"] = evaluation

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def write_summary(
    eval_dir: Path,
    df: pd.DataFrame,
    alignment_metrics: pd.DataFrame,
    questionnaire_profiles: pd.DataFrame | None,
    run_metrics: dict,
) -> None:
    """Write one readable summary file containing key results and crosstabs."""
    summary_path = eval_dir / "summary.txt"

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Cluster evaluation summary\n")
        f.write("=" * 32 + "\n\n")

        f.write("Experiment settings\n")
        f.write("-" * 19 + "\n")

        if run_metrics:
            useful_keys = [
                "model",
                "latent_dim",
                "n_components",
                "window_size",
                "step_size",
                "epochs",
                "best_k",
                "best_silhouette",
                "final_training_loss",
                "embedding_method",
                "clustering_method",
                "covariance_type",
                "linkage",
                "selection_metric",
            ]

            for key in useful_keys:
                if key in run_metrics:
                    f.write(f"{key}: {run_metrics[key]}\n")
        else:
            f.write("metrics.json not found.\n")

        f.write("\nDataset summary\n")
        f.write("-" * 15 + "\n")
        f.write(f"Phase-level samples: {len(df)}\n")
        f.write(f"Clusters: {df['Cluster'].nunique()}\n")

        for col in ALIGNMENT_COLUMNS:
            if col in df.columns:
                f.write(f"{col} values: {df[col].nunique()}\n")

        f.write("\nAlignment metrics\n")
        f.write("-" * 17 + "\n")
        f.write(alignment_metrics.to_string(index=False))
        f.write("\n\n")

        f.write("Interpretation guide\n")
        f.write("-" * 20 + "\n")
        f.write(
            "ARI/NMI near 0 means weak correspondence with the metadata variable.\n"
            "Higher Phase alignment supports phase-related/emotional-state structure.\n"
            "Higher Puzzler or parent alignment suggests role-related physiological structure.\n"
            "Higher Cohort, Individual, or Round alignment suggests baseline/session confounds.\n\n"
        )

        for col in ALIGNMENT_COLUMNS:
            if col in df.columns:
                f.write(f"{col} x Cluster\n")
                f.write("-" * (len(col) + 10) + "\n")
                f.write(make_crosstab_text(df, col))
                f.write("\n\n")

        if questionnaire_profiles is not None:
            f.write("Questionnaire profiles\n")
            f.write("-" * 22 + "\n")
            f.write(
                "Saved to questionnaire_cluster_profiles.csv.\n"
                "Use this to inspect whether clusters differ in frustration, "
                "nervousness, upset, alertness, difficulty, etc.\n"
                "Note: role variables such as Puzzler and parent are excluded "
                "from questionnaire profiles and evaluated separately above.\n"
            )
        else:
            f.write("No numeric questionnaire columns found.\n")

    print(f"Summary saved to: {summary_path}")


def parse_args() -> argparse.Namespace:
    default_output_dir = ADVANCED_DIR / "outputs" / "conv_ae_32"

    parser = argparse.ArgumentParser(
        description="Evaluate cluster assignments from a model output folder."
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help="Output directory containing phase_metadata_with_clusters.csv.",
    )

    parser.add_argument(
        "--eval-dir-name",
        type=str,
        default="eval",
        help="Name of the subfolder where evaluation files are saved.",
    )

    parser.add_argument(
        "--update-metrics-json",
        action="store_true",
        help="Also add evaluation metrics into <output-dir>/metrics.json.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = args.output_dir
    eval_dir = output_dir / args.eval_dir_name
    eval_dir.mkdir(parents=True, exist_ok=True)

    phase_meta_path = output_dir / "phase_metadata_with_clusters.csv"

    if not phase_meta_path.exists():
        raise FileNotFoundError(
            f"Could not find {phase_meta_path}. "
            "Run the model script first."
        )

    df = pd.read_csv(phase_meta_path)

    if "Cluster" not in df.columns:
        raise ValueError(
            "phase_metadata_with_clusters.csv must contain a 'Cluster' column."
        )

    print(f"Loaded {len(df)} phase-level rows from: {phase_meta_path}")
    print(f"Saving evaluation files to: {eval_dir}")

    alignment_metrics = compute_alignment_metrics(df)
    alignment_metrics.to_csv(
        eval_dir / "cluster_alignment_metrics.csv",
        index=False,
    )

    questionnaire_profiles = compute_questionnaire_profiles(df)

    if questionnaire_profiles is not None:
        questionnaire_profiles.to_csv(
            eval_dir / "questionnaire_cluster_profiles.csv"
        )

    run_metrics = load_metrics_json(output_dir)

    write_summary(
        eval_dir=eval_dir,
        df=df,
        alignment_metrics=alignment_metrics,
        questionnaire_profiles=questionnaire_profiles,
        run_metrics=run_metrics,
    )

    if args.update_metrics_json:
        update_metrics_json(output_dir, alignment_metrics)
        print(f"metrics.json updated in: {output_dir}")

    print("\nEvaluation complete.")
    print(f"Saved: {eval_dir / 'cluster_alignment_metrics.csv'}")

    if questionnaire_profiles is not None:
        print(f"Saved: {eval_dir / 'questionnaire_cluster_profiles.csv'}")

    print(f"Saved: {eval_dir / 'summary.txt'}")


if __name__ == "__main__":
    main()