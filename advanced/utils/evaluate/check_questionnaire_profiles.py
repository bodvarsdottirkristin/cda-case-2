"""
Check whether clusters differ in questionnaire / emotion variables.

This script reads:
    <output-dir>/phase_metadata_with_clusters.csv

and saves:
    <output-dir>/eval/questionnaire_cluster_profiles.csv
    <output-dir>/eval/questionnaire_cluster_differences.csv
    <output-dir>/eval/questionnaire_summary.txt

Example:
    python advanced/utils/evaluate/check_questionnaire_profiles.py \
        --output-dir advanced/outputs/conv_ae_32_cohort_norm
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


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


def get_questionnaire_columns(df: pd.DataFrame) -> list[str]:
    """
    Detect numeric columns that may correspond to questionnaire/emotion values.

    We exclude metadata, clustering labels, and window index/time columns.
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
        "puzzler",
        "parent",
        "Parent",
    }

    numeric_cols = df.select_dtypes(include="number").columns

    return [col for col in numeric_cols if col not in excluded_cols]


def compute_cluster_profiles(
    df: pd.DataFrame,
    questionnaire_cols: list[str],
) -> pd.DataFrame:
    """
    Compute mean, std, and count of each questionnaire variable per cluster.
    """
    return df.groupby("Cluster")[questionnaire_cols].agg(["mean", "std", "count"])


def compute_cluster_differences(
    df: pd.DataFrame,
    questionnaire_cols: list[str],
) -> pd.DataFrame:
    """
    Compute a compact interpretation table.

    For each questionnaire variable, we compute:
        - cluster with highest mean
        - cluster with lowest mean
        - difference between highest and lowest cluster mean
        - global standard deviation
        - standardized difference = difference / global std

    The standardized difference is a rough effect-size-like quantity.
    Larger absolute values suggest stronger cluster differences.
    """
    rows = []

    cluster_means = df.groupby("Cluster")[questionnaire_cols].mean(numeric_only=True)

    for col in questionnaire_cols:
        means = cluster_means[col].dropna()

        if means.empty:
            continue

        max_cluster = means.idxmax()
        min_cluster = means.idxmin()

        max_mean = means.loc[max_cluster]
        min_mean = means.loc[min_cluster]
        mean_difference = max_mean - min_mean

        global_std = df[col].std(skipna=True)

        if pd.isna(global_std) or global_std == 0:
            standardized_difference = np.nan
        else:
            standardized_difference = mean_difference / global_std

        rows.append(
            {
                "variable": col,
                "cluster_with_highest_mean": max_cluster,
                "highest_cluster_mean": max_mean,
                "cluster_with_lowest_mean": min_cluster,
                "lowest_cluster_mean": min_mean,
                "mean_difference": mean_difference,
                "global_std": global_std,
                "standardized_difference": standardized_difference,
            }
        )

    out = pd.DataFrame(rows)

    if not out.empty:
        out = out.sort_values(
            "standardized_difference",
            key=lambda s: s.abs(),
            ascending=False,
        )

    return out


def write_summary(
    eval_dir: Path,
    df: pd.DataFrame,
    questionnaire_cols: list[str],
    differences: pd.DataFrame,
) -> None:
    """
    Write a readable summary for the report.
    """
    summary_path = eval_dir / "questionnaire_summary.txt"

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Questionnaire / emotion cluster check\n")
        f.write("=" * 38 + "\n\n")

        f.write(f"Number of phase-level samples: {len(df)}\n")
        f.write(f"Number of clusters: {df['Cluster'].nunique()}\n")
        f.write(f"Questionnaire-like numeric variables found: {len(questionnaire_cols)}\n\n")

        if not questionnaire_cols:
            f.write("No numeric questionnaire columns were found.\n")
            return

        f.write("Variables checked\n")
        f.write("-" * 17 + "\n")
        for col in questionnaire_cols:
            f.write(f"- {col}\n")

        f.write("\nLargest cluster differences\n")
        f.write("-" * 28 + "\n")

        if differences.empty:
            f.write("No usable differences could be computed.\n")
            return

        top = differences.head(10)

        f.write(
            top[
                [
                    "variable",
                    "cluster_with_highest_mean",
                    "highest_cluster_mean",
                    "cluster_with_lowest_mean",
                    "lowest_cluster_mean",
                    "mean_difference",
                    "standardized_difference",
                ]
            ].to_string(index=False)
        )

        f.write("\n\nInterpretation guide\n")
        f.write("-" * 20 + "\n")
        f.write(
            "Look for variables with large mean_difference and standardized_difference.\n"
            "If frustration, nervousness, upset, hostile, afraid, or difficulty are clearly higher in one cluster,\n"
            "that cluster may have some emotional interpretation.\n"
            "If all differences are small, the clusters are probably not emotionally meaningful.\n"
        )

    print(f"Summary saved to: {summary_path}")


def parse_args() -> argparse.Namespace:
    default_output_dir = ADVANCED_DIR / "outputs" / "conv_ae_32"

    parser = argparse.ArgumentParser(
        description="Check questionnaire/emotion differences between clusters."
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help="Experiment output directory containing phase_metadata_with_clusters.csv.",
    )

    parser.add_argument(
        "--eval-dir-name",
        type=str,
        default="eval",
        help="Subfolder where evaluation files are saved.",
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
            f"Could not find {phase_meta_path}. Run the model script first."
        )

    df = pd.read_csv(phase_meta_path)

    if "Cluster" not in df.columns:
        raise ValueError("phase_metadata_with_clusters.csv must contain a Cluster column.")

    questionnaire_cols = get_questionnaire_columns(df)

    print(f"Loaded {len(df)} phase-level rows from: {phase_meta_path}")
    print(f"Found {len(questionnaire_cols)} questionnaire-like numeric columns.")

    if not questionnaire_cols:
        write_summary(
            eval_dir=eval_dir,
            df=df,
            questionnaire_cols=questionnaire_cols,
            differences=pd.DataFrame(),
        )
        return

    profiles = compute_cluster_profiles(df, questionnaire_cols)
    profiles_path = eval_dir / "questionnaire_cluster_profiles.csv"
    profiles.to_csv(profiles_path)
    print(f"Saved: {profiles_path}")

    differences = compute_cluster_differences(df, questionnaire_cols)
    differences_path = eval_dir / "questionnaire_cluster_differences.csv"
    differences.to_csv(differences_path, index=False)
    print(f"Saved: {differences_path}")

    write_summary(
        eval_dir=eval_dir,
        df=df,
        questionnaire_cols=questionnaire_cols,
        differences=differences,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()