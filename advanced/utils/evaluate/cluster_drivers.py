"""
Analyze which variables are driving cluster membership.

This script helps explain bad/weak clustering results by testing whether
clusters are associated with metadata variables or questionnaire variables.

Input:
    <output-dir>/phase_metadata_with_clusters.csv

Outputs:
    <output-dir>/eval/categorical_cluster_drivers.csv
    <output-dir>/eval/questionnaire_cluster_drivers.csv
    <output-dir>/eval/cluster_driver_summary.txt

Example:
    python advanced/utils/evaluate/cluster_drivers.py \
        --output-dir advanced/outputs/conv_ae_32_cohort_norm
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, kruskal


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


CATEGORICAL_VARIABLES = [
    "Phase",
    "Puzzler",
    "Cohort",
    "Round",
    "Individual",
]


EXCLUDED_NUMERIC_COLUMNS = {
    "Cluster",
    "Puzzler",
    "puzzler",
    "parent",
    "Parent",
    "WindowStartIndex",
    "WindowEndIndex",
}


def cramers_v(table: pd.DataFrame) -> float:
    """
    Compute Cramér's V effect size for a contingency table.

    Interpretation:
        0.00 ≈ no association
        0.10 ≈ weak association
        0.30 ≈ moderate association
        0.50+ ≈ strong association
    """
    chi2, _, _, _ = chi2_contingency(table)
    n = table.to_numpy().sum()

    if n == 0:
        return np.nan

    r, k = table.shape
    denominator = n * (min(r - 1, k - 1))

    if denominator == 0:
        return np.nan

    return float(np.sqrt(chi2 / denominator))


def epsilon_squared_kruskal(groups: list[np.ndarray], h_stat: float) -> float:
    """
    Approximate epsilon-squared effect size for Kruskal-Wallis.

    Larger values mean the variable differs more across clusters.
    """
    n = sum(len(g) for g in groups)
    k = len(groups)

    if n <= k:
        return np.nan

    return float((h_stat - k + 1) / (n - k))


def get_questionnaire_columns(df: pd.DataFrame) -> list[str]:
    """
    Select numeric questionnaire-like columns.

    Metadata/role/window columns are excluded.
    """
    numeric_cols = df.select_dtypes(include="number").columns

    return [
        col
        for col in numeric_cols
        if col not in EXCLUDED_NUMERIC_COLUMNS
    ]


def analyze_categorical_drivers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Test association between categorical variables and Cluster.

    Uses:
        chi-square test
        Cramér's V effect size
    """
    rows: list[dict[str, object]] = []

    for col in CATEGORICAL_VARIABLES:
        if col not in df.columns:
            continue

        usable = df[[col, "Cluster"]].dropna()

        if usable.empty or usable[col].nunique() < 2 or usable["Cluster"].nunique() < 2:
            rows.append(
                {
                    "variable": col,
                    "test": "chi_square",
                    "p_value": np.nan,
                    "cramers_v": np.nan,
                    "n_unique_values": usable[col].nunique() if not usable.empty else 0,
                    "n_samples_used": len(usable),
                    "interpretation": "not enough variation",
                }
            )
            continue

        table = pd.crosstab(usable[col], usable["Cluster"])

        chi2, p_value, dof, _ = chi2_contingency(table)
        v = cramers_v(table)

        rows.append(
            {
                "variable": col,
                "test": "chi_square",
                "chi2": float(chi2),
                "dof": int(dof),
                "p_value": float(p_value),
                "cramers_v": v,
                "n_unique_values": int(usable[col].nunique()),
                "n_samples_used": int(len(usable)),
                "interpretation": interpret_effect_size(v),
            }
        )

    out = pd.DataFrame(rows)

    if not out.empty and "cramers_v" in out.columns:
        out = out.sort_values("cramers_v", ascending=False, na_position="last")

    return out


def analyze_questionnaire_drivers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Test whether questionnaire variables differ across clusters.

    Uses:
        Kruskal-Wallis test
        epsilon-squared effect size

    Kruskal-Wallis is used because questionnaire scores are ordinal/noisy and
    we do not want to assume normality.
    """
    questionnaire_cols = get_questionnaire_columns(df)
    rows: list[dict[str, object]] = []

    for col in questionnaire_cols:
        groups = []

        for _, group in df.groupby("Cluster"):
            values = group[col].dropna().to_numpy()

            if len(values) > 0:
                groups.append(values)

        if len(groups) < 2:
            continue

        try:
            h_stat, p_value = kruskal(*groups)
            eps2 = epsilon_squared_kruskal(groups, h_stat)
        except Exception:
            h_stat, p_value, eps2 = np.nan, np.nan, np.nan

        cluster_means = df.groupby("Cluster")[col].mean()
        highest_cluster = cluster_means.idxmax()
        lowest_cluster = cluster_means.idxmin()
        mean_difference = cluster_means.max() - cluster_means.min()

        rows.append(
            {
                "variable": col,
                "test": "kruskal_wallis",
                "h_statistic": float(h_stat) if not pd.isna(h_stat) else np.nan,
                "p_value": float(p_value) if not pd.isna(p_value) else np.nan,
                "epsilon_squared": eps2,
                "highest_mean_cluster": highest_cluster,
                "highest_mean": float(cluster_means.max()),
                "lowest_mean_cluster": lowest_cluster,
                "lowest_mean": float(cluster_means.min()),
                "mean_difference": float(mean_difference),
                "n_samples_used": int(df[col].notna().sum()),
                "interpretation": interpret_effect_size(eps2),
            }
        )

    out = pd.DataFrame(rows)

    if not out.empty and "epsilon_squared" in out.columns:
        out = out.sort_values(
            "epsilon_squared",
            ascending=False,
            na_position="last",
        )

    return out


def interpret_effect_size(value: float | None) -> str:
    """
    Rough effect-size interpretation.

    Used for both Cramér's V and epsilon-squared.
    """
    if value is None or pd.isna(value):
        return "unknown"

    if value < 0.05:
        return "very weak"
    if value < 0.10:
        return "weak"
    if value < 0.30:
        return "moderate"
    if value < 0.50:
        return "strong"

    return "very strong"


def write_summary(
    eval_dir: Path,
    categorical_df: pd.DataFrame,
    questionnaire_df: pd.DataFrame,
) -> None:
    summary_path = eval_dir / "cluster_driver_summary.txt"

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Cluster driver analysis\n")
        f.write("=" * 24 + "\n\n")

        f.write("Goal\n")
        f.write("-" * 4 + "\n")
        f.write(
            "This analysis identifies which variables are most associated with "
            "cluster membership. It helps explain whether clusters are driven "
            "by experimental phase, role, cohort/session effects, individual "
            "differences, or questionnaire variables.\n\n"
        )

        f.write("Categorical metadata drivers\n")
        f.write("-" * 28 + "\n")

        if categorical_df.empty:
            f.write("No categorical driver results available.\n\n")
        else:
            cols = [
                "variable",
                "p_value",
                "cramers_v",
                "interpretation",
                "n_samples_used",
            ]
            f.write(categorical_df[cols].to_string(index=False))
            f.write("\n\n")

        f.write("Questionnaire / numeric drivers\n")
        f.write("-" * 33 + "\n")

        if questionnaire_df.empty:
            f.write("No questionnaire driver results available.\n\n")
        else:
            cols = [
                "variable",
                "p_value",
                "epsilon_squared",
                "interpretation",
                "highest_mean_cluster",
                "lowest_mean_cluster",
                "mean_difference",
            ]
            f.write(questionnaire_df[cols].head(15).to_string(index=False))
            f.write("\n\n")

        f.write("Interpretation guide\n")
        f.write("-" * 20 + "\n")
        f.write(
            "For categorical variables, Cramér's V measures association with clusters.\n"
            "For questionnaire variables, epsilon-squared estimates how strongly the "
            "variable differs across clusters.\n\n"
            "If Phase has very weak association but Cohort or Individual has stronger "
            "association, clusters are likely driven by baseline/session/subject effects "
            "rather than emotional phases.\n"
            "If questionnaire variables show only weak effects, then the clusters have "
            "limited emotional interpretability.\n"
        )

    print(f"Summary saved to: {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze which variables are associated with cluster membership."
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ADVANCED_DIR / "outputs" / "conv_ae_32_cohort_norm",
        help="Experiment output folder containing phase_metadata_with_clusters.csv.",
    )

    parser.add_argument(
        "--eval-dir-name",
        type=str,
        default="eval",
        help="Evaluation subfolder name.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = args.output_dir
    eval_dir = output_dir / args.eval_dir_name
    eval_dir.mkdir(parents=True, exist_ok=True)

    phase_meta_path = output_dir / "phase_metadata_with_clusters.csv"

    if not phase_meta_path.exists():
        raise FileNotFoundError(f"Could not find {phase_meta_path}")

    df = pd.read_csv(phase_meta_path)

    if "Cluster" not in df.columns:
        raise ValueError("phase_metadata_with_clusters.csv must contain Cluster column.")

    categorical_df = analyze_categorical_drivers(df)
    questionnaire_df = analyze_questionnaire_drivers(df)

    categorical_path = eval_dir / "categorical_cluster_drivers.csv"
    questionnaire_path = eval_dir / "questionnaire_cluster_drivers.csv"

    categorical_df.to_csv(categorical_path, index=False)
    questionnaire_df.to_csv(questionnaire_path, index=False)

    write_summary(
        eval_dir=eval_dir,
        categorical_df=categorical_df,
        questionnaire_df=questionnaire_df,
    )

    print("Saved:")
    print(categorical_path)
    print(questionnaire_path)


if __name__ == "__main__":
    main()