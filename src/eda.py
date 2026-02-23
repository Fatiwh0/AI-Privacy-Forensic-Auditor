from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency, pointbiserialr

matplotlib.use("Agg")
sns.set_theme(style="whitegrid")


def _ensure_dirs() -> Dict[str, Path]:
    fig_dir = Path("outputs/figures")
    table_dir = Path("outputs/tables")
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    return {"figures": fig_dir, "tables": table_dir}


def _load_dataset_from_context_or_disk(context: Dict[str, Any]) -> pd.DataFrame | None:
    if "clean_df" in context:
        return context["clean_df"].copy()

    train_path = Path("data/processed/train.csv")
    val_path = Path("data/processed/val.csv")
    test_path = Path("data/processed/test.csv")
    if train_path.exists() and val_path.exists() and test_path.exists():
        return pd.concat(
            [pd.read_csv(train_path), pd.read_csv(val_path), pd.read_csv(test_path)],
            axis=0,
            ignore_index=True,
        )
    return None


def _target_binary(series: pd.Series) -> pd.Series:
    mapping = {"<=50K": 0, ">50K": 1}
    mapped = series.astype(str).str.strip().map(mapping)
    if mapped.isna().all():
        return pd.Series(pd.factorize(series)[0], index=series.index)
    return mapped.fillna(0).astype(int)


def _cramers_v(contingency: pd.DataFrame) -> float:
    if contingency.empty:
        return 0.0
    chi2, _, _, _ = chi2_contingency(contingency)
    n = contingency.values.sum()
    if n == 0:
        return 0.0
    phi2 = chi2 / n
    r, k = contingency.shape
    phi2_corr = max(0.0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    r_corr = r - ((r - 1) ** 2) / (n - 1)
    k_corr = k - ((k - 1) ** 2) / (n - 1)
    denom = max(1e-12, min((k_corr - 1), (r_corr - 1)))
    return float(np.sqrt(phi2_corr / denom))


def _plot_numeric_histograms(df: pd.DataFrame, numeric_cols: list[str], out_path: Path) -> None:
    cols = [c for c in numeric_cols if c in df.columns]
    if not cols:
        return
    n = len(cols)
    rows = int(np.ceil(n / 3))
    fig, axes = plt.subplots(rows, 3, figsize=(18, 4.5 * rows))
    axes = np.array(axes).reshape(-1)
    for i, col in enumerate(cols):
        ax = axes[i]
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        sns.histplot(series, bins=30, kde=True, ax=ax, color="#1f77b4")
        ax.set_title(f"Distribution - {col}")
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _plot_categorical_bars(df: pd.DataFrame, categorical_cols: list[str], out_path: Path) -> None:
    cols = [c for c in categorical_cols if c in df.columns]
    if not cols:
        return
    n = len(cols)
    rows = int(np.ceil(n / 2))
    fig, axes = plt.subplots(rows, 2, figsize=(16, 4.5 * rows))
    axes = np.array(axes).reshape(-1)
    for i, col in enumerate(cols):
        ax = axes[i]
        counts = df[col].fillna("MISSING").astype(str).value_counts().head(10)
        sns.barplot(x=counts.values, y=counts.index, ax=ax, color="#2ca02c")
        ax.set_title(f"Top categories - {col}")
        ax.set_xlabel("Count")
        ax.set_ylabel("")
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _plot_correlation_heatmap(corr_df: pd.DataFrame, out_path: Path) -> None:
    if corr_df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_df, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title("Numeric Correlation Matrix")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _plot_boxplots_by_target(
    df: pd.DataFrame,
    numeric_cols: list[str],
    target_col: str,
    out_path: Path,
) -> None:
    cols = [c for c in numeric_cols if c in df.columns]
    if target_col not in df.columns or not cols:
        return
    n = len(cols)
    rows = int(np.ceil(n / 3))
    fig, axes = plt.subplots(rows, 3, figsize=(18, 4.5 * rows))
    axes = np.array(axes).reshape(-1)
    for i, col in enumerate(cols):
        ax = axes[i]
        temp = df[[col, target_col]].copy()
        temp[col] = pd.to_numeric(temp[col], errors="coerce")
        sns.boxplot(data=temp, x=target_col, y=col, ax=ax)
        ax.set_title(f"{col} by {target_col}")
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _plot_pca_scatter(
    pca_df: pd.DataFrame,
    target_col: str,
    out_path: Path,
    explained_variance: tuple[float, float],
) -> None:
    if pca_df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    if target_col in pca_df.columns:
        sns.scatterplot(
            data=pca_df,
            x="pc1",
            y="pc2",
            hue=target_col,
            alpha=0.6,
            s=24,
            ax=ax,
        )
    else:
        sns.scatterplot(data=pca_df, x="pc1", y="pc2", alpha=0.6, s=24, ax=ax)
    ax.set_title(
        f"PCA Projection (PC1={explained_variance[0]:.2%}, PC2={explained_variance[1]:.2%})"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def run(config: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    df = _load_dataset_from_context_or_disk(context)
    if df is None:
        return {
            "step": "eda",
            "status": "skipped",
            "message": "No cleaned dataset available. Run preprocessing first.",
        }

    dirs = _ensure_dirs()
    schema = config.get("schema", {})
    target_col = schema.get("target", "income")
    numeric_cols = [c for c in schema.get("numeric_features", []) if c in df.columns]
    categorical_cols = [c for c in schema.get("categorical_features", []) if c in df.columns]

    table_paths: Dict[str, str] = {}
    figure_paths: Dict[str, str] = {}

    # Univariate tables
    numeric_summary = df[numeric_cols].apply(pd.to_numeric, errors="coerce").describe().T
    numeric_summary["missing"] = df[numeric_cols].isna().sum()
    numeric_summary_path = dirs["tables"] / "eda_univariate_numeric_summary.csv"
    numeric_summary.to_csv(numeric_summary_path)
    table_paths["univariate_numeric_summary"] = str(numeric_summary_path)

    cat_rows: list[Dict[str, Any]] = []
    for col in categorical_cols + ([target_col] if target_col in df.columns else []):
        counts = df[col].fillna("MISSING").astype(str).value_counts(dropna=False)
        total = counts.sum()
        for category, count in counts.items():
            cat_rows.append(
                {
                    "feature": col,
                    "category": category,
                    "count": int(count),
                    "percentage": float(count / total),
                }
            )
    categorical_summary = pd.DataFrame(cat_rows)
    categorical_summary_path = dirs["tables"] / "eda_univariate_categorical_summary.csv"
    categorical_summary.to_csv(categorical_summary_path, index=False)
    table_paths["univariate_categorical_summary"] = str(categorical_summary_path)

    missing_summary = pd.DataFrame(
        {
            "column": df.columns,
            "missing_count": df.isna().sum().values,
            "missing_pct": (df.isna().mean().values * 100.0),
        }
    )
    missing_summary_path = dirs["tables"] / "eda_missingness_summary.csv"
    missing_summary.to_csv(missing_summary_path, index=False)
    table_paths["missingness_summary"] = str(missing_summary_path)

    # Univariate figures
    hist_path = dirs["figures"] / "eda_univariate_numeric_histograms.png"
    _plot_numeric_histograms(df=df, numeric_cols=numeric_cols, out_path=hist_path)
    figure_paths["univariate_numeric_histograms"] = str(hist_path)

    cat_bar_path = dirs["figures"] / "eda_univariate_categorical_bars.png"
    _plot_categorical_bars(df=df, categorical_cols=categorical_cols + [target_col], out_path=cat_bar_path)
    figure_paths["univariate_categorical_bars"] = str(cat_bar_path)

    # Bivariate tables
    corr_df = (
        df[numeric_cols].apply(pd.to_numeric, errors="coerce").corr().round(4)
        if numeric_cols
        else pd.DataFrame()
    )
    corr_path = dirs["tables"] / "eda_bivariate_numeric_correlation.csv"
    corr_df.to_csv(corr_path)
    table_paths["bivariate_numeric_correlation"] = str(corr_path)

    target_numeric = _target_binary(df[target_col]) if target_col in df.columns else pd.Series(dtype=int)
    pb_rows: list[Dict[str, Any]] = []
    for col in numeric_cols:
        series = pd.to_numeric(df[col], errors="coerce")
        valid = ~(series.isna() | target_numeric.isna())
        if valid.sum() < 3:
            continue
        corr, pval = pointbiserialr(target_numeric[valid], series[valid])
        pb_rows.append(
            {
                "feature": col,
                "point_biserial_corr": float(corr),
                "p_value": float(pval),
            }
        )
    pb_df = pd.DataFrame(pb_rows).sort_values("p_value", ascending=True)
    pb_path = dirs["tables"] / "eda_bivariate_numeric_vs_target.csv"
    pb_df.to_csv(pb_path, index=False)
    table_paths["bivariate_numeric_vs_target"] = str(pb_path)

    chi_rows: list[Dict[str, Any]] = []
    if target_col in df.columns:
        for col in categorical_cols:
            temp = df[[col, target_col]].copy()
            temp[col] = temp[col].fillna("MISSING").astype(str)
            temp[target_col] = temp[target_col].fillna("MISSING").astype(str)
            contingency = pd.crosstab(temp[col], temp[target_col])
            if contingency.empty:
                continue
            chi2, pval, dof, _ = chi2_contingency(contingency)
            chi_rows.append(
                {
                    "feature": col,
                    "chi2": float(chi2),
                    "dof": int(dof),
                    "p_value": float(pval),
                    "cramers_v": _cramers_v(contingency),
                }
            )
    chi_df = pd.DataFrame(chi_rows).sort_values("p_value", ascending=True)
    chi_path = dirs["tables"] / "eda_bivariate_categorical_vs_target.csv"
    chi_df.to_csv(chi_path, index=False)
    table_paths["bivariate_categorical_vs_target"] = str(chi_path)

    # Bivariate figures
    corr_fig_path = dirs["figures"] / "eda_bivariate_correlation_heatmap.png"
    _plot_correlation_heatmap(corr_df=corr_df, out_path=corr_fig_path)
    figure_paths["bivariate_correlation_heatmap"] = str(corr_fig_path)

    boxplot_path = dirs["figures"] / "eda_bivariate_boxplots_by_target.png"
    _plot_boxplots_by_target(
        df=df,
        numeric_cols=numeric_cols,
        target_col=target_col,
        out_path=boxplot_path,
    )
    figure_paths["bivariate_boxplots_by_target"] = str(boxplot_path)

    # Multivariate: PCA using numpy SVD to avoid heavy dependencies.
    num_df = df[numeric_cols].apply(pd.to_numeric, errors="coerce").copy()
    num_df = num_df.fillna(num_df.mean())
    pca_results = pd.DataFrame()
    pca_info = {"pc1_var": 0.0, "pc2_var": 0.0}
    if not num_df.empty and num_df.shape[1] >= 2:
        X = num_df.to_numpy(dtype=float)
        mu = X.mean(axis=0, keepdims=True)
        sigma = X.std(axis=0, keepdims=True)
        sigma[sigma == 0] = 1.0
        Z = (X - mu) / sigma

        _, singular_values, vt = np.linalg.svd(Z, full_matrices=False)
        eigvals = (singular_values ** 2) / max(1, Z.shape[0] - 1)
        explained = eigvals / eigvals.sum()
        pcs = Z @ vt.T[:, :2]

        pca_results = pd.DataFrame(pcs, columns=["pc1", "pc2"], index=df.index)
        if target_col in df.columns:
            pca_results[target_col] = df[target_col].astype(str).values

        pca_info["pc1_var"] = float(explained[0]) if len(explained) > 0 else 0.0
        pca_info["pc2_var"] = float(explained[1]) if len(explained) > 1 else 0.0

        explained_df = pd.DataFrame(
            {
                "component": [f"PC{i+1}" for i in range(len(explained))],
                "explained_variance_ratio": explained,
            }
        )
        explained_path = dirs["tables"] / "eda_multivariate_pca_explained_variance.csv"
        explained_df.to_csv(explained_path, index=False)
        table_paths["multivariate_pca_explained_variance"] = str(explained_path)

        loadings_df = pd.DataFrame(vt.T, index=numeric_cols, columns=[f"PC{i+1}" for i in range(vt.shape[0])])
        loadings_path = dirs["tables"] / "eda_multivariate_pca_loadings.csv"
        loadings_df.to_csv(loadings_path)
        table_paths["multivariate_pca_loadings"] = str(loadings_path)

        pca_scatter_path = dirs["figures"] / "eda_multivariate_pca_scatter.png"
        _plot_pca_scatter(
            pca_df=pca_results,
            target_col=target_col,
            out_path=pca_scatter_path,
            explained_variance=(pca_info["pc1_var"], pca_info["pc2_var"]),
        )
        figure_paths["multivariate_pca_scatter"] = str(pca_scatter_path)

    return {
        "step": "eda",
        "status": "ok",
        "message": "EDA completed with univariate, bivariate, and multivariate outputs.",
        "rows_analyzed": int(df.shape[0]),
        "table_outputs": table_paths,
        "figure_outputs": figure_paths,
        "pca_summary": pca_info,
    }
