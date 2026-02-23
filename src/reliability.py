from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import t

matplotlib.use("Agg")
sns.set_theme(style="whitegrid")

try:
    from sklearn.calibration import calibration_curve
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        brier_score_loss,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    SKLEARN_AVAILABLE = True
except ModuleNotFoundError:
    SKLEARN_AVAILABLE = False


METRIC_COLUMNS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "brier",
]


def _ensure_output_dirs() -> Dict[str, Path]:
    tables_dir = Path("outputs/tables")
    figures_dir = Path("outputs/figures")
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    return {"tables": tables_dir, "figures": figures_dir}


def _target_to_binary(series: pd.Series) -> pd.Series:
    mapping = {"<=50K": 0, ">50K": 1}
    mapped = series.astype(str).str.strip().map(mapping)
    if mapped.isna().all():
        values, _ = pd.factorize(series)
        return pd.Series(values, index=series.index).astype(int)
    return mapped.fillna(0).astype(int)


def _load_clean_df(context: Dict[str, Any]) -> pd.DataFrame | None:
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


def _confidence_interval(values: pd.Series, alpha: float = 0.95) -> tuple[float, float]:
    clean = values.dropna().astype(float)
    n = len(clean)
    if n == 0:
        return float("nan"), float("nan")
    if n == 1:
        x = float(clean.iloc[0])
        return x, x
    mean = clean.mean()
    stderr = clean.std(ddof=1) / np.sqrt(n)
    margin = t.ppf((1 + alpha) / 2.0, df=n - 1) * stderr
    return float(mean - margin), float(mean + margin)


def _build_model_pipeline(numeric_cols: list[str], categorical_cols: list[str]) -> Any:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ]
    )
    model = LogisticRegression(max_iter=1200, class_weight="balanced")
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def _plot_metric_distributions(metrics_df: pd.DataFrame, out_path: Path) -> None:
    melted = metrics_df.melt(id_vars=["run"], value_vars=METRIC_COLUMNS, var_name="metric", value_name="value")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=melted, x="metric", y="value", ax=ax, color="#9ecae1")
    sns.stripplot(data=melted, x="metric", y="value", ax=ax, color="#1f77b4", size=4, alpha=0.8)
    ax.set_title("Reliability - Metric Distribution Across Runs")
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _plot_calibration(calibration_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    ax.plot(
        calibration_df["mean_predicted_probability"],
        calibration_df["fraction_of_positives"],
        marker="o",
        color="#d62728",
        label="Observed",
    )
    ax.set_title("Reliability - Calibration Curve")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def run(config: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    if not SKLEARN_AVAILABLE:
        return {
            "step": "reliability",
            "status": "skipped",
            "message": (
                "scikit-learn is required for reliability modeling. "
                "Install dependencies in your venv: pip install -r requirements.txt"
            ),
        }

    df = _load_clean_df(context)
    if df is None:
        return {
            "step": "reliability",
            "status": "skipped",
            "message": "No cleaned dataset found. Run preprocessing first.",
        }

    schema_cfg = config.get("schema", {})
    target_col = schema_cfg.get("target", "income")
    if target_col not in df.columns:
        return {
            "step": "reliability",
            "status": "error",
            "message": f"Target column '{target_col}' not found in dataset.",
        }

    reliability_cfg = config.get("reliability", {})
    n_runs = int(reliability_cfg.get("n_runs", 10))
    test_size = float(reliability_cfg.get("test_size", 0.2))
    random_state = int(reliability_cfg.get("random_state", 42))
    n_bins = int(reliability_cfg.get("calibration_bins", 10))

    numeric_cols = [c for c in schema_cfg.get("numeric_features", []) if c in df.columns]
    categorical_cols = [c for c in schema_cfg.get("categorical_features", []) if c in df.columns]
    feature_cols = numeric_cols + categorical_cols
    if not feature_cols:
        return {
            "step": "reliability",
            "status": "error",
            "message": "No configured feature columns available for reliability analysis.",
        }

    model_df = df[feature_cols + [target_col]].copy()
    model_df = model_df.dropna(subset=[target_col]).reset_index(drop=True)

    X = model_df[feature_cols]
    y = _target_to_binary(model_df[target_col])

    splitter = StratifiedShuffleSplit(
        n_splits=n_runs,
        test_size=test_size,
        random_state=random_state,
    )

    run_rows: list[Dict[str, Any]] = []
    all_probs: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    for run_idx, (train_idx, test_idx) in enumerate(splitter.split(X, y), start=1):
        pipeline = _build_model_pipeline(numeric_cols=numeric_cols, categorical_cols=categorical_cols)

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)
        prob = pipeline.predict_proba(X_test)[:, 1]

        all_probs.append(prob)
        all_targets.append(y_test.to_numpy())

        roc_auc = float(roc_auc_score(y_test, prob)) if len(np.unique(y_test)) > 1 else float("nan")
        run_rows.append(
            {
                "run": run_idx,
                "accuracy": float(accuracy_score(y_test, pred)),
                "precision": float(precision_score(y_test, pred, zero_division=0)),
                "recall": float(recall_score(y_test, pred, zero_division=0)),
                "f1": float(f1_score(y_test, pred, zero_division=0)),
                "roc_auc": roc_auc,
                "brier": float(brier_score_loss(y_test, prob)),
            }
        )

    metrics_df = pd.DataFrame(run_rows)
    dirs = _ensure_output_dirs()
    run_metrics_path = dirs["tables"] / "reliability_run_metrics.csv"
    metrics_df.to_csv(run_metrics_path, index=False)

    summary_rows: list[Dict[str, Any]] = []
    for metric in METRIC_COLUMNS:
        series = metrics_df[metric].dropna()
        ci_low, ci_high = _confidence_interval(series)
        summary_rows.append(
            {
                "metric": metric,
                "mean": float(series.mean()) if len(series) else float("nan"),
                "std": float(series.std(ddof=1)) if len(series) > 1 else 0.0,
                "min": float(series.min()) if len(series) else float("nan"),
                "max": float(series.max()) if len(series) else float("nan"),
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "n_runs": int(len(series)),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    summary_path = dirs["tables"] / "reliability_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    _plot_metric_distributions(
        metrics_df=metrics_df,
        out_path=dirs["figures"] / "reliability_metric_distributions.png",
    )

    y_all = np.concatenate(all_targets) if all_targets else np.array([])
    p_all = np.concatenate(all_probs) if all_probs else np.array([])
    if len(y_all) and len(np.unique(y_all)) > 1:
        frac_pos, mean_pred = calibration_curve(y_all, p_all, n_bins=n_bins, strategy="quantile")
        calibration_df = pd.DataFrame(
            {
                "mean_predicted_probability": mean_pred,
                "fraction_of_positives": frac_pos,
            }
        )
    else:
        calibration_df = pd.DataFrame(
            {
                "mean_predicted_probability": [],
                "fraction_of_positives": [],
            }
        )
    calibration_path = dirs["tables"] / "reliability_calibration_curve.csv"
    calibration_df.to_csv(calibration_path, index=False)
    _plot_calibration(
        calibration_df=calibration_df if not calibration_df.empty else pd.DataFrame(
            {"mean_predicted_probability": [0.0, 1.0], "fraction_of_positives": [0.0, 1.0]}
        ),
        out_path=dirs["figures"] / "reliability_calibration_curve.png",
    )

    context["reliability_metrics_df"] = metrics_df
    context["reliability_summary_df"] = summary_df

    key_stats = summary_df.set_index("metric")["mean"].to_dict()
    return {
        "step": "reliability",
        "status": "ok",
        "message": "Reliability analysis completed (repeated runs, CI, calibration).",
        "n_runs": n_runs,
        "run_metrics_path": str(run_metrics_path),
        "summary_path": str(summary_path),
        "calibration_path": str(calibration_path),
        "key_metric_means": key_stats,
    }
