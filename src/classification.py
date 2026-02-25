from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

try:
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


MODEL_REGISTRY = {
    "logistic_regression": lambda seed: LogisticRegression(max_iter=1200, class_weight="balanced", random_state=seed),
    "random_forest": lambda seed: RandomForestClassifier(
        n_estimators=250,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=seed,
    ),
}


def _load_clean_df(context: Dict[str, Any]) -> pd.DataFrame | None:
    if "clean_df" in context:
        return context["clean_df"].copy()

    parts: List[pd.DataFrame] = []
    for split in ("train", "val", "test"):
        path = Path(f"data/processed/{split}.csv")
        if not path.exists():
            return None
        parts.append(pd.read_csv(path))
    return pd.concat(parts, axis=0, ignore_index=True)


def _ensure_output_dirs() -> Dict[str, Path]:
    table_dir = Path("outputs/tables")
    table_dir.mkdir(parents=True, exist_ok=True)
    return {"tables": table_dir}


def _target_to_binary(series: pd.Series) -> pd.Series:
    mapping = {"<=50K": 0, ">50K": 1}
    mapped = series.astype(str).str.strip().map(mapping)
    if mapped.isna().all():
        return pd.Series(pd.factorize(series)[0], index=series.index)
    return mapped.fillna(0).astype(int)


def _build_pipeline(numeric_cols: list[str], categorical_cols: list[str], model_name: str, seed: int) -> Pipeline:
    numeric_pipe = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
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
    model = MODEL_REGISTRY[model_name](seed)
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def run(config: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    if not SKLEARN_AVAILABLE:
        return {
            "step": "classification",
            "status": "skipped",
            "message": (
                "scikit-learn is required for classification. "
                "Install dependencies in your venv: pip install -r requirements.txt"
            ),
        }

    df = _load_clean_df(context)
    if df is None:
        return {
            "step": "classification",
            "status": "skipped",
            "message": "No cleaned dataset available. Run preprocessing first.",
        }

    schema_cfg = config.get("schema", {})
    cls_cfg = config.get("classification", {})

    target_col = schema_cfg.get("target", "income")
    if target_col not in df.columns:
        return {
            "step": "classification",
            "status": "error",
            "message": f"Target column '{target_col}' not found.",
        }

    numeric_cols = [c for c in schema_cfg.get("numeric_features", []) if c in df.columns]
    categorical_cols = [c for c in schema_cfg.get("categorical_features", []) if c in df.columns]
    feature_cols = numeric_cols + categorical_cols
    if not feature_cols:
        return {
            "step": "classification",
            "status": "error",
            "message": "No configured feature columns available for classification.",
        }

    configured_models = [m for m in cls_cfg.get("baseline_models", ["logistic_regression", "random_forest"]) if m in MODEL_REGISTRY]
    if not configured_models:
        return {
            "step": "classification",
            "status": "skipped",
            "message": "No supported baseline models configured.",
        }

    work_df = df[feature_cols + [target_col]].dropna(subset=[target_col]).reset_index(drop=True)
    X = work_df[feature_cols]
    y = _target_to_binary(work_df[target_col])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=int(config.get("split", {}).get("random_state", 42)),
        stratify=y,
    )

    rows: list[Dict[str, Any]] = []
    for model_name in configured_models:
        pipeline = _build_pipeline(
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            model_name=model_name,
            seed=int(config.get("split", {}).get("random_state", 42)),
        )
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)
        prob = pipeline.predict_proba(X_test)[:, 1]

        rows.append(
            {
                "model": model_name,
                "accuracy": float(accuracy_score(y_test, pred)),
                "precision": float(precision_score(y_test, pred, zero_division=0)),
                "recall": float(recall_score(y_test, pred, zero_division=0)),
                "f1": float(f1_score(y_test, pred, zero_division=0)),
                "roc_auc": float(roc_auc_score(y_test, prob)),
            }
        )

    metrics_df = pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)
    best_model = metrics_df.iloc[0]["model"] if not metrics_df.empty else None

    dirs = _ensure_output_dirs()
    metrics_path = dirs["tables"] / "classification_baselines.csv"
    metrics_df.to_csv(metrics_path, index=False)

    return {
        "step": "classification",
        "status": "ok",
        "message": f"Classification baselines completed. Best model by f1: {best_model}.",
        "metrics_path": str(metrics_path),
        "best_model": best_model,
    }
