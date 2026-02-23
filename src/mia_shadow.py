from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")
sns.set_theme(style="whitegrid")

try:
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
        roc_curve,
    )
    from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    SKLEARN_AVAILABLE = True
except ModuleNotFoundError:
    SKLEARN_AVAILABLE = False


def _ensure_dirs() -> Dict[str, Path]:
    tables_dir = Path("outputs/tables")
    figures_dir = Path("outputs/figures")
    models_dir = Path("outputs/models")
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    return {"tables": tables_dir, "figures": figures_dir, "models": models_dir}


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


def _build_target_pipeline(
    numeric_cols: list[str],
    categorical_cols: list[str],
    model_name: str,
    seed: int,
) -> Any:
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

    if model_name == "logistic_regression":
        model = LogisticRegression(max_iter=1200, class_weight="balanced", random_state=seed)
    else:
        model = RandomForestClassifier(
            n_estimators=300,
            random_state=seed,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )

    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def _entropy_binary(prob_1: np.ndarray) -> np.ndarray:
    eps = 1e-12
    p1 = np.clip(prob_1, eps, 1.0 - eps)
    p0 = 1.0 - p1
    return -(p1 * np.log(p1) + p0 * np.log(p0))


def _build_attack_features(
    y_true: np.ndarray,
    prob_pos: np.ndarray,
    feature_list: list[str],
) -> pd.DataFrame:
    y_true = np.asarray(y_true).astype(int)
    prob_pos = np.asarray(prob_pos).astype(float)
    prob_neg = 1.0 - prob_pos
    max_conf = np.maximum(prob_pos, prob_neg)
    top2_gap = np.abs(prob_pos - prob_neg)
    entropy = _entropy_binary(prob_pos)
    true_class_conf = np.where(y_true == 1, prob_pos, prob_neg)
    loss = -np.log(np.clip(true_class_conf, 1e-12, 1.0))

    catalog = {
        "max_confidence": max_conf,
        "entropy": entropy,
        "top2_gap": top2_gap,
        "true_class_confidence": true_class_conf,
        "loss": loss,
        "prob_positive": prob_pos,
        "prob_negative": prob_neg,
    }

    chosen = {}
    for name in feature_list:
        if name in catalog:
            chosen[name] = catalog[name]

    if not chosen:
        chosen = {
            "max_confidence": max_conf,
            "entropy": entropy,
            "top2_gap": top2_gap,
            "true_class_confidence": true_class_conf,
        }
    return pd.DataFrame(chosen)


def _collect_member_features(
    model: Any,
    X_member: pd.DataFrame,
    y_member: np.ndarray,
    X_non_member: pd.DataFrame,
    y_non_member: np.ndarray,
    feature_list: list[str],
) -> pd.DataFrame:
    prob_member = model.predict_proba(X_member)[:, 1]
    prob_non_member = model.predict_proba(X_non_member)[:, 1]

    feat_member = _build_attack_features(y_member, prob_member, feature_list)
    feat_non_member = _build_attack_features(y_non_member, prob_non_member, feature_list)

    feat_member["membership"] = 1
    feat_non_member["membership"] = 0
    return pd.concat([feat_member, feat_non_member], axis=0, ignore_index=True)


def _plot_score_distributions(
    y_true: np.ndarray,
    y_score: np.ndarray,
    out_path: Path,
) -> None:
    df = pd.DataFrame({"membership": y_true, "attack_score": y_score})
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.kdeplot(
        data=df[df["membership"] == 1],
        x="attack_score",
        fill=True,
        alpha=0.4,
        label="Member",
        color="#d62728",
        ax=ax,
    )
    sns.kdeplot(
        data=df[df["membership"] == 0],
        x="attack_score",
        fill=True,
        alpha=0.4,
        label="Non-member",
        color="#1f77b4",
        ax=ax,
    )
    ax.set_title("MIA Attack Score Distribution")
    ax.set_xlabel("Predicted Membership Probability")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _plot_roc(y_true: np.ndarray, y_score: np.ndarray, out_path: Path) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#2ca02c", label=f"Attack ROC (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("MIA Attack ROC Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return float(auc)


def _risk_level(attack_auc: float) -> str:
    if attack_auc >= 0.80:
        return "high"
    if attack_auc >= 0.65:
        return "moderate"
    return "low"


def run(config: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    if not SKLEARN_AVAILABLE:
        return {
            "step": "mia_shadow",
            "status": "skipped",
            "message": (
                "scikit-learn is required for MIA analysis. "
                "Install dependencies in your venv: pip install -r requirements.txt"
            ),
        }

    df = _load_clean_df(context)
    if df is None:
        return {
            "step": "mia_shadow",
            "status": "skipped",
            "message": "No cleaned dataset available. Run preprocessing first.",
        }

    schema_cfg = config.get("schema", {})
    target_col = schema_cfg.get("target", "income")
    if target_col not in df.columns:
        return {
            "step": "mia_shadow",
            "status": "error",
            "message": f"Target column '{target_col}' not found.",
        }

    numeric_cols = [c for c in schema_cfg.get("numeric_features", []) if c in df.columns]
    categorical_cols = [c for c in schema_cfg.get("categorical_features", []) if c in df.columns]
    feature_cols = numeric_cols + categorical_cols
    if not feature_cols:
        return {
            "step": "mia_shadow",
            "status": "error",
            "message": "No feature columns available for MIA.",
        }

    mia_cfg = config.get("mia", {})
    n_shadow = int(mia_cfg.get("shadow_models", 5))
    world_split = float(mia_cfg.get("world_split", 0.5))
    member_fraction = float(mia_cfg.get("member_fraction", 0.5))
    seed = int(config.get("split", {}).get("random_state", 42))
    if not (0.05 <= world_split <= 0.95):
        return {
            "step": "mia_shadow",
            "status": "error",
            "message": f"Invalid mia.world_split={world_split}. Use value between 0.05 and 0.95.",
        }
    if not (0.05 <= member_fraction <= 0.95):
        return {
            "step": "mia_shadow",
            "status": "error",
            "message": f"Invalid mia.member_fraction={member_fraction}. Use value between 0.05 and 0.95.",
        }
    attack_feature_list = mia_cfg.get(
        "probability_features",
        ["max_confidence", "entropy", "top2_gap", "true_class_confidence"],
    )
    target_model_name = mia_cfg.get("target_model", "random_forest")

    work_df = df[feature_cols + [target_col]].dropna(subset=[target_col]).reset_index(drop=True)
    X = work_df[feature_cols]
    y = _target_to_binary(work_df[target_col])

    X_target_pool, X_shadow_pool, y_target_pool, y_shadow_pool = train_test_split(
        X,
        y,
        test_size=world_split,
        random_state=seed,
        stratify=y,
    )

    X_target_member, X_target_nonmember, y_target_member, y_target_nonmember = train_test_split(
        X_target_pool,
        y_target_pool,
        test_size=(1.0 - member_fraction),
        random_state=seed + 1,
        stratify=y_target_pool,
    )
    target_model = _build_target_pipeline(
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        model_name=target_model_name,
        seed=seed,
    )
    target_model.fit(X_target_member, y_target_member)

    target_nonmember_prob = target_model.predict_proba(X_target_nonmember)[:, 1]
    target_nonmember_pred = (target_nonmember_prob >= 0.5).astype(int)
    target_utility_metrics = pd.DataFrame(
        [
            {
                "split": "target_nonmember_eval",
                "accuracy": accuracy_score(y_target_nonmember, target_nonmember_pred),
                "precision": precision_score(y_target_nonmember, target_nonmember_pred, zero_division=0),
                "recall": recall_score(y_target_nonmember, target_nonmember_pred, zero_division=0),
                "f1": f1_score(y_target_nonmember, target_nonmember_pred, zero_division=0),
                "roc_auc": roc_auc_score(y_target_nonmember, target_nonmember_prob),
            }
        ]
    )

    splitter = StratifiedShuffleSplit(
        n_splits=n_shadow,
        test_size=(1.0 - member_fraction),
        random_state=seed + 100,
    )
    attack_train_frames: list[pd.DataFrame] = []
    shadow_rows: list[Dict[str, Any]] = []
    for shadow_id, (member_idx, nonmember_idx) in enumerate(splitter.split(X_shadow_pool, y_shadow_pool), start=1):
        X_shadow_member = X_shadow_pool.iloc[member_idx]
        y_shadow_member = y_shadow_pool.iloc[member_idx].to_numpy()
        X_shadow_nonmember = X_shadow_pool.iloc[nonmember_idx]
        y_shadow_nonmember = y_shadow_pool.iloc[nonmember_idx].to_numpy()

        shadow_model = _build_target_pipeline(
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            model_name=target_model_name,
            seed=seed + shadow_id,
        )
        shadow_model.fit(X_shadow_member, y_shadow_member)

        frame = _collect_member_features(
            model=shadow_model,
            X_member=X_shadow_member,
            y_member=y_shadow_member,
            X_non_member=X_shadow_nonmember,
            y_non_member=y_shadow_nonmember,
            feature_list=attack_feature_list,
        )
        frame["shadow_id"] = shadow_id
        attack_train_frames.append(frame)
        shadow_rows.append(
            {
                "shadow_id": shadow_id,
                "member_rows": int(len(X_shadow_member)),
                "non_member_rows": int(len(X_shadow_nonmember)),
            }
        )

    attack_train_df = pd.concat(attack_train_frames, axis=0, ignore_index=True)

    attack_test_df = _collect_member_features(
        model=target_model,
        X_member=X_target_member,
        y_member=y_target_member.to_numpy(),
        X_non_member=X_target_nonmember,
        y_non_member=y_target_nonmember.to_numpy(),
        feature_list=attack_feature_list,
    )

    attack_features = [c for c in attack_train_df.columns if c not in ("membership", "shadow_id")]
    X_attack_train = attack_train_df[attack_features]
    y_attack_train = attack_train_df["membership"].astype(int)
    X_attack_test = attack_test_df[attack_features]
    y_attack_test = attack_test_df["membership"].astype(int)

    attack_model = RandomForestClassifier(
        n_estimators=400,
        random_state=seed + 999,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    attack_model.fit(X_attack_train, y_attack_train)
    attack_pred = attack_model.predict(X_attack_test)
    attack_score = attack_model.predict_proba(X_attack_test)[:, 1]

    attack_auc = roc_auc_score(y_attack_test, attack_score)
    attack_metrics = {
        "attack_accuracy": float(accuracy_score(y_attack_test, attack_pred)),
        "attack_precision": float(precision_score(y_attack_test, attack_pred, zero_division=0)),
        "attack_recall": float(recall_score(y_attack_test, attack_pred, zero_division=0)),
        "attack_f1": float(f1_score(y_attack_test, attack_pred, zero_division=0)),
        "attack_auc": float(attack_auc),
        "risk_level": _risk_level(float(attack_auc)),
    }

    cm = confusion_matrix(y_attack_test, attack_pred, labels=[0, 1])
    cm_df = pd.DataFrame(
        cm,
        index=["true_non_member", "true_member"],
        columns=["pred_non_member", "pred_member"],
    )

    dirs = _ensure_dirs()
    attack_train_path = dirs["tables"] / "mia_attack_train_features.csv"
    attack_test_path = dirs["tables"] / "mia_attack_test_features.csv"
    shadow_summary_path = dirs["tables"] / "mia_shadow_summary.csv"
    target_utility_path = dirs["tables"] / "mia_target_model_metrics.csv"
    attack_metrics_path = dirs["tables"] / "mia_attack_metrics.csv"
    confusion_path = dirs["tables"] / "mia_attack_confusion_matrix.csv"
    feature_importance_path = dirs["tables"] / "mia_attack_feature_importance.csv"

    attack_train_df.to_csv(attack_train_path, index=False)
    attack_test_df.to_csv(attack_test_path, index=False)
    pd.DataFrame(shadow_rows).to_csv(shadow_summary_path, index=False)
    target_utility_metrics.to_csv(target_utility_path, index=False)
    pd.DataFrame([attack_metrics]).to_csv(attack_metrics_path, index=False)
    cm_df.to_csv(confusion_path)

    fi_df = pd.DataFrame(
        {
            "feature": attack_features,
            "importance": attack_model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    fi_df.to_csv(feature_importance_path, index=False)

    roc_fig_path = dirs["figures"] / "mia_attack_roc_curve.png"
    dist_fig_path = dirs["figures"] / "mia_attack_score_distribution.png"
    _plot_roc(y_attack_test.to_numpy(), attack_score, roc_fig_path)
    _plot_score_distributions(y_attack_test.to_numpy(), attack_score, dist_fig_path)

    context["mia_attack_metrics"] = attack_metrics
    context["mia_attack_test_df"] = attack_test_df

    return {
        "step": "mia_shadow",
        "status": "ok",
        "message": "MIA shadow attack completed with leakage-risk metrics.",
        "shadow_models": n_shadow,
        "attack_train_rows": int(len(attack_train_df)),
        "attack_test_rows": int(len(attack_test_df)),
        "target_model": target_model_name,
        "attack_metrics": attack_metrics,
        "artifacts": {
            "attack_metrics": str(attack_metrics_path),
            "confusion_matrix": str(confusion_path),
            "feature_importance": str(feature_importance_path),
            "attack_roc_figure": str(roc_fig_path),
            "score_distribution_figure": str(dist_fig_path),
        },
    }
