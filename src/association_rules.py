from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd


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


def _discretize_numeric(df: pd.DataFrame, numeric_cols: Iterable[str]) -> pd.DataFrame:
    result = pd.DataFrame(index=df.index)
    for col in numeric_cols:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        non_null = series.dropna()
        if non_null.empty:
            continue
        q1, q2 = non_null.quantile([0.33, 0.66]).tolist()
        bins = [-float("inf"), q1, q2, float("inf")]
        labels = ["low", "mid", "high"]
        bucketed = pd.cut(series, bins=bins, labels=labels, include_lowest=True)
        result[col] = bucketed.astype("string").fillna("MISSING")
    return result


def _prepare_transactions(
    df: pd.DataFrame,
    categorical_cols: Iterable[str],
    numeric_cols: Iterable[str],
    max_categories_per_col: int = 12,
) -> pd.DataFrame:
    transactions = pd.DataFrame(index=df.index)

    for col in categorical_cols:
        if col not in df.columns:
            continue
        values = df[col].astype("string").fillna("MISSING")
        top_values = values.value_counts().head(max_categories_per_col).index
        values = values.where(values.isin(top_values), other="OTHER")
        transactions[col] = values

    binned_numeric = _discretize_numeric(df, numeric_cols)
    for col in binned_numeric.columns:
        transactions[col] = binned_numeric[col]

    if transactions.empty:
        return transactions

    for col in transactions.columns:
        transactions[col] = transactions[col].astype(str).map(lambda v: f"{col}={v}")

    return transactions


def _itemset_support(transactions: pd.DataFrame, items: tuple[str, ...]) -> float:
    if transactions.empty:
        return 0.0
    mask = pd.Series(True, index=transactions.index)
    for item in items:
        mask &= (transactions == item).any(axis=1)
    return float(mask.mean())


def _mine_frequent_itemsets(
    transactions: pd.DataFrame,
    min_support: float,
    max_len: int = 3,
) -> Dict[tuple[str, ...], float]:
    if transactions.empty:
        return {}

    unique_items = sorted(pd.unique(transactions.to_numpy().ravel()))
    unique_items = [item for item in unique_items if isinstance(item, str) and item]

    supports: Dict[tuple[str, ...], float] = {}
    for k in range(1, max_len + 1):
        for combo in combinations(unique_items, k):
            support = _itemset_support(transactions, combo)
            if support >= min_support:
                supports[combo] = support
    return supports


def _generate_rules(
    supports: Dict[tuple[str, ...], float],
    min_confidence: float,
    min_lift: float,
) -> pd.DataFrame:
    rows: list[Dict[str, Any]] = []

    for itemset, support_itemset in supports.items():
        if len(itemset) < 2:
            continue

        for r in range(1, len(itemset)):
            for antecedent in combinations(itemset, r):
                consequent = tuple(i for i in itemset if i not in antecedent)
                ant_support = supports.get(tuple(sorted(antecedent)))
                cons_support = supports.get(tuple(sorted(consequent)))
                if not ant_support or not cons_support:
                    continue

                confidence = support_itemset / ant_support
                lift = confidence / cons_support
                if confidence < min_confidence or lift < min_lift:
                    continue

                rows.append(
                    {
                        "antecedent": " & ".join(antecedent),
                        "consequent": " & ".join(consequent),
                        "support": support_itemset,
                        "confidence": confidence,
                        "lift": lift,
                        "antecedent_support": ant_support,
                        "consequent_support": cons_support,
                    }
                )

    if not rows:
        return pd.DataFrame(
            columns=[
                "antecedent",
                "consequent",
                "support",
                "confidence",
                "lift",
                "antecedent_support",
                "consequent_support",
            ]
        )

    rules_df = pd.DataFrame(rows)
    rules_df = rules_df.sort_values(["lift", "confidence", "support"], ascending=False).reset_index(drop=True)
    return rules_df


def run(config: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    df = _load_clean_df(context)
    if df is None:
        return {
            "step": "association_rules",
            "status": "skipped",
            "message": "No cleaned dataset available. Run preprocessing first.",
        }

    schema_cfg = config.get("schema", {})
    assoc_cfg = config.get("association_rules", {})

    categorical_cols = [c for c in schema_cfg.get("categorical_features", []) if c in df.columns]
    numeric_cols = [c for c in schema_cfg.get("numeric_features", []) if c in df.columns]
    if not categorical_cols and not numeric_cols:
        return {
            "step": "association_rules",
            "status": "error",
            "message": "No eligible columns found for association-rule mining.",
        }

    min_support = float(assoc_cfg.get("min_support", 0.05))
    min_confidence = float(assoc_cfg.get("min_confidence", 0.6))
    min_lift = float(assoc_cfg.get("min_lift", 1.1))

    transactions = _prepare_transactions(
        df=df,
        categorical_cols=categorical_cols,
        numeric_cols=numeric_cols,
    )
    supports = _mine_frequent_itemsets(transactions, min_support=min_support, max_len=3)
    rules_df = _generate_rules(supports, min_confidence=min_confidence, min_lift=min_lift)

    dirs = _ensure_output_dirs()
    rules_path = dirs["tables"] / "association_rules.csv"
    rules_df.to_csv(rules_path, index=False)

    top_rules_path = dirs["tables"] / "association_rules_top10.csv"
    rules_df.head(10).to_csv(top_rules_path, index=False)

    return {
        "step": "association_rules",
        "status": "ok",
        "message": f"Association-rule mining complete. Generated {len(rules_df)} rules.",
        "rules_path": str(rules_path),
        "top_rules_path": str(top_rules_path),
        "n_rules": int(len(rules_df)),
    }
