from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd


def _normalize_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    string_columns = df.select_dtypes(include=["object", "string"]).columns
    for col in string_columns:
        df[col] = df[col].map(lambda x: x.strip() if isinstance(x, str) else x)
    return df


def _normalize_target(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    if target_col not in df.columns:
        return df
    df[target_col] = (
        df[target_col]
        .astype(str)
        .str.strip()
        .str.replace(".", "", regex=False)
    )
    return df


def _stratified_sample_indices(
    df: pd.DataFrame,
    target_col: str,
    frac: float,
    random_state: int,
) -> list[int]:
    rng = np.random.default_rng(random_state)
    chosen_indices: list[int] = []
    for _, group in df.groupby(target_col, dropna=False):
        n = len(group)
        take = int(round(n * frac))
        if frac > 0 and take == 0 and n > 0:
            take = 1
        if take >= n:
            selected = group.index.to_numpy()
        else:
            selected = rng.choice(group.index.to_numpy(), size=take, replace=False)
        chosen_indices.extend(selected.tolist())
    return chosen_indices


def _split_dataset(df: pd.DataFrame, target_col: str, split_cfg: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    train_size = float(split_cfg.get("train_size", 0.6))
    val_size = float(split_cfg.get("val_size", 0.2))
    test_size = float(split_cfg.get("test_size", 0.2))
    random_state = int(split_cfg.get("random_state", 42))
    stratify_enabled = bool(split_cfg.get("stratify", True))

    total = train_size + val_size + test_size
    if abs(total - 1.0) > 1e-9:
        raise ValueError(
            f"Invalid split sizes (sum={total}). Expected train+val+test to equal 1.0."
        )

    temp_size = val_size + test_size

    if temp_size == 0:
        return {
            "train": df.copy(),
            "val": pd.DataFrame(columns=df.columns),
            "test": pd.DataFrame(columns=df.columns),
        }

    if stratify_enabled and target_col in df.columns:
        temp_indices = _stratified_sample_indices(
            df=df,
            target_col=target_col,
            frac=temp_size,
            random_state=random_state,
        )
    else:
        temp_count = int(round(len(df) * temp_size))
        rng = np.random.default_rng(random_state)
        temp_indices = rng.choice(df.index.to_numpy(), size=temp_count, replace=False).tolist()

    temp_df = df.loc[temp_indices].copy()
    train_df = df.drop(index=temp_indices).copy()

    if test_size == 0:
        return {
            "train": train_df,
            "val": temp_df,
            "test": pd.DataFrame(columns=df.columns),
        }
    if val_size == 0:
        return {
            "train": train_df,
            "val": pd.DataFrame(columns=df.columns),
            "test": temp_df,
        }

    val_ratio_in_temp = val_size / temp_size
    if stratify_enabled and target_col in temp_df.columns:
        val_indices = _stratified_sample_indices(
            df=temp_df,
            target_col=target_col,
            frac=val_ratio_in_temp,
            random_state=random_state + 1,
        )
    else:
        val_count = int(round(len(temp_df) * val_ratio_in_temp))
        rng = np.random.default_rng(random_state + 1)
        val_indices = rng.choice(temp_df.index.to_numpy(), size=val_count, replace=False).tolist()

    val_df = temp_df.loc[val_indices].copy()
    test_df = temp_df.drop(index=val_indices).copy()

    return {"train": train_df, "val": val_df, "test": test_df}


def _save_split_artifacts(splits: Dict[str, pd.DataFrame]) -> Dict[str, str]:
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    output_paths: Dict[str, str] = {}
    for split_name, split_df in splits.items():
        out_path = processed_dir / f"{split_name}.csv"
        split_df.to_csv(out_path, index=False)
        output_paths[split_name] = str(out_path)
    return output_paths


def _save_summary_table(summary: Dict[str, Any]) -> str:
    tables_dir = Path("outputs/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame([summary])
    out_path = tables_dir / "preprocessing_summary.csv"
    summary_df.to_csv(out_path, index=False)
    return str(out_path)


def run(config: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    if "raw_df" not in context:
        return {
            "step": "preprocessing",
            "status": "skipped",
            "message": "No raw dataframe found in context. Run ingestion first.",
        }

    df: pd.DataFrame = context["raw_df"].copy()
    schema_cfg = config.get("schema", {})
    cleaning_cfg = config.get("cleaning", {})
    split_cfg = config.get("split", {})

    target_col = schema_cfg.get("target", "income")
    missing_tokens = cleaning_cfg.get("missing_tokens", ["?", ""])
    drop_duplicates = bool(cleaning_cfg.get("drop_duplicates", True))

    initial_rows = int(df.shape[0])
    initial_missing = int(df.isna().sum().sum())

    # Standardize textual fields before missing-token replacement.
    df = _normalize_string_columns(df)

    # Convert dataset-specific missing tokens to actual NaN values.
    for token in missing_tokens:
        df = df.replace(token, pd.NA)

    missing_after_token_replacement = int(df.isna().sum().sum())

    duplicates_removed = 0
    if drop_duplicates:
        before = int(df.shape[0])
        df = df.drop_duplicates().reset_index(drop=True)
        duplicates_removed = before - int(df.shape[0])

    # Drop rows with missing target if any, keep feature-level missingness for later imputation strategy.
    missing_target_rows = 0
    if target_col in df.columns:
        before = int(df.shape[0])
        df = df[~df[target_col].isna()].copy()
        missing_target_rows = before - int(df.shape[0])

    df = _normalize_target(df, target_col=target_col)
    final_rows = int(df.shape[0])
    final_missing = int(df.isna().sum().sum())

    splits = _split_dataset(df, target_col=target_col, split_cfg=split_cfg)
    output_paths = _save_split_artifacts(splits)

    context["clean_df"] = df
    context["train_df"] = splits["train"]
    context["val_df"] = splits["val"]
    context["test_df"] = splits["test"]

    summary = {
        "initial_rows": initial_rows,
        "final_rows": final_rows,
        "rows_removed": initial_rows - final_rows,
        "duplicates_removed": duplicates_removed,
        "rows_removed_missing_target": missing_target_rows,
        "initial_missing_cells": initial_missing,
        "missing_cells_after_token_replacement": missing_after_token_replacement,
        "final_missing_cells": final_missing,
        "train_rows": int(splits["train"].shape[0]),
        "val_rows": int(splits["val"].shape[0]),
        "test_rows": int(splits["test"].shape[0]),
    }
    summary_path = _save_summary_table(summary)

    return {
        "step": "preprocessing",
        "status": "ok",
        "message": "Preprocessing completed and split artifacts saved.",
        "summary_path": summary_path,
        "split_paths": output_paths,
        "summary": summary,
    }
