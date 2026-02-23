from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd


def run(config: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    dataset_cfg = config.get("dataset", {})
    local_path = Path(dataset_cfg.get("local_path", "data/raw/adult.csv"))

    result: Dict[str, Any] = {
        "step": "ingestion",
        "dataset_name": dataset_cfg.get("name", "unknown"),
        "local_path": str(local_path),
        "status": "pending",
        "message": "Ingestion placeholder.",
    }

    if not local_path.exists():
        result["status"] = "skipped"
        result["message"] = (
            f"Dataset file not found at {local_path}. "
            "Add the CSV file to continue."
        )
        return result

    sep = dataset_cfg.get("separator", ",")
    has_header = dataset_cfg.get("has_header", True)
    header = 0 if has_header else None

    df = pd.read_csv(local_path, sep=sep, header=header)
    context["raw_df"] = df

    result["status"] = "ok"
    result["rows"] = int(df.shape[0])
    result["columns"] = int(df.shape[1])
    result["message"] = "Dataset loaded into memory."
    return result

