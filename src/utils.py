from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def ensure_output_dirs(experiments_cfg: Dict[str, Any]) -> None:
    outputs = experiments_cfg.get("outputs", {})
    for key in ("base_dir", "figures_dir", "tables_dir", "models_dir", "reports_dir"):
        value = outputs.get(key)
        if value:
            Path(value).mkdir(parents=True, exist_ok=True)


def save_json(payload: Dict[str, Any], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

