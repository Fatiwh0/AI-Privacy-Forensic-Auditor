from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


RENAME_MAP = {
    "education-num": "education_num",
    "marital-status": "marital_status",
    "capital-gain": "capital_gain",
    "capital-loss": "capital_loss",
    "hours-per-week": "hours_per_week",
    "native-country": "native_country",
    "class": "income",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Adult dataset from OpenML.")
    parser.add_argument(
        "--output",
        default="data/raw/adult.csv",
        help="Output CSV path expected by the pipeline config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from sklearn.datasets import fetch_openml
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: scikit-learn. Install requirements first: "
            "pip install -r requirements.txt"
        ) from exc

    # OpenML provides a clean dataframe with features and target.
    bunch = fetch_openml(name="adult", version=2, as_frame=True)
    df = pd.concat([bunch.data, bunch.target.rename("class")], axis=1)
    df = df.rename(columns=RENAME_MAP)

    # Normalize income labels so downstream code has one convention.
    if "income" in df.columns:
        df["income"] = (
            df["income"]
            .astype(str)
            .str.strip()
            .str.replace(".", "", regex=False)
        )

    df.to_csv(output_path, index=False)
    print(f"Saved: {output_path} | shape={df.shape}")


if __name__ == "__main__":
    main()
