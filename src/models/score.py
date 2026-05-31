"""Score new users with the saved KKBox churn champion model."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.preprocess import apply_preprocessor, load_preprocessor
from src.utils.config import get_path, get_value, load_config


def _load_pickle(path: Path) -> Any:
    """Load a pickle artifact without importing the training stack."""

    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    with path.open("rb") as handle:
        return pickle.load(handle)


def _read_frame(path: Path) -> pd.DataFrame:
    """Read a scoring input file from Parquet or CSV."""

    if not path.exists():
        raise FileNotFoundError(f"Scoring input not found: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError("Scoring input must be a .parquet or .csv file.")


def _predict_churn_proba(model: Any, X: pd.DataFrame) -> pd.Series:
    """Return positive-class probabilities as a Series."""

    if not hasattr(model, "predict_proba"):
        raise TypeError("Champion model must expose predict_proba().")
    return pd.Series(model.predict_proba(X)[:, 1], index=X.index, name="churn_probability")


def score_frame(input_path: Path, output_path: Path, config_path: Path) -> pd.DataFrame:
    """Load an engineered feature frame, transform it, and write predictions."""

    config = load_config(config_path)
    models_dir = get_path(config, "models_dir", base_dir=PROJECT_ROOT)
    target_col = str(get_value(config, "project", "target_col"))
    id_col = str(get_value(config, "project", "id_col"))
    threshold = float(get_value(config, "modeling", "decision_threshold"))

    model = _load_pickle(models_dir / str(get_value(config, "artifacts", "champion_model_file")))
    preprocessor = load_preprocessor(models_dir / str(get_value(config, "artifacts", "preprocessor_file")))

    frame = _read_frame(input_path)
    output = pd.DataFrame(index=frame.index)
    if id_col in frame.columns:
        output[id_col] = frame[id_col].astype("string")

    features = frame.drop(columns=[target_col], errors="ignore")
    transformed = apply_preprocessor(preprocessor, features, split_name="score")
    scores = _predict_churn_proba(model, transformed)

    output["churn_probability"] = scores
    output["predicted_churn"] = (scores >= threshold).astype("int8")
    output["threshold"] = threshold

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".parquet":
        output.to_parquet(output_path, index=False)
    elif output_path.suffix.lower() == ".csv":
        output.to_csv(output_path, index=False)
    else:
        raise ValueError("Scoring output must be a .parquet or .csv file.")

    print(f"Saved predictions to: {output_path}")
    return output


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Score an engineered KKBox feature frame.")
    parser.add_argument("--input", required=True, type=Path, help="Input feature frame (.parquet or .csv).")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Prediction output path (.csv or .parquet).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "config.yaml",
        help="Project config path.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config_for_default = load_config(args.config)
    default_reports_dir = get_path(config_for_default, "reports_dir", base_dir=PROJECT_ROOT)
    default_output = default_reports_dir / str(get_value(config_for_default, "artifacts", "predictions_file"))
    score_frame(args.input, args.output or default_output, args.config)
