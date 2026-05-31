"""Preprocessing pipeline for the KKBox churn prediction project.

This module handles three responsibilities:
  1. Splitting the feature frame into train / validation / test sets.
  2. Building a fitted sklearn ColumnTransformer (RobustScaler for numerics,
     OrdinalEncoder for categoricals).
  3. Persisting the splits and the fitted preprocessor for downstream use.

Design notes
------------
* The preprocessor is **fit on the training split only** — val and test are
  transformed with the training statistics to prevent data leakage.
* RobustScaler is preferred over StandardScaler because KKBox behavioral
  features (total_secs, trans_count, total_spend) contain extreme whale-user
  outliers that would distort mean/std estimates.
* Tree-based models (LightGBM, Random Forest) are scale-invariant; the
  scaler primarily benefits the Logistic Regression baseline.
* All split ratios and random seeds are read from config.yaml.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, RobustScaler

from src.utils.config import get_value

logger = logging.getLogger(__name__)

__all__ = [
    "ColumnGroups",
    "DataSplits",
    "identify_column_groups",
    "split_dataset",
    "build_preprocessor",
    "fit_preprocessor",
    "apply_preprocessor",
    "save_splits",
    "load_splits",
    "save_preprocessor",
    "load_preprocessor",
    "run_preprocessing",
]

# ---------------------------------------------------------------------------
# Constants — columns that should never be treated as features
# ---------------------------------------------------------------------------

# These columns carry no predictive signal or would cause leakage.
_NON_FEATURE_COLS: frozenset[str] = frozenset(
    {
        "msno",               # user identifier
        "is_churn",           # target label
        "analysis_reference_date",  # audit timestamp, not a feature
    }
)

# Raw datetime columns: too high-cardinality to encode directly; their
# information is already captured in derived features (member_age_days, etc.)
_DATETIME_DROP_COLS: frozenset[str] = frozenset(
    {
        "registration_init_time",
        "last_transaction_date",
        "last_expire_date",
        "last_log_date",
    }
)

# ---------------------------------------------------------------------------
# Data classes (no external dependencies — plain Python)
# ---------------------------------------------------------------------------
def _sanitise_for_sklearn(df: pd.DataFrame) -> pd.DataFrame:
    """Convert pandas extension dtypes to numpy-compatible types for sklearn.

    sklearn transformers (OrdinalEncoder, RobustScaler) do not understand
    pandas nullable dtypes (StringDtype, BooleanDtype) which use ``pd.NA``
    as the missing-value sentinel. This function converts those columns to
    plain numpy object arrays where missing values become ``np.nan``.

    Also handles ``object``-dtype columns that contain a mix of ``pd.NA``
    and numeric/string values — these arise when engineer.py combines a
    nullable Series with a float column via assignment.  sklearn's
    OrdinalEncoder calls ``sorted()`` on the unique values, which raises
    ``TypeError: boolean value of NA is ambiguous`` if ``pd.NA`` is present.

    Args:
        df: Input DataFrame potentially containing pandas extension dtypes.

    Returns:
        A copy of the DataFrame with all extension dtypes coerced to numpy-
        compatible equivalents (object for strings, float64 for booleans).
    """
    out = df.copy()
    for col in out.columns:
        dtype = out[col].dtype

        if isinstance(dtype, pd.StringDtype):
            # to_numpy(na_value=np.nan) replaces pd.NA with np.nan correctly.
            # astype(object) alone keeps pd.NA as pd.NA, which is wrong.
            out[col] = out[col].to_numpy(dtype=object, na_value=np.nan)

        elif isinstance(dtype, pd.CategoricalDtype):
            out[col] = out[col].astype(object)

        elif isinstance(dtype, pd.BooleanDtype):
            # BooleanDtype also uses pd.NA; convert to float so np.nan works.
            out[col] = out[col].to_numpy(dtype=float, na_value=np.nan)

        elif pd.api.types.is_object_dtype(dtype) and out[col].isna().any():
            # object columns can contain pd.NA mixed with float/string values.
            # to_numpy with na_value=np.nan replaces ALL NA sentinels
            # (pd.NA, None, np.nan) uniformly so sklearn sees a clean array.
            out[col] = out[col].to_numpy(dtype=object, na_value=np.nan)

    return out

class ColumnGroups:
    """Container for categorised column names.

    Attributes:
        numeric: Columns to pass through RobustScaler.
        categorical: Columns to pass through OrdinalEncoder.
        drop: Columns to exclude from the feature matrix entirely.
    """

    __slots__ = ("numeric", "categorical", "drop")

    def __init__(
        self,
        numeric: list[str],
        categorical: list[str],
        drop: list[str],
    ) -> None:
        self.numeric = numeric
        self.categorical = categorical
        self.drop = drop

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"ColumnGroups("
            f"numeric={len(self.numeric)}, "
            f"categorical={len(self.categorical)}, "
            f"drop={len(self.drop)})"
        )


class DataSplits:
    """Container for the three feature/label split pairs.

    Attributes:
        X_train: Training feature matrix.
        X_val: Validation feature matrix.
        X_test: Test feature matrix.
        y_train: Training labels.
        y_val: Validation labels.
        y_test: Test labels.
    """

    __slots__ = (
        "X_train", "X_val", "X_test",
        "y_train", "y_val", "y_test",
    )

    def __init__(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_val: pd.Series,
        y_test: pd.Series,
    ) -> None:
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test

    def log_shapes(self) -> None:
        """Emit split shape information to the logger."""
        logger.info(
            "Split shapes | "
            "train=(%d, %d) | val=(%d, %d) | test=(%d, %d)",
            *self.X_train.shape,
            *self.X_val.shape,
            *self.X_test.shape,
        )
        logger.info(
            "Churn rates  | "
            "train=%.4f | val=%.4f | test=%.4f",
            self.y_train.mean(),
            self.y_val.mean(),
            self.y_test.mean(),
        )


# ---------------------------------------------------------------------------
# Column identification
# ---------------------------------------------------------------------------


def identify_column_groups(
    frame: pd.DataFrame,
    *,
    extra_drop: list[str] | None = None,
) -> ColumnGroups:
    """Categorise all columns into numeric, categorical, and drop groups.

    The classification logic:
    - Drop: ID, target, audit, raw datetime columns (listed in constants).
    - Categorical: object, string, or CategoricalDtype columns that remain.
    - Numeric: all other columns.

    Args:
        frame: The engineered feature DataFrame.
        extra_drop: Additional column names to force into the drop group
                    (e.g. raw columns superseded by derived ones).

    Returns:
        A ``ColumnGroups`` instance with three sorted lists.
    """
    forced_drop = _NON_FEATURE_COLS | _DATETIME_DROP_COLS
    if extra_drop:
        forced_drop = forced_drop | set(extra_drop)

    numeric: list[str] = []
    categorical: list[str] = []
    drop: list[str] = []

    for col in frame.columns:
        if col in forced_drop:
            drop.append(col)
            continue

        dtype = frame[col].dtype
        if (
            pd.api.types.is_object_dtype(dtype)
            or pd.api.types.is_string_dtype(dtype)
            or isinstance(dtype, pd.CategoricalDtype)
        ):
            categorical.append(col)
        else:
            numeric.append(col)

    groups = ColumnGroups(
        numeric=sorted(numeric),
        categorical=sorted(categorical),
        drop=sorted(drop),
    )
    logger.info(
        "Column groups identified | numeric=%d, categorical=%d, drop=%d",
        len(numeric),
        len(categorical),
        len(drop),
    )
    return groups


# ---------------------------------------------------------------------------
# Train / validation / test split
# ---------------------------------------------------------------------------


def split_dataset(
    frame: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the feature frame into train, validation, and test sets.

    Splitting strategy
    ------------------
    A **stratified random split** is used (not a temporal split) because
    ``train.csv`` is already a user-level snapshot for a single observation
    period — temporal leakage at the row level is not a concern, and all
    temporal signals are encoded in the derived features. Stratification
    preserves the minority churn class ratio across all three splits.

    Two-pass approach:
      1. Split out the test set first.
      2. Split the remainder into train and validation.

    Args:
        frame: Engineered feature DataFrame (must contain ``is_churn``).
        config: Parsed project configuration.

    Returns:
        Tuple of (train_df, val_df, test_df) DataFrames.

    Raises:
        KeyError: If ``is_churn`` is absent from ``frame``.
        ValueError: If split sizes are invalid (sum ≥ 1.0).
    """
    if "is_churn" not in frame.columns:
        raise KeyError("Feature frame must contain 'is_churn' column.")

    random_state: int = int(
        get_value(config, "project", "random_state", default=42)
    )
    val_size: float = float(
        get_value(config, "split", "validation_size", default=0.15)
    )
    test_size: float = float(
        get_value(config, "split", "test_size", default=0.15)
    )
    stratify: bool = bool(
        get_value(config, "split", "stratify", default=True)
    )

    if val_size + test_size >= 1.0:
        raise ValueError(
            f"val_size ({val_size}) + test_size ({test_size}) must be < 1.0"
        )

    stratify_col = frame["is_churn"] if stratify else None

    # Pass 1: carve out the test set from the full frame.
    train_val_df, test_df = train_test_split(
        frame,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_col,
    )

    # Pass 2: split remaining data into train and validation.
    # Recalculate val proportion relative to the train+val pool size.
    val_proportion = val_size / (1.0 - test_size)
    stratify_col_tv = train_val_df["is_churn"] if stratify else None

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_proportion,
        random_state=random_state,
        stratify=stratify_col_tv,
    )

    logger.info(
        "Dataset split | total=%d -> train=%d (%.1f%%), "
        "val=%d (%.1f%%), test=%d (%.1f%%)",
        len(frame),
        len(train_df), 100 * len(train_df) / len(frame),
        len(val_df),   100 * len(val_df)   / len(frame),
        len(test_df),  100 * len(test_df)  / len(frame),
    )

    churn_rates = {
        "train": train_df["is_churn"].mean(),
        "val":   val_df["is_churn"].mean(),
        "test":  test_df["is_churn"].mean(),
    }
    logger.info("Churn rates after split: %s", churn_rates)

    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# Preprocessor: build and fit
# ---------------------------------------------------------------------------


from sklearn.impute import SimpleImputer   # ← thêm import này ở đầu file

def build_preprocessor(groups: ColumnGroups) -> ColumnTransformer:
    """Construct an unfitted sklearn ColumnTransformer.

    Transformer choices
    -------------------
    * **SimpleImputer(median)** (numeric): Imputes NaN from derived ratio
      features (e.g. spend_per_transaction where denominator=0). Median is
      used instead of mean or 0 because derived ratios are right-skewed and
      0 is already the fill for users with no activity in the merger stage.
    * **RobustScaler** (numeric): Uses median and IQR — robust to whale-user
      outliers common in streaming behavioral features.
    * **OrdinalEncoder** (categorical): Assigns stable integer codes for tree
      models. Unknown/missing categories map to -1 via encoded_missing_value.

    Args:
        groups: Column categorisation from ``identify_column_groups()``.

    Returns:
        An unfitted ``ColumnTransformer``.
    """
    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",  # fit on train, apply to val/test
                    # Median is safer than 0 for ratio features: a user with
                    # no transactions genuinely has unknown spend-per-tx, not 0.
                ),
            ),
            (
                "scaler",
                RobustScaler(
                    with_centering=True,
                    with_scaling=True,
                    quantile_range=(25.0, 75.0),
                ),
            ),
        ],
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "encoder",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-1,  # np.nan → -1, handled natively
                ),
            ),
        ],
    )

    transformers: list[tuple[str, Any, list[str]]] = []
    if groups.numeric:
        transformers.append(("numeric", numeric_pipeline, groups.numeric))
    if groups.categorical:
        transformers.append(("categorical", categorical_pipeline, groups.categorical))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )

    logger.info(
        "Preprocessor built | "
        "SimpleImputer(median)+RobustScaler on %d numeric cols, "
        "OrdinalEncoder on %d categorical cols.",
        len(groups.numeric),
        len(groups.categorical),
    )
    return preprocessor

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import FunctionTransformer


def _to_str_array(X) -> np.ndarray:
    """Coerce a 2-D object array/DataFrame to uniform string + np.nan.

    ``identify_column_groups`` classifies any object-dtype column as
    categorical.  However, engineer.py can produce columns whose object
    array contains a mix of str values **and** actual float values (not just
    np.nan).  When that data enters ``OneHotEncoder``, sklearn calls
    ``sorted()`` on the unique values which raises::

        TypeError: '<' not supported between instances of 'str' and 'float'

    This transformer runs before ``SimpleImputer`` to ensure every
    non-null entry is a Python ``str``.  Null entries (np.nan, None,
    pd.NA) are preserved as ``np.nan`` so ``SimpleImputer`` can later
    fill them with ``"__missing__"``.

    Note: ``FunctionTransformer(validate=False)`` passes the raw input
    (which may be a pandas DataFrame, not a numpy array) directly.  We
    normalise to a numpy object array first to avoid 2-D boolean-index
    assignment errors that occur on DataFrames.
    """
    # Normalise to numpy object array regardless of input type
    if hasattr(X, "to_numpy"):
        arr = X.to_numpy(dtype=object, na_value=np.nan)
    else:
        arr = np.asarray(X, dtype=object)

    mask = pd.isnull(arr)          # 2-D boolean: True where value is NA-like

    # Step 1: convert everything to str (np.nan → "nan" as a side-effect)
    str_arr = np.vectorize(str)(arr)

    # Step 2: build output and restore NA positions
    out = np.empty(arr.shape, dtype=object)
    out[:] = str_arr               # all positions get their str representation
    out[mask] = np.nan             # scalar assignment to 2-D mask — always safe

    return out


def build_linear_preprocessor(
    groups: ColumnGroups,
    X_ref: "pd.DataFrame | None" = None,
    max_ohe_categories: int = 50,
) -> ColumnTransformer:
    """Construct a preprocessor suited for linear models (Logistic Regression).

    Linear models require proper categorical encoding — OrdinalEncoder imposes
    a false numerical ordering on categories (e.g. 0 < 1 < 2) which causes
    logistic regression to learn inverted or garbage coefficients. OneHotEncoder
    creates independent binary columns, preserving the true categorical structure.

    Args:
        groups: Column categorisation from ``identify_column_groups()``.
        X_ref: Optional reference DataFrame used to measure cardinality of
            categorical columns before building the pipeline.  Any column
            with more than ``max_ohe_categories`` unique non-null values is
            silently dropped from the categorical pipeline.  This prevents
            numeric columns accidentally stored as ``object`` dtype (e.g.
            continuous floats) from exploding the OHE output into millions
            of columns.
        max_ohe_categories: Upper bound on unique categories allowed for
            OneHotEncoding.  Columns exceeding this limit are excluded.
            Default: 50.

    Returns:
        An unfitted ``ColumnTransformer`` for linear models.
    """
    # ── Cardinality guard ─────────────────────────────────────────────────────
    safe_categorical = groups.categorical
    if X_ref is not None and groups.categorical:
        safe_categorical = []
        dropped_high_card = []
        for col in groups.categorical:
            if col not in X_ref.columns:
                continue
            n_unique = int(X_ref[col].nunique(dropna=True))
            if n_unique <= max_ohe_categories:
                safe_categorical.append(col)
            else:
                dropped_high_card.append((col, n_unique))
        if dropped_high_card:
            logger.warning(
                "Dropping %d high-cardinality columns from linear categorical "
                "pipeline (max_ohe_categories=%d). These are likely numeric "
                "columns stored as object dtype: %s",
                len(dropped_high_card),
                max_ohe_categories,
                [(c, n) for c, n in dropped_high_card],
            )

    # ── Pipelines ─────────────────────────────────────────────────────────────
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  RobustScaler(with_centering=True, with_scaling=True)),
            # ── Clip extreme values after scaling ────────────────────────────
            # Some features have post-RobustScaler values of ±3e11 (e.g. raw
            # date integers or un-log-scaled counts with near-zero IQR).
            # These values overwhelm L2 regularization: the solver is forced to
            # shrink ALL coefficients (including the intercept) to ~0 to avoid
            # numerical overflow, collapsing all predictions to sigmoid(0)=0.5.
            #
            # Clipping to [-10, 10] IQR-units is standard practice for LR:
            #   - Normal features (well-scaled): values in [-2, 2], not clipped.
            #   - Moderate outliers (up to 10 IQR-units): kept as-is.
            #   - Extreme outliers (>10 IQR-units): capped at ±10, treated as
            #     a "this user is an extreme outlier" soft indicator.
            #
            # Does NOT use get_feature_names_out(); compatible with all sklearn.
            ("clip", FunctionTransformer(
                lambda X: np.clip(X, -10.0, 10.0),
                validate=False,
            )),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            # Step 1: coerce all non-null values to str.
            # Needed because engineer.py can produce object columns with mixed
            # float + str entries that would crash OneHotEncoder's sort().
            (
                "to_str",
                FunctionTransformer(_to_str_array, validate=False),
            ),
            # Step 2: fill remaining NaN with a dedicated category string.
            (
                "imputer",
                SimpleImputer(strategy="constant", fill_value="__missing__"),
            ),
            # Step 3: one-hot encode.
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",   # unknown at inference -> all-zero row
                    sparse_output=False,       # return dense array
                    drop="if_binary",          # drop one column for binary features
                                               # to avoid perfect multicollinearity
                ),
            ),
        ]
    )

    transformers: list[tuple[str, Any, list[str]]] = []
    if groups.numeric:
        transformers.append(("numeric", numeric_pipeline, groups.numeric))
    if safe_categorical:
        transformers.append(("categorical", categorical_pipeline, safe_categorical))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )

    logger.info(
        "Linear preprocessor built | "
        "SimpleImputer+RobustScaler on %d numeric, "
        "to_str+SimpleImputer+OHE on %d categorical (max_ohe=%d, dropped %d high-card).",
        len(groups.numeric),
        len(safe_categorical),
        max_ohe_categories,
        len(groups.categorical) - len(safe_categorical),
    )
    return preprocessor

def fit_preprocessor(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
) -> ColumnTransformer:
    """Fit the preprocessor on the training set only.

    Fitting on train-only prevents data leakage: the scaling statistics
    (median, IQR) and encoder vocabularies are derived solely from
    training data, then applied identically to val and test.

    Args:
        preprocessor: Unfitted ColumnTransformer from ``build_preprocessor()``.
        X_train: Training feature matrix (excluding the target column).

    Returns:
        The same transformer, now fitted.
    """
    logger.info(
        "Fitting preprocessor on X_train with shape %s ...", X_train.shape
    )
    # Sanitise pandas extension dtypes (StringDtype uses pd.NA, not np.nan)
    # before passing to sklearn which only understands np.nan.
    X_train_clean = _sanitise_for_sklearn(X_train)
    preprocessor.fit(X_train_clean)
    logger.info("Preprocessor fitting complete.")
    return preprocessor


def apply_preprocessor(
    preprocessor: ColumnTransformer,
    X: pd.DataFrame,
    split_name: str = "unknown",
) -> pd.DataFrame:
    """Apply a fitted preprocessor to a feature matrix and return a DataFrame.

    Args:
        preprocessor: Fitted ColumnTransformer.
        X: Feature matrix to transform.
        split_name: Label used in log messages (e.g. ``"train"``, ``"val"``).

    Returns:
        Transformed DataFrame with original column names preserved.
    """
    # Sanitise extension dtypes before sklearn sees the data.
    X_clean = _sanitise_for_sklearn(X)

    transformed_array = preprocessor.transform(X_clean)

    try:
        feature_names = preprocessor.get_feature_names_out()
    except AttributeError:
        feature_names = [
            f"feature_{i}" for i in range(transformed_array.shape[1])
        ]

    result = pd.DataFrame(
        transformed_array,
        columns=feature_names,
        index=X.index,
    )
    logger.info(
        "Transformed %s split | input shape=%s -> output shape=%s",
        split_name,
        X.shape,
        result.shape,
    )
    return result


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------


def save_splits(splits: DataSplits, processed_dir: Path) -> None:
    """Persist all six split DataFrames/Series as Parquet files.

    Files written:
        ``X_train.parquet``, ``X_val.parquet``, ``X_test.parquet``
        ``y_train.parquet``, ``y_val.parquet``, ``y_test.parquet``

    Args:
        splits: Fitted DataSplits container.
        processed_dir: Destination directory (created if absent).
    """
    processed_dir.mkdir(parents=True, exist_ok=True)

    artifacts: dict[str, pd.DataFrame | pd.Series] = {
        "X_train": splits.X_train,
        "X_val":   splits.X_val,
        "X_test":  splits.X_test,
        "y_train": splits.y_train,
        "y_val":   splits.y_val,
        "y_test":  splits.y_test,
    }

    for name, data in artifacts.items():
        path = processed_dir / f"{name}.parquet"
        if isinstance(data, pd.Series):
            data.to_frame().to_parquet(path, index=True)
        else:
            data.to_parquet(path, index=True)
        logger.info("Saved %s -> %s  (shape=%s)", name, path, data.shape)


def load_splits(processed_dir: Path) -> DataSplits:
    """Reload the six split files written by ``save_splits()``.

    Args:
        processed_dir: Directory containing the Parquet split files.

    Returns:
        A populated ``DataSplits`` instance.

    Raises:
        FileNotFoundError: If any expected Parquet file is missing.
    """
    names = ["X_train", "X_val", "X_test", "y_train", "y_val", "y_test"]
    loaded: dict[str, pd.DataFrame | pd.Series] = {}

    for name in names:
        path = processed_dir / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"Expected split file not found: {path}"
            )
        df = pd.read_parquet(path)
        # Labels were saved as single-column DataFrames; squeeze back to Series.
        if name.startswith("y_"):
            loaded[name] = df.iloc[:, 0].rename("is_churn")
        else:
            loaded[name] = df
        logger.info("Loaded %s from %s  (shape=%s)", name, path, df.shape)

    return DataSplits(
        X_train=loaded["X_train"],
        X_val=loaded["X_val"],
        X_test=loaded["X_test"],
        y_train=loaded["y_train"],
        y_val=loaded["y_val"],
        y_test=loaded["y_test"],
    )


def save_preprocessor(preprocessor: ColumnTransformer, path: Path) -> None:
    """Serialise the fitted preprocessor as a pickle file.

    Pickle is used (rather than joblib) to avoid joblib version sensitivity
    when loading artifacts in different environments.

    Args:
        preprocessor: Fitted ColumnTransformer to save.
        path: Destination ``.pkl`` file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(preprocessor, handle, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Preprocessor saved -> %s", path)


def load_preprocessor(path: Path) -> ColumnTransformer:
    """Load a previously serialised preprocessor.

    Args:
        path: Path to the ``.pkl`` file written by ``save_preprocessor()``.

    Returns:
        The deserialized, fitted ColumnTransformer.

    Raises:
        FileNotFoundError: If the pickle file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Preprocessor file not found: {path}")
    with path.open("rb") as handle:
        preprocessor: ColumnTransformer = pickle.load(handle)
    logger.info("Preprocessor loaded from %s", path)
    return preprocessor


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_preprocessing(
    config: Mapping[str, Any],
    project_root: Path,
) -> DataSplits:
    """Execute the full preprocessing stage end-to-end.

    Steps
    -----
    1. Load the engineered feature frame from ``data/processed/``.
    2. Split into train / val / test (stratified).
    3. Identify column groups (numeric vs categorical vs drop).
    4. Build and fit the ColumnTransformer on the training split only.
    5. Transform all three splits.
    6. Save splits and the fitted preprocessor to ``data/processed/``.

    Args:
        config: Parsed project configuration (from ``config.yaml``).
        project_root: Absolute path to the project root directory.

    Returns:
        A ``DataSplits`` instance containing all transformed splits.

    Raises:
        FileNotFoundError: If the feature frame Parquet file is absent.
    """
    target_col: str = str(
        get_value(config, "project", "target_col", default="is_churn")
    )

    # -------------------------------------------------------------------------
    # 1. Load feature frame
    # -------------------------------------------------------------------------
    processed_dir = project_root / "data" / "processed"
    output_file = get_value(
        config, "feature_engineering", "output_file",
        default="feature_frame.parquet",
    )
    feature_frame_path = processed_dir / str(output_file)

    if not feature_frame_path.exists():
        raise FileNotFoundError(
            f"Feature frame not found at {feature_frame_path}. "
            "Run the feature engineering stage first."
        )

    logger.info("Loading feature frame from %s ...", feature_frame_path)
    frame = pd.read_parquet(feature_frame_path)
    logger.info("Feature frame loaded | shape=%s", frame.shape)

    # ---------------------------------------------------------------------
    # Optional CV target encoding for specified categorical columns
    # ---------------------------------------------------------------------
    te_cols = list(
        get_value(config, "feature_engineering", "target_encode_columns", default=[])
    )
    if te_cols:
        logger.info("Applying CV target encoding to columns: %s", te_cols)
        # We will perform encoding in-place on a copy of the frame to avoid
        # mutating the upstream artifact.
        frame = frame.copy()
        # We'll compute encoders using the split defined below, so postpone
        # heavy per-split mapping until after split_dataset() creates train/val/test.

    # -------------------------------------------------------------------------
    # 2. Split into train / val / test
    # -------------------------------------------------------------------------
    train_df, val_df, test_df = split_dataset(frame, config)

    # If CV target-encoding requested, compute out-of-fold encodings using
    # training folds and apply to val/test. Encoded columns are added with
    # suffix `_te` and raw columns are left for possible dropping later.
    if te_cols:
        def _cv_encode_column(col: str, n_splits: int = 5, smoothing: float = 10.0):
            # Prepare arrays
            X_col = train_df[col].astype(object)
            y = train_df[target_col]
            oof = pd.Series(index=train_df.index, dtype=float)
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=int(get_value(config, 'project', 'random_state', default=42)))
            global_mean = float(y.mean())
            for train_idx, val_idx in kf.split(X_col):
                tr_vals = X_col.iloc[train_idx]
                tr_y = y.iloc[train_idx]
                stats = tr_y.groupby(tr_vals).agg(['mean','count'])
                means = stats['mean']
                counts = stats['count']
                smooth = (counts * means + smoothing * global_mean) / (counts + smoothing)
                mapped = X_col.iloc[val_idx].map(smooth)
                oof.iloc[val_idx] = mapped
            # Fill any remaining NaN with global mean
            oof = oof.fillna(global_mean)
            # Build mapping from full training set for val/test application
            full_stats = y.groupby(train_df[col]).agg(['mean','count'])
            full_means = full_stats['mean']
            full_counts = full_stats['count']
            full_smooth = (full_counts * full_means + smoothing * global_mean) / (full_counts + smoothing)
            # Apply to val and test
            val_mapped = val_df[col].map(full_smooth).fillna(global_mean)
            test_mapped = test_df[col].map(full_smooth).fillna(global_mean)
            return oof.rename(f"{col}_te"), val_mapped.rename(f"{col}_te"), test_mapped.rename(f"{col}_te")

        te_n_splits = int(get_value(config, 'feature_engineering', 'target_encode_splits', default=5))
        te_smoothing = float(get_value(config, 'feature_engineering', 'target_encode_smoothing', default=10.0))
        for col in te_cols:
            if col not in train_df.columns:
                logger.warning("Requested TE column '%s' not found in frame; skipping.", col)
                continue
            oof_col, val_col, test_col = _cv_encode_column(col, n_splits=te_n_splits, smoothing=te_smoothing)
            train_df[oof_col.name] = oof_col
            val_df[val_col.name] = val_col
            test_df[test_col.name] = test_col
        logger.info("CV target-encoding complete. Encoded columns added with suffix _te.")

    # Separate features from labels. Keep msno in X for traceability; it is
    # listed in _NON_FEATURE_COLS so ColumnTransformer will drop it.
    X_train_raw = train_df.drop(columns=[target_col])
    X_val_raw   = val_df.drop(columns=[target_col])
    X_test_raw  = test_df.drop(columns=[target_col])

    y_train = train_df[target_col].astype("int8")
    y_val   = val_df[target_col].astype("int8")
    y_test  = test_df[target_col].astype("int8")

    # -------------------------------------------------------------------------
    # 3. Identify column groups
    # -------------------------------------------------------------------------
    # Raw columns superseded by cleaner derived versions are force-dropped so
    # the model does not see duplicate information.
    extra_drop = [
        c for c in ["bd", "gender", "city", "registered_via"]
        if c in X_train_raw.columns
    ]
    groups = identify_column_groups(X_train_raw, extra_drop=extra_drop)

    # -------------------------------------------------------------------------
    # 4. Build + fit the preprocessor (train only)
    # -------------------------------------------------------------------------
    preprocessor = build_preprocessor(groups)
    fit_preprocessor(preprocessor, X_train_raw)

    # -------------------------------------------------------------------------
    # 5. Transform all splits
    # -------------------------------------------------------------------------
    X_train = apply_preprocessor(preprocessor, X_train_raw, "train")
    X_val   = apply_preprocessor(preprocessor, X_val_raw,   "val")
    X_test  = apply_preprocessor(preprocessor, X_test_raw,  "test")

    splits = DataSplits(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
    )
    splits.log_shapes()

    # -------------------------------------------------------------------------
    # 6. Persist splits and preprocessor
    # -------------------------------------------------------------------------
    save_splits(splits, processed_dir)

    preprocessor_path = project_root / "models" / "preprocessor.pkl"
    save_preprocessor(preprocessor, preprocessor_path)

    logger.info("Preprocessing stage complete.")
    return splits
