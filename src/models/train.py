"""Model training pipeline for the KKBox churn prediction project.

Workflow
--------
1. Load preprocessed train / validation splits from ``data/processed/``.
2. Load the raw feature frame and build a OneHot linear preprocessor for LR.
3. Train all baseline candidate models under MLflow tracking:
   - Logistic Regression  (OneHot + RobustScaler)
   - Random Forest        (OrdinalEncoded, tree-native)
   - XGBoost              (OrdinalEncoded, scale_pos_weight)
   - LightGBM             (OrdinalEncoded, scale_pos_weight)
4. Train advanced XGBoost candidates as official contenders:
    - Optuna-tuned XGBoost
    - Top-K-features XGBoost (based on baseline XGB gain ranking)
5. Select the champion by highest validation AUC-PR across all contenders.
6. Persist the champion model and a full comparison table.

Design decisions
----------------
* Hold-out validation (not cross-validation): ~1M users gives stable
  metric estimates on a fixed val split without the 5× compute of CV.
* Class imbalance:
  - ``class_weight='balanced'`` for sklearn models (LR, RF).
  - ``scale_pos_weight = n_neg / n_pos`` for XGBoost and LightGBM.
* Champion metric: AUC-PR is preferred over AUC-ROC for imbalanced
  churn data because it is more sensitive to minority-class performance.
* LR uses a separate OneHot+RobustScaler preprocessor built from the raw
  feature frame; tree models use the OrdinalEncoded splits from preprocessing.
* XGBoost and LightGBM use early stopping on validation average_precision
  to prevent overfitting and avoid manual ``n_estimators`` tuning.
* Every run is logged to MLflow for full experiment reproducibility.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.features.preprocess import (
    _sanitise_for_sklearn,
    build_linear_preprocessor,
    identify_column_groups,
    split_dataset,
)
from src.models.booster_model import BoosterModel
from src.models.feature_subset_model import FeatureSubsetModel
from src.utils.config import get_value

logger = logging.getLogger(__name__)

__all__ = [
    "MetricsBundle",
    "ModelResult",
    "train_logistic_regression",
    "train_random_forest",
    "train_xgboost",
    "train_lightgbm",
    "select_champion",
    "save_model",
    "load_model",
    "save_comparison_table",
    "run_training",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class MetricsBundle:
    """All evaluation metrics for one model on one split.

    Attributes:
        auc_roc: Area under the ROC curve.
        auc_pr: Area under the Precision-Recall curve (primary metric).
        f1: F1 score at the configured decision threshold.
        precision: Precision at the configured decision threshold.
        recall: Recall at the configured decision threshold.
        lift_at_top10: Churn lift in the top-decile of predicted scores
                       versus random targeting.
    """

    auc_roc: float
    auc_pr: float
    f1: float
    precision: float
    recall: float
    lift_at_top10: float

    def to_dict(self, prefix: str = "") -> dict[str, float]:
        """Serialise to a flat dict, optionally prefixed (e.g. ``'val_'``).

        Args:
            prefix: String prepended to each key (e.g. ``"val_"``).

        Returns:
            Flat dictionary of metric names to float values.
        """
        return {
            f"{prefix}auc_roc":       round(self.auc_roc, 6),
            f"{prefix}auc_pr":        round(self.auc_pr, 6),
            f"{prefix}f1":            round(self.f1, 6),
            f"{prefix}precision":     round(self.precision, 6),
            f"{prefix}recall":        round(self.recall, 6),
            f"{prefix}lift_at_top10": round(self.lift_at_top10, 4),
        }


@dataclass
class ModelResult:
    """Container for a trained model and its evaluation outputs.

    Attributes:
        name: Human-readable model identifier (e.g. ``"xgboost"``).
        model: The fitted scikit-learn-compatible estimator.
        params: Hyper-parameters passed to the model constructor.
        train_metrics: MetricsBundle evaluated on the training split.
        val_metrics: MetricsBundle evaluated on the validation split.
        mlflow_run_id: MLflow run ID, or ``None`` if tracking is disabled.
        best_iteration: Best early-stopping iteration (0 for other models).
    """

    name: str
    model: Any
    params: dict[str, Any]
    train_metrics: MetricsBundle
    val_metrics: MetricsBundle
    mlflow_run_id: str | None = None
    best_iteration: int = 0
    decision_threshold: float = 0.5


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _lift_at_top_k(
    y_true: pd.Series | np.ndarray,
    y_score: np.ndarray,
    k: float = 0.10,
) -> float:
    """Compute churn lift in the top-k fraction of predicted scores.

    Lift = precision_in_top_k / baseline_churn_rate.

    Args:
        y_true: True binary labels.
        y_score: Predicted churn probabilities.
        k: Fraction of users to target (default 10%).

    Returns:
        Lift scalar, or 0.0 if the baseline churn rate is zero.
    """
    n_top = max(1, int(len(y_true) * k))
    top_indices = np.argsort(y_score)[::-1][:n_top]
    precision_top_k = float(np.asarray(y_true)[top_indices].mean())
    baseline = float(np.asarray(y_true).mean())
    if baseline == 0.0:
        return 0.0
    return precision_top_k / baseline


def _compute_metrics(
    y_true: pd.Series | np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> MetricsBundle:
    """Compute the full metric suite for one split.

    Args:
        y_true: True binary labels.
        y_score: Predicted churn probabilities (positive class).
        threshold: Decision threshold for precision/recall/F1.

    Returns:
        Populated ``MetricsBundle``.
    """
    y_pred = (y_score >= threshold).astype(int)
    return MetricsBundle(
        auc_roc=float(roc_auc_score(y_true, y_score)),
        auc_pr=float(average_precision_score(y_true, y_score)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        lift_at_top10=_lift_at_top_k(y_true, y_score, k=0.10),
    )


def _tune_decision_threshold(
    y_true: pd.Series | np.ndarray,
    y_score: np.ndarray,
    *,
    metric: str = "f2",
    grid_step: float = 0.01,
) -> float:
    """Pick a classification threshold on validation scores.

    The default metric is F2 so the operating point is biased toward recall,
    which is more appropriate for churn targeting than the default 0.5 cutoff.
    """

    y_true_arr = np.asarray(y_true).ravel().astype(int)
    thresholds = np.arange(grid_step, 1.0, grid_step)
    if thresholds.size == 0:
        return 0.5

    best_threshold = 0.5
    best_score = float("-inf")

    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype(int)
        if metric == "f1":
            score = float(f1_score(y_true_arr, y_pred, zero_division=0))
        elif metric == "f2":
            score = float(fbeta_score(y_true_arr, y_pred, beta=2.0, zero_division=0))
        else:
            raise ValueError(f"Unsupported decision threshold metric: {metric}")

        if score > best_score:
            best_score = score
            best_threshold = float(threshold)

    return best_threshold


def _scale_pos_weight(y_train: pd.Series | np.ndarray) -> float:
    """Compute scale_pos_weight from training labels (n_neg / n_pos).

    Args:
        y_train: Training target labels.

    Returns:
        Float ratio of negative to positive samples.
    """
    arr = np.asarray(y_train)
    n_pos = int((arr == 1).sum())
    n_neg = int((arr == 0).sum())
    if n_pos == 0:
        raise ValueError("Training labels contain no positive (churn) samples.")
    ratio = n_neg / n_pos
    logger.info(
        "Class counts | neg=%d, pos=%d -> scale_pos_weight=%.2f",
        n_neg, n_pos, ratio,
    )
    return ratio


# _sanitise_for_sklearn is imported from src.features.preprocess.
# It handles StringDtype, BooleanDtype, CategoricalDtype, and object columns
# that contain pd.NA or mixed float+str values — all cases that cause
# sklearn encoders to crash.


# ---------------------------------------------------------------------------
# MLflow helper
# ---------------------------------------------------------------------------


def _log_to_mlflow(
    result: ModelResult,
    experiment_name: str,
    tracking_uri: str | None,
) -> str | None:
    """Log a ModelResult to MLflow and return the run ID.

    Args:
        result: Populated ModelResult to log.
        experiment_name: MLflow experiment name from config.
        tracking_uri: MLflow tracking server URI (``None`` = local).

    Returns:
        The MLflow run ID string, or ``None`` on failure.
    """
    try:
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name=result.name) as run:
            mlflow.log_params(result.params)
            mlflow.log_metrics(result.train_metrics.to_dict(prefix="train_"))
            mlflow.log_metrics(result.val_metrics.to_dict(prefix="val_"))

            if result.best_iteration > 0:
                mlflow.log_param("best_iteration", result.best_iteration)
            mlflow.log_param("decision_threshold", result.decision_threshold)

            if isinstance(result.model, lgb.LGBMClassifier):
                mlflow.lightgbm.log_model(result.model, name="model")
            else:
                mlflow.sklearn.log_model(result.model, name="model")

            run_id = run.info.run_id
            logger.info(
                "MLflow run logged | model=%s, run_id=%s", result.name, run_id
            )
            return run_id

    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "MLflow logging failed for '%s': %s. "
            "Training will continue without tracking.",
            result.name, exc,
        )
        return None


# ---------------------------------------------------------------------------
# Individual model trainers
# ---------------------------------------------------------------------------


def train_logistic_regression(
    X_train: np.ndarray | pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    X_val: np.ndarray | pd.DataFrame,
    y_val: pd.Series | np.ndarray,
    config: Mapping[str, Any],
    threshold: float = 0.5,
) -> ModelResult:
    """Train a Logistic Regression model.

    Expects OneHot-encoded and RobustScaled input (built via
    ``build_linear_preprocessor`` in ``run_training``).  Using raw
    OrdinalEncoded features would impose a false numeric ordering on
    categorical columns, causing the coefficients to learn garbage signals.

    Args:
        X_train: Training feature matrix (OneHot + RobustScaled).
        y_train: Training labels.
        X_val: Validation feature matrix (same preprocessing).
        y_val: Validation labels.
        config: Parsed project configuration.
        threshold: Decision threshold for binary classification metrics.

    Returns:
        Populated ``ModelResult``.
    """
    random_state: int = int(
        get_value(config, "project", "random_state", default=42)
    )
    params: dict[str, Any] = {
        "solver":       str(get_value(config, "logistic_regression", "solver",   default="lbfgs")),
        "C":            float(get_value(config, "logistic_regression", "C",      default=1.0)),
        "max_iter":     int(get_value(config, "logistic_regression", "max_iter", default=1000)),
        "random_state": random_state,
    }
    # class_weight intentionally omitted: 'balanced' makes weight_pos*N_pos
    # ~ weight_neg*N_neg, forcing the intercept-only solution sigmoid(b)~0.5
    # for all users. Model instead learns natural P(churn)~0.064 probabilities.
    # penalty intentionally omitted: lbfgs default is l2; explicit 'l2' triggers
    # FutureWarning in sklearn 1.0+.
    logger.info("Training Logistic Regression | params=%s", params)

    model = LogisticRegression(**params)
    model.fit(X_train, y_train)

    train_score = model.predict_proba(X_train)[:, 1]
    val_score   = model.predict_proba(X_val)[:, 1]

    threshold_metric = str(get_value(config, "modeling", "decision_threshold_metric", default="f2"))
    threshold_grid_step = float(get_value(config, "modeling", "decision_threshold_grid_step", default=0.01))
    tuned_threshold = _tune_decision_threshold(
        y_val,
        val_score,
        metric=threshold_metric,
        grid_step=threshold_grid_step,
    )

    train_metrics = _compute_metrics(y_train, train_score, tuned_threshold)
    val_metrics   = _compute_metrics(y_val,   val_score,   tuned_threshold)

    logger.info(
        "Logistic Regression | val_auc_roc=%.4f  val_auc_pr=%.4f  "
        "val_f1=%.4f  lift@10%%=%.2f",
        val_metrics.auc_roc, val_metrics.auc_pr,
        val_metrics.f1,      val_metrics.lift_at_top10,
    )

    return ModelResult(
        name="logistic_regression",
        model=model,
        params=params,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        decision_threshold=tuned_threshold,
    )


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    config: Mapping[str, Any],
    threshold: float = 0.5,
) -> ModelResult:
    """Train a Random Forest classifier.

    Random Forest is included as a strong non-linear baseline and provides
    reliable feature importance estimates via mean decrease in impurity.

    Args:
        X_train: Training feature matrix (OrdinalEncoded).
        y_train: Training labels.
        X_val: Validation feature matrix.
        y_val: Validation labels.
        config: Parsed project configuration.
        threshold: Decision threshold for binary classification metrics.

    Returns:
        Populated ``ModelResult``.
    """
    random_state: int = int(
        get_value(config, "project", "random_state", default=42)
    )
    params: dict[str, Any] = {
        "n_estimators":      int(get_value(config, "random_forest", "n_estimators", default=300)),
        "max_depth":         get_value(config, "random_forest", "max_depth", default=None),
        "min_samples_split": int(get_value(config, "random_forest", "min_samples_split", default=2)),
        "min_samples_leaf":  int(get_value(config, "random_forest", "min_samples_leaf", default=1)),
        "class_weight":      get_value(config, "random_forest", "class_weight", default="balanced"),
        "n_jobs":            int(get_value(config, "random_forest", "n_jobs", default=-1)),
        "random_state":      random_state,
    }
    logger.info("Training Random Forest | params=%s", params)

    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    train_score = model.predict_proba(X_train)[:, 1]
    val_score   = model.predict_proba(X_val)[:, 1]

    threshold_metric = str(get_value(config, "modeling", "decision_threshold_metric", default="f2"))
    threshold_grid_step = float(get_value(config, "modeling", "decision_threshold_grid_step", default=0.01))
    tuned_threshold = _tune_decision_threshold(
        y_val,
        val_score,
        metric=threshold_metric,
        grid_step=threshold_grid_step,
    )

    train_metrics = _compute_metrics(y_train, train_score, tuned_threshold)
    val_metrics   = _compute_metrics(y_val,   val_score,   tuned_threshold)

    logger.info(
        "Random Forest | val_auc_roc=%.4f  val_auc_pr=%.4f  "
        "val_f1=%.4f  lift@10%%=%.2f",
        val_metrics.auc_roc, val_metrics.auc_pr,
        val_metrics.f1,      val_metrics.lift_at_top10,
    )

    return ModelResult(
        name="random_forest",
        model=model,
        params=params,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
    )


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    config: Mapping[str, Any],
    threshold: float = 0.5,
) -> ModelResult:
    """Train an XGBoost gradient boosting model with early stopping.

    Early stopping monitors ``aucpr`` (AUC-PR) on the validation set and
    halts when no improvement is seen for ``early_stopping_rounds`` rounds.
    ``scale_pos_weight`` compensates for the class imbalance.

    Args:
        X_train: Training feature matrix (OrdinalEncoded).
        y_train: Training labels.
        X_val: Validation feature matrix.
        y_val: Validation labels.
        config: Parsed project configuration.
        threshold: Decision threshold for binary classification metrics.

    Returns:
        Populated ``ModelResult`` with ``best_iteration`` set.
    """
    random_state: int = int(
        get_value(config, "project", "random_state", default=42)
    )
    early_stopping_rounds: int = int(
        get_value(config, "xgboost", "early_stopping_rounds", default=50)
    )
    log_eval_period: int = int(
        get_value(config, "xgboost", "log_eval_period", default=100)
    )

    params: dict[str, Any] = {
        "n_estimators":         int(get_value(config, "xgboost", "n_estimators", default=1000)),
        "learning_rate":        float(get_value(config, "xgboost", "learning_rate", default=0.05)),
        "max_depth":            int(get_value(config, "xgboost", "max_depth", default=6)),
        "subsample":            float(get_value(config, "xgboost", "subsample", default=0.8)),
        "colsample_bytree":     float(get_value(config, "xgboost", "colsample_bytree", default=0.8)),
        "tree_method":          str(get_value(config, "xgboost", "tree_method", default="hist")),
        "eval_metric":          str(get_value(config, "xgboost", "eval_metric", default="aucpr")),
        "early_stopping_rounds": early_stopping_rounds,
        "scale_pos_weight":     _scale_pos_weight(y_train),
        "random_state":         random_state,
        "n_jobs":               -1,
        "verbosity":            0,
    }
    logger.info("Training XGBoost | params=%s", params)

    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=log_eval_period,
    )

    best_iter = int(model.best_iteration) if hasattr(model, "best_iteration") else params["n_estimators"]

    train_score = model.predict_proba(X_train)[:, 1]
    val_score   = model.predict_proba(X_val)[:, 1]

    threshold_metric = str(get_value(config, "modeling", "decision_threshold_metric", default="f2"))
    threshold_grid_step = float(get_value(config, "modeling", "decision_threshold_grid_step", default=0.01))
    tuned_threshold = _tune_decision_threshold(
        y_val,
        val_score,
        metric=threshold_metric,
        grid_step=threshold_grid_step,
    )

    train_metrics = _compute_metrics(y_train, train_score, tuned_threshold)
    val_metrics   = _compute_metrics(y_val,   val_score,   tuned_threshold)

    logger.info(
        "XGBoost | best_iter=%d  val_auc_roc=%.4f  val_auc_pr=%.4f  "
        "val_f1=%.4f  lift@10%%=%.2f",
        best_iter,
        val_metrics.auc_roc, val_metrics.auc_pr,
        val_metrics.f1,      val_metrics.lift_at_top10,
    )

    return ModelResult(
        name="xgboost",
        model=model,
        params=params,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        best_iteration=best_iter,
        decision_threshold=tuned_threshold,
    )


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    config: Mapping[str, Any],
    threshold: float = 0.5,
) -> ModelResult:
    """Train a LightGBM gradient boosting model with early stopping.

    Early stopping monitors ``average_precision`` (AUC-PR) on the validation
    set, consistent with the project's champion metric.  ``scale_pos_weight``
    compensates for the churn minority class.

    Args:
        X_train: Training feature matrix (OrdinalEncoded).
        y_train: Training labels.
        X_val: Validation feature matrix.
        y_val: Validation labels.
        config: Parsed project configuration.
        threshold: Decision threshold for binary classification metrics.

    Returns:
        Populated ``ModelResult`` with ``best_iteration`` set.
    """
    random_state: int = int(
        get_value(config, "project", "random_state", default=42)
    )
    early_stopping_rounds: int = int(
        get_value(config, "lightgbm", "early_stopping_rounds", default=100)
    )
    log_eval_period: int = int(
        get_value(config, "lightgbm", "log_eval_period", default=100)
    )
    metric: str = str(
        get_value(config, "lightgbm", "metric", default="aucpr")
    )

    params: dict[str, Any] = {
        "n_estimators":     int(get_value(config, "lightgbm", "n_estimators", default=1000)),
        "learning_rate":    float(get_value(config, "lightgbm", "learning_rate", default=0.05)),
        "num_leaves":       int(get_value(config, "lightgbm", "num_leaves", default=31)),
        "max_depth":        int(get_value(config, "lightgbm", "max_depth", default=-1)),
        "min_child_samples":int(get_value(config, "lightgbm", "min_child_samples", default=50)),
        "reg_alpha":        float(get_value(config, "lightgbm", "reg_alpha", default=0.1)),
        "reg_lambda":       float(get_value(config, "lightgbm", "reg_lambda", default=1.0)),
        "subsample":        float(get_value(config, "lightgbm", "subsample", default=0.8)),
        "colsample_bytree": float(get_value(config, "lightgbm", "colsample_bytree", default=0.8)),
        "scale_pos_weight": _scale_pos_weight(y_train),
        # Setting metric in the constructor ensures early_stopping callback
        # monitors ONLY this metric (not binary_logloss which is the default).
        # "auc" is stable, well-calibrated, and "higher = better" — the callback
        # correctly infers the maximisation direction.
        "metric":           metric,
        "random_state":     random_state,
        "n_jobs":           -1,
        "verbose":          -1,
    }
    logger.info("Training LightGBM | params=%s", params)

    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        # eval_metric is intentionally omitted here — the metric is already set
        # in the constructor via params["metric"].  Passing it again in fit()
        # would add a SECOND metric (causing early_stopping to monitor both
        # binary_logloss and our metric, and stop on whichever degrades first).
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=early_stopping_rounds,
                verbose=False,
            ),
            lgb.log_evaluation(period=log_eval_period),
        ],
    )

    best_iter = int(model.best_iteration_) if model.best_iteration_ else params["n_estimators"]

    train_score = model.predict_proba(X_train)[:, 1]
    val_score   = model.predict_proba(X_val)[:, 1]

    threshold_metric = str(get_value(config, "modeling", "decision_threshold_metric", default="f2"))
    threshold_grid_step = float(get_value(config, "modeling", "decision_threshold_grid_step", default=0.01))
    tuned_threshold = _tune_decision_threshold(
        y_val,
        val_score,
        metric=threshold_metric,
        grid_step=threshold_grid_step,
    )

    train_metrics = _compute_metrics(y_train, train_score, tuned_threshold)
    val_metrics   = _compute_metrics(y_val,   val_score,   tuned_threshold)

    logger.info(
        "LightGBM | best_iter=%d  val_auc_roc=%.4f  val_auc_pr=%.4f  "
        "val_f1=%.4f  lift@10%%=%.2f",
        best_iter,
        val_metrics.auc_roc, val_metrics.auc_pr,
        val_metrics.f1,      val_metrics.lift_at_top10,
    )

    return ModelResult(
        name="lightgbm",
        model=model,
        params=params,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        best_iteration=best_iter,
        decision_threshold=tuned_threshold,
    )


def train_xgboost_optuna_candidate(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    config: Mapping[str, Any],
    base_xgb_result: ModelResult,
    threshold: float = 0.5,
) -> ModelResult:
    """Train an Optuna-tuned XGBoost candidate with optional auto top-K search."""

    random_state = int(get_value(config, "project", "random_state", default=42))
    n_trials = int(get_value(config, "optuna", "n_trials", default=50))
    timeout = get_value(config, "optuna", "timeout", default=None)
    num_boost_round = int(get_value(config, "optuna", "num_boost_round", default=1000))
    early_stopping = int(get_value(config, "optuna", "early_stopping_rounds", default=50))
    verbose_eval = int(get_value(config, "optuna", "verbose_eval", default=25))
    direction = str(get_value(config, "optuna", "direction", default="maximize"))
    eval_metric = str(get_value(config, "xgboost", "eval_metric", default="aucpr"))
    tune_top_k = bool(get_value(config, "optuna", "tune_top_k", default=True))
    top_k_min = int(get_value(config, "optuna", "top_k_min", default=10))
    top_k_max = int(get_value(config, "optuna", "top_k_max", default=100))
    top_k_step = int(get_value(config, "optuna", "top_k_step", default=5))

    base_booster = base_xgb_result.model.get_booster()
    gain_map = base_booster.get_score(importance_type="gain")
    ranked_features = [
        feat
        for feat, _ in sorted(gain_map.items(), key=lambda kv: kv[1], reverse=True)
        if feat in X_train.columns
    ]
    if not ranked_features:
        ranked_features = list(X_train.columns)

    k_min = max(1, min(top_k_min, len(ranked_features)))
    k_max = max(k_min, min(top_k_max, len(ranked_features)))

    y_train_arr = np.asarray(y_train).ravel()
    y_val_arr = np.asarray(y_val).ravel()

    def objective(trial: optuna.Trial) -> float:
        if tune_top_k:
            top_k = trial.suggest_int("top_k", k_min, k_max, step=top_k_step)
        else:
            top_k = len(ranked_features)
        selected = ranked_features[:top_k]

        dtrain = xgb.DMatrix(
            X_train[selected].values,
            label=y_train_arr,
            feature_names=selected,
        )
        dval = xgb.DMatrix(
            X_val[selected].values,
            label=y_val_arr,
            feature_names=selected,
        )

        params = {
            "objective": "binary:logistic",
            "eval_metric": eval_metric,
            "booster": "gbtree",
            "eta": trial.suggest_float("eta", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "lambda": trial.suggest_float("lambda", 1e-3, 10.0, log=True),
            "alpha": trial.suggest_float("alpha", 1e-3, 10.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "seed": random_state,
            "verbosity": 0,
        }
        booster = xgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            evals=[(dval, "valid")],
            early_stopping_rounds=early_stopping,
            verbose_eval=verbose_eval,
        )
        trial.set_user_attr("best_iteration", int(booster.best_iteration))
        trial.set_user_attr("best_ntree_limit", int(booster.best_iteration) + 1)
        trial.set_user_attr("top_k", int(top_k))
        return float(booster.best_score)

    logger.info("Training Optuna XGBoost candidate | n_trials=%d", n_trials)
    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)

    best = study.best_params
    best_trial = study.best_trial
    best_iteration = int(best_trial.user_attrs.get("best_iteration", 0))
    best_ntree_limit = int(best_trial.user_attrs.get("best_ntree_limit", best_iteration + 1))
    best_top_k = int(best_trial.user_attrs.get("top_k", len(ranked_features)))
    selected_features = ranked_features[:best_top_k]

    final_params = best.copy()
    final_params.update(
        {
            "objective": "binary:logistic",
            "eval_metric": eval_metric,
            "seed": random_state,
            "verbosity": 0,
        }
    )
    dtrain_final = xgb.DMatrix(
        X_train[selected_features].values,
        label=y_train_arr,
        feature_names=selected_features,
    )
    dval_final = xgb.DMatrix(
        X_val[selected_features].values,
        label=y_val_arr,
        feature_names=selected_features,
    )

    final_booster = xgb.train(
        final_params,
        dtrain_final,
        num_boost_round=best_ntree_limit,
        evals=[(dval_final, "valid")],
        verbose_eval=verbose_eval,
    )

    train_score = final_booster.predict(dtrain_final)
    val_score = final_booster.predict(dval_final)

    threshold_metric = str(get_value(config, "modeling", "decision_threshold_metric", default="f2"))
    threshold_grid_step = float(get_value(config, "modeling", "decision_threshold_grid_step", default=0.01))
    tuned_threshold = _tune_decision_threshold(
        y_val,
        val_score,
        metric=threshold_metric,
        grid_step=threshold_grid_step,
    )

    train_metrics = _compute_metrics(y_train, train_score, tuned_threshold)
    val_metrics = _compute_metrics(y_val, val_score, tuned_threshold)

    model = FeatureSubsetModel(
        model=BoosterModel(final_booster, selected_features),
        feature_names=selected_features,
    )
    params_out = dict(final_params)
    params_out["best_optuna_val_aucpr"] = float(study.best_value)
    params_out["best_ntree_limit"] = int(best_ntree_limit)
    params_out["selected_top_k"] = int(best_top_k)
    params_out["feature_count"] = int(len(selected_features))

    return ModelResult(
        name="xgboost_optuna_topk",
        model=model,
        params=params_out,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        best_iteration=best_iteration,
        decision_threshold=tuned_threshold,
    )


def train_xgboost_topk_candidate(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    config: Mapping[str, Any],
    base_xgb_result: ModelResult,
    threshold: float = 0.5,
) -> ModelResult:
    """Train a top-K-features XGBoost candidate for champion selection."""

    top_k = int(get_value(config, "analysis", "top_k_features", default=30))
    booster = base_xgb_result.model.get_booster()
    gain_map = booster.get_score(importance_type="gain")
    if not gain_map:
        raise ValueError("Top-K candidate requires XGBoost gain importances, but none were found.")

    ranked_features = [
        feat
        for feat, _ in sorted(gain_map.items(), key=lambda kv: kv[1], reverse=True)
        if feat in X_train.columns
    ]
    selected_features = ranked_features[:top_k]
    if not selected_features:
        raise ValueError("Top-K candidate found no overlapping features in training matrix.")

    logger.info("Training top-K XGBoost candidate | top_k=%d | selected=%d", top_k, len(selected_features))
    X_train_topk = X_train[selected_features]
    X_val_topk = X_val[selected_features]

    params = {
        "n_estimators": int(get_value(config, "xgboost", "n_estimators", default=1000)),
        "learning_rate": float(get_value(config, "xgboost", "learning_rate", default=0.05)),
        "max_depth": int(get_value(config, "xgboost", "max_depth", default=6)),
        "subsample": float(get_value(config, "xgboost", "subsample", default=0.8)),
        "colsample_bytree": float(get_value(config, "xgboost", "colsample_bytree", default=0.8)),
        "tree_method": str(get_value(config, "xgboost", "tree_method", default="hist")),
        "eval_metric": str(get_value(config, "xgboost", "eval_metric", default="aucpr")),
        "early_stopping_rounds": int(get_value(config, "xgboost", "early_stopping_rounds", default=100)),
        "scale_pos_weight": _scale_pos_weight(y_train),
        "random_state": int(get_value(config, "project", "random_state", default=42)),
        "n_jobs": -1,
        "verbosity": 0,
    }
    log_eval_period = int(get_value(config, "xgboost", "log_eval_period", default=100))

    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train_topk,
        y_train,
        eval_set=[(X_val_topk, y_val)],
        verbose=log_eval_period,
    )

    best_iter = int(model.best_iteration) if hasattr(model, "best_iteration") else int(params["n_estimators"])
    train_score = model.predict_proba(X_train_topk)[:, 1]
    val_score = model.predict_proba(X_val_topk)[:, 1]

    threshold_metric = str(get_value(config, "modeling", "decision_threshold_metric", default="f2"))
    threshold_grid_step = float(get_value(config, "modeling", "decision_threshold_grid_step", default=0.01))
    tuned_threshold = _tune_decision_threshold(
        y_val,
        val_score,
        metric=threshold_metric,
        grid_step=threshold_grid_step,
    )

    train_metrics = _compute_metrics(y_train, train_score, tuned_threshold)
    val_metrics = _compute_metrics(y_val, val_score, tuned_threshold)

    wrapped = FeatureSubsetModel(model=model, feature_names=selected_features)
    params_out = dict(params)
    params_out["selected_topk_features"] = selected_features

    return ModelResult(
        name="xgboost_topk",
        model=wrapped,
        params=params_out,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        best_iteration=best_iter,
        decision_threshold=tuned_threshold,
    )


# ---------------------------------------------------------------------------
# Champion selection
# ---------------------------------------------------------------------------


def select_champion(
    results: list[ModelResult],
    champion_metric: str = "auc_pr",
) -> ModelResult:
    """Select the best model by a validation metric.

    Args:
        results: List of trained ModelResults to compare.
        champion_metric: Attribute name on ``MetricsBundle`` to rank by.

    Returns:
        The ``ModelResult`` with the highest validation metric.

    Raises:
        ValueError: If ``results`` is empty or metric name is invalid.
    """
    if not results:
        raise ValueError("Cannot select a champion from an empty results list.")

    for result in results:
        if not hasattr(result.val_metrics, champion_metric):
            raise ValueError(
                f"Unknown champion metric '{champion_metric}'. "
                f"Valid options: {list(MetricsBundle.__dataclass_fields__.keys())}"
            )

    champion = max(
        results,
        key=lambda r: getattr(r.val_metrics, champion_metric),
    )

    logger.info(
        "Champion selected | model=%s  %s=%.4f",
        champion.name,
        champion_metric,
        getattr(champion.val_metrics, champion_metric),
    )
    for result in sorted(
        results,
        key=lambda r: getattr(r.val_metrics, champion_metric),
        reverse=True,
    ):
        logger.info(
            "  %-25s  val_%s=%.4f",
            result.name,
            champion_metric,
            getattr(result.val_metrics, champion_metric),
        )

    return champion


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------


def save_model(model: Any, path: Path) -> None:
    """Serialise a fitted model to a pickle file.

    Args:
        model: Fitted scikit-learn-compatible estimator.
        path: Destination ``.pkl`` file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Model saved -> %s", path)


def load_model(path: Path) -> Any:
    """Load a previously serialised model from disk.

    Args:
        path: Path to the ``.pkl`` file written by ``save_model()``.

    Returns:
        The deserialised estimator.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    with path.open("rb") as handle:
        model = pickle.load(handle)
    logger.info("Model loaded from %s", path)
    return model


def save_comparison_table(
    results: list[ModelResult],
    path: Path,
) -> pd.DataFrame:
    """Build and save a side-by-side model comparison CSV.

    Args:
        results: List of all trained ModelResults.
        path: Destination ``.csv`` file path.

    Returns:
        The comparison DataFrame (also saved to disk).
    """
    rows: list[dict[str, Any]] = []
    for result in results:
        row: dict[str, Any] = {"model": result.name}
        row.update(result.train_metrics.to_dict(prefix="train_"))
        row.update(result.val_metrics.to_dict(prefix="val_"))
        row["best_iteration"] = result.best_iteration
        row["decision_threshold"] = round(result.decision_threshold, 6)
        row["mlflow_run_id"]  = result.mlflow_run_id or ""
        rows.append(row)

    comparison = (
        pd.DataFrame(rows)
        .sort_values("val_auc_pr", ascending=False)
        .reset_index(drop=True)
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(path, index=False)
    logger.info("Model comparison table saved -> %s", path)
    logger.info("\n%s", comparison.to_string(index=False))

    return comparison


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_training(
    config: Mapping[str, Any],
    project_root: Path,
) -> tuple[ModelResult, list[ModelResult], pd.DataFrame]:
    """Execute the full training stage end-to-end.

    Steps
    -----
    1. Load OrdinalEncoded splits (X_train / X_val) from ``data/processed/``.
    2. If Logistic Regression is a candidate, load the raw feature frame and
       fit a OneHot+RobustScaler linear preprocessor to produce X_train_lr /
       X_val_lr.  This avoids the false ordinal ordering that OrdinalEncoded
       features impose on linear models.
    3. Train all candidate models, dispatching linear models to OneHot data
       and tree models to OrdinalEncoded data.
    4. Select the champion by ``config.modeling.champion_metric`` (AUC-PR).
    5. Save each model individually and the champion separately; write a
       comparison CSV to ``reports/``.

    Args:
        config: Parsed project configuration (from ``config.yaml``).
        project_root: Absolute path to the project root directory.

    Returns:
        Tuple of ``(champion_result, all_results, comparison_df)``.

    Raises:
        FileNotFoundError: If the preprocessed split files are absent.
    """
    processed_dir = project_root / "data" / "processed"
    models_dir    = project_root / "models"
    reports_dir   = project_root / "reports"

    threshold: float = float(
        get_value(config, "modeling", "decision_threshold", default=0.5)
    )
    champion_metric: str = str(
        get_value(config, "modeling", "champion_metric", default="auc_pr")
    )
    experiment_name: str = str(
        get_value(config, "mlflow", "experiment_name", default="kkbox-churn")
    )
    tracking_uri: str | None = get_value(config, "mlflow", "tracking_uri")
    candidates: list[str] = list(
        get_value(
            config,
            "modeling",
            "candidate_models",
            default=["logistic_regression", "random_forest", "lightgbm"],
        )
    )
    use_optuna_candidate: bool = bool(
        get_value(config, "modeling", "use_optuna_candidate", default=True)
    )
    use_topk_candidate: bool = bool(
        get_value(config, "modeling", "use_topk_candidate", default=True)
    )

    # -------------------------------------------------------------------------
    # 1. Load OrdinalEncoded splits (for tree models)
    # -------------------------------------------------------------------------
    logger.info("Loading preprocessed splits from %s ...", processed_dir)

    def _load(name: str) -> pd.DataFrame | pd.Series:
        path = processed_dir / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"Split file not found: {path}. "
                "Run 'python src/features/run_preprocessing.py' first."
            )
        df = pd.read_parquet(path)
        if name.startswith("y_"):
            return df.iloc[:, 0].rename("is_churn")
        return df

    X_train: pd.DataFrame = _load("X_train")
    X_val:   pd.DataFrame = _load("X_val")
    y_train: pd.Series    = _load("y_train")
    y_val:   pd.Series    = _load("y_val")

    logger.info(
        "Splits loaded | X_train=%s, X_val=%s | "
        "churn_rate_train=%.4f, churn_rate_val=%.4f",
        X_train.shape, X_val.shape,
        y_train.mean(), y_val.mean(),
    )

    # -------------------------------------------------------------------------
    # 2. Build OneHot linear preprocessor for Logistic Regression
    # -------------------------------------------------------------------------
    # Tree models (RF, XGB, LGB) use the OrdinalEncoded splits above.
    # LR needs OneHotEncoded + RobustScaled data built from the raw feature
    # frame, because OrdinalEncoded integers impose a false ordering on
    # categorical features that confounds linear coefficients.
    X_train_lr: np.ndarray | None = None
    X_val_lr:   np.ndarray | None = None
    y_train_lr: pd.Series  | None = None
    y_val_lr:   pd.Series  | None = None

    if "logistic_regression" in candidates:
        output_file = str(
            get_value(config, "feature_engineering", "output_file",
                      default="feature_frame.parquet")
        )
        ff_path = processed_dir / output_file
        if not ff_path.exists():
            raise FileNotFoundError(
                f"Feature frame not found: {ff_path}. "
                "Run 'python src/features/run_engineer.py' first."
            )

        logger.info("Loading feature frame for linear preprocessor: %s", ff_path)
        ff = pd.read_parquet(ff_path)
        target_col: str = str(
            get_value(config, "project", "target_col", default="is_churn")
        )

        # Reproduce the SAME train/val split used by run_preprocessing.py.
        # split_dataset uses the same random_state and sizes from config,
        # so train_ff rows are identical to those in X_train / y_train.
        train_ff, val_ff, _ = split_dataset(ff, config)

        # Columns to exclude from the linear feature matrix.
        _LINEAR_DROP: set[str] = {
            target_col, "msno", "analysis_reference_date",
            "registration_init_time", "last_transaction_date",
            "last_expire_date", "last_log_date",
            # Raw demographic columns superseded by derived features:
            "bd", "gender", "city", "registered_via",
            # Raw listening-time columns: values reach ±3e11 after RobustScaler
            # (near-zero IQR, heavy-tailed) → overwhelm L2 regularisation.
            # Use log_total_secs / log_mean_secs instead (already in feature frame).
            "total_secs", "mean_secs",
        }
        feat_cols = [c for c in ff.columns if c not in _LINEAR_DROP]

        groups_lr = identify_column_groups(
            train_ff[feat_cols], extra_drop=list(_LINEAR_DROP)
        )
        # Pass X_ref so high-cardinality object columns (numeric features
        # accidentally stored as object dtype) are excluded from OHE.
        # Without this guard, OHE can produce millions of columns and OOM.
        lr_preprocessor = build_linear_preprocessor(
            groups_lr,
            X_ref=train_ff[feat_cols],
            max_ohe_categories=50,
        )

        # Sanitise pandas extension dtypes before sklearn sees the data.
        train_lr_clean = _sanitise_for_sklearn(train_ff[feat_cols])
        val_lr_clean   = _sanitise_for_sklearn(val_ff[feat_cols])

        logger.info(
            "Fitting linear preprocessor | numeric=%d, categorical=%d",
            len(groups_lr.numeric), len(groups_lr.categorical),
        )
        X_train_lr = lr_preprocessor.fit_transform(train_lr_clean)
        X_val_lr   = lr_preprocessor.transform(val_lr_clean)
        y_train_lr = pd.Series(train_ff[target_col].values, name="is_churn")
        y_val_lr   = pd.Series(val_ff[target_col].values,   name="is_churn")

        logger.info(
            "Linear preprocessor fitted | X_train_lr=%s", X_train_lr.shape
        )

    # -------------------------------------------------------------------------
    # 3. Train all candidate models
    # -------------------------------------------------------------------------
    # Linear models receive OneHot-preprocessed data; tree models receive
    # the OrdinalEncoded splits from data/processed/.
    _LINEAR_MODELS: set[str] = {"logistic_regression"}

    trainer_map: dict[str, Any] = {
        "logistic_regression": train_logistic_regression,
        "random_forest":       train_random_forest,
        "xgboost":             train_xgboost,
        "lightgbm":            train_lightgbm,
    }

    all_results: list[ModelResult] = []

    for model_name in candidates:
        if model_name not in trainer_map:
            logger.warning(
                "Unknown candidate model '%s' in config — skipping.", model_name
            )
            continue

        logger.info("=" * 60)
        logger.info("STAGE: %s", model_name)
        logger.info("=" * 60)

        trainer_fn = trainer_map[model_name]

        if model_name in _LINEAR_MODELS:
            if X_train_lr is None:
                logger.warning(
                    "Linear preprocessed data not available for '%s' — skipping.",
                    model_name,
                )
                continue
            result = trainer_fn(
                X_train_lr, y_train_lr,
                X_val_lr,   y_val_lr,
                config, threshold,
            )
        else:
            result = trainer_fn(
                X_train, y_train,
                X_val,   y_val,
                config, threshold,
            )

        result.mlflow_run_id = _log_to_mlflow(result, experiment_name, tracking_uri)
        all_results.append(result)

    # ---------------------------------------------------------------------
    # 3b. Advanced official candidates (Optuna + Top-K)
    # ---------------------------------------------------------------------
    xgb_base = next((r for r in all_results if r.name == "xgboost"), None)

    if use_optuna_candidate and xgb_base is not None:
        logger.info("=" * 60)
        logger.info("STAGE: xgboost_optuna_topk")
        logger.info("=" * 60)
        optuna_result = train_xgboost_optuna_candidate(
            X_train,
            y_train,
            X_val,
            y_val,
            config,
            xgb_base,
            threshold,
        )
        optuna_result.mlflow_run_id = _log_to_mlflow(optuna_result, experiment_name, tracking_uri)
        all_results.append(optuna_result)

    if use_topk_candidate and xgb_base is not None:
        logger.info("=" * 60)
        logger.info("STAGE: xgboost_topk")
        logger.info("=" * 60)
        topk_result = train_xgboost_topk_candidate(
            X_train,
            y_train,
            X_val,
            y_val,
            config,
            xgb_base,
            threshold,
        )
        topk_result.mlflow_run_id = _log_to_mlflow(topk_result, experiment_name, tracking_uri)
        all_results.append(topk_result)

    if not all_results:
        raise ValueError(
            "No models were trained. "
            "Check 'candidate_models' in config.yaml."
        )

    # -------------------------------------------------------------------------
    # 4. Champion selection
    # -------------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STAGE: Champion Selection  (metric: val_%s)", champion_metric)
    logger.info("=" * 60)

    champion = select_champion(all_results, champion_metric)

    # -------------------------------------------------------------------------
    # 5. Persist champion, individual models, and comparison table
    # -------------------------------------------------------------------------
    save_model(champion.model, models_dir / "champion_model.pkl")

    champion_threshold_path = models_dir / str(
        get_value(config, "artifacts", "champion_threshold_file", default="champion_threshold.txt")
    )
    champion_threshold_path.write_text(f"{champion.decision_threshold:.6f}", encoding="utf-8")
    logger.info("Champion threshold written -> %s  (%.6f)", champion_threshold_path, champion.decision_threshold)

    for result in all_results:
        save_model(result.model, models_dir / f"{result.name}.pkl")

    comparison_df = save_comparison_table(
        all_results,
        reports_dir / "model_comparison.csv",
    )

    # Write the champion name so downstream stages (tuner, evaluator) can
    # discover it without parsing the CSV.
    champion_name_path = models_dir / "champion_name.txt"
    champion_name_path.write_text(champion.name, encoding="utf-8")
    logger.info(
        "Champion name written -> %s  (%s)",
        champion_name_path, champion.name,
    )

    logger.info("=" * 60)
    logger.info("Training stage complete. Champion: %s", champion.name)
    logger.info("=" * 60)

    return champion, all_results, comparison_df