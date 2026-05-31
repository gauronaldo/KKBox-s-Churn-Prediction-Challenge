from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class FeatureSubsetModel:
    """Wrapper that applies a fixed feature subset before prediction."""

    model: object
    feature_names: list[str]

    def predict_proba(self, X: pd.DataFrame):
        X_aligned = X.reindex(columns=self.feature_names)
        return self.model.predict_proba(X_aligned)

    def predict(self, X: pd.DataFrame):
        proba = self.predict_proba(X)[:, 1]
        return np.asarray(proba >= 0.5, dtype=int)
