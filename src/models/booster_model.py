from dataclasses import dataclass

import numpy as np
import xgboost as xgb


@dataclass
class BoosterModel:
    booster: xgb.Booster
    feature_names: list[str]

    def get_booster(self):
        return self.booster

    def predict_proba(self, X):
        matrix = xgb.DMatrix(X, feature_names=self.feature_names)
        proba = self.booster.predict(matrix)
        return np.column_stack([1.0 - proba, proba])

    def predict(self, X):
        proba = self.predict_proba(X)[:, 1]
        return (proba >= 0.5).astype(int)