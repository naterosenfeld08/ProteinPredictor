"""
MLP + RandomForest ensemble for protein ΔΔG prediction.

This module exists because the project's existing inference path assumes:
  - sklearn-style models (with .predict), and/or
  - EnsembleModel (RF + XGB)

We provide a custom ensemble with:
  - `predict(X)` for deterministic predictions
  - `predict_with_uncertainty(X)` that uses RF tree variance as uncertainty
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor

from core.mlp_baseline import BaselineMLP


def _predict_with_rf_uncertainty(
    rf_model: RandomForestRegressor, X: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (mean_predictions, std_across_trees)."""
    predictions_per_tree = np.array([tree.predict(X) for tree in rf_model.estimators_])
    pred_mean = np.mean(predictions_per_tree, axis=0)
    pred_std = np.std(predictions_per_tree, axis=0)
    return pred_mean, pred_std


@dataclass(frozen=True)
class MLPEngineConfig:
    """Shape and dropout for ``mlp_baseline.BaselineMLP`` when loading from a saved ensemble."""

    input_dim: int
    dropout: float


class MLPRandomForestEnsemble:
    """
    Weighted ensemble:
      y = w_rf * y_rf + (1 - w_rf) * y_mlp
    """

    def __init__(
        self,
        rf_model: RandomForestRegressor,
        mlp_state_dict: Dict[str, Any],
        mlp_config: MLPEngineConfig,
        feature_mean: np.ndarray,
        feature_std: np.ndarray,
        weight_rf: float = 0.5,
        device: Optional[str] = None,
    ):
        self.rf_model = rf_model
        # Ensure everything is on CPU before pickling
        self.mlp_state_dict = {k: v.detach().cpu() for k, v in mlp_state_dict.items()}
        self.mlp_config = mlp_config
        self.feature_mean = np.asarray(feature_mean, dtype=np.float32)
        self.feature_std = np.asarray(feature_std, dtype=np.float32)
        self.weight_rf = float(weight_rf)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Lazily constructed on first predict to keep pickles smaller.
        self._mlp_model: Optional[torch.nn.Module] = None

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_mlp_model"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._mlp_model = None

    def _ensure_mlp_loaded(self) -> None:
        if self._mlp_model is not None:
            return
        model = BaselineMLP(input_dim=self.mlp_config.input_dim, dropout=self.mlp_config.dropout)
        model.load_state_dict(self.mlp_state_dict)
        model.eval()
        model.to(self.device)
        self._mlp_model = model

    def _predict_mlp(self, X: np.ndarray, batch_size: int = 512) -> np.ndarray:
        self._ensure_mlp_loaded()
        assert self._mlp_model is not None

        X = np.asarray(X, dtype=np.float32)
        X_norm = (X - self.feature_mean) / self.feature_std

        preds = []
        with torch.no_grad():
            for start in range(0, X_norm.shape[0], batch_size):
                batch = torch.from_numpy(X_norm[start : start + batch_size]).to(self.device)
                out = self._mlp_model(batch).cpu().numpy().reshape(-1)
                preds.append(out)
        return np.concatenate(preds, axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        y_rf = self.rf_model.predict(X).reshape(-1)
        y_mlp = self._predict_mlp(X)
        return self.weight_rf * y_rf + (1.0 - self.weight_rf) * y_mlp

    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
          (predictions, uncertainty_std)

        Uncertainty comes only from the RandomForest component; the MLP is treated as deterministic.
        """
        X = np.asarray(X)
        y_rf_mean, y_rf_std = _predict_with_rf_uncertainty(self.rf_model, X)
        y_mlp = self._predict_mlp(X)
        y_ens = self.weight_rf * y_rf_mean + (1.0 - self.weight_rf) * y_mlp
        # Simple propagation: uncertainty scales with RF weight.
        y_ens_std = self.weight_rf * y_rf_std
        return y_ens, y_ens_std

