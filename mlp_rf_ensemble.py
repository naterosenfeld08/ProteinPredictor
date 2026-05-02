"""
Backward-compatible root imports for legacy module path.

Canonical implementation lives in `core.mlp_rf_ensemble`. Keeping this shim
ensures older pickle/module references can still resolve symbols from the repo root.
"""

from __future__ import annotations

from core.mlp_rf_ensemble import MLPEngineConfig, MLPRandomForestEnsemble

__all__ = ["MLPEngineConfig", "MLPRandomForestEnsemble"]

