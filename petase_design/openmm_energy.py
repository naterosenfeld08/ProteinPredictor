"""
Tier 3 (planned): short OpenMM energy minimization + decomposition.

Requires: OpenMM, a parameterized system (OpenFF + PDBFixer, or AMBER files).
Not implemented in P0 — :func:`minimize_and_score` raises until a pipeline is wired in.

Suggested workflow for a future implementation:
  1. PDB → protonation (PDBFixer) at target pH
  2. Parameterize (OpenFF 2.x + OpenMM ``ForceField``)
  3. Minimize (500–2000 steps, implicit solvent GBSA)
  4. Report potential energy + breakdown as a stability proxy

Compare WT vs mutant at the same protocol — relative energy as a ΔΔG-stability proxy (not experimental Tm).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def minimize_and_score(_pdb_path: Path, _platform: str = "CPU") -> dict[str, Any]:
    raise NotImplementedError("Install OpenMM + parameterization pipeline; see module docstring.")
