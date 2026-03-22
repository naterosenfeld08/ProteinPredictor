from __future__ import annotations

"""
Tier 3 (planned): short OpenMM energy minimization + decomposition.

Requires: openmm, a parameterized system (OpenFF + pdbfixer pipeline, or AMBER files).
Not implemented in P0 — placeholder for roadmap.

Suggested workflow:
  1. PDB → protonation (PDBFixer) at target pH
  2. Parameterize (OpenFF 2.x + OpenMM ForceField)
  3. Minimize (500–2000 steps, implicit solvent GBSA)
  4. Report potential energy + breakdown (nonbonded vs bonded) as stability proxy

Compare WT vs mutant at same protocol — relative energy as ΔΔG_stab proxy (still not experimental Tm).
"""

from pathlib import Path
from typing import Any


def minimize_and_score(_pdb_path: Path, _platform: str = "CPU") -> dict[str, Any]:
    raise NotImplementedError("Install OpenMM + parameterization pipeline; see module docstring.")
