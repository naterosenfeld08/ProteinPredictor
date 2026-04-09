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


def minimize_and_score(
    _pdb_path: Path,
    _platform: str = "CPU",
    *,
    max_iterations: int = 500,
) -> dict[str, Any]:
    """
    Best-effort OpenMM minimization for protein-only PDBs.

    Returns a payload with `ok`, total energy, and coarse force-group decomposition.
    On missing OpenMM or setup failures, returns `ok=False` with an error string.
    """
    try:
        import openmm as mm
        from openmm import app, unit
    except Exception as e:
        return {"ok": False, "error": f"openmm_import_failed: {e}"}

    try:
        pdb = app.PDBFile(str(_pdb_path))
        ff = app.ForceField("amber14-all.xml", "implicit/gbn2.xml")
        system = ff.createSystem(
            pdb.topology,
            nonbondedMethod=app.NoCutoff,
            constraints=app.HBonds,
        )

        for i, force in enumerate(system.getForces()):
            force.setForceGroup(i % 31)

        integrator = mm.LangevinIntegrator(
            300 * unit.kelvin,
            1.0 / unit.picosecond,
            0.002 * unit.picoseconds,
        )
        try:
            platform = mm.Platform.getPlatformByName(str(_platform))
            simulation = app.Simulation(pdb.topology, system, integrator, platform)
        except Exception:
            simulation = app.Simulation(pdb.topology, system, integrator)
        simulation.context.setPositions(pdb.positions)
        simulation.minimizeEnergy(maxIterations=int(max_iterations))
        state = simulation.context.getState(getEnergy=True)
        total = float(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))

        force_breakdown: dict[str, float] = {}
        for i, force in enumerate(system.getForces()):
            s_i = simulation.context.getState(getEnergy=True, groups={i % 31})
            e_i = float(s_i.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
            force_breakdown[type(force).__name__] = force_breakdown.get(type(force).__name__, 0.0) + e_i

        n_atoms = sum(1 for _ in pdb.topology.atoms())
        n_residues = sum(1 for _ in pdb.topology.residues())
        per_residue = total / max(1, n_residues)
        return {
            "ok": True,
            "platform": str(_platform),
            "n_atoms": int(n_atoms),
            "n_residues": int(n_residues),
            "total_energy_kj_mol": float(total),
            "energy_per_residue_kj_mol": float(per_residue),
            "force_breakdown_kj_mol": force_breakdown,
        }
    except Exception as e:
        return {"ok": False, "error": f"openmm_minimize_failed: {e}"}
