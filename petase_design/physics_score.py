from __future__ import annotations

"""
Physics-informed and geometry proxies for thermostability screening.

Tier 0: sequence-only (hydropathy, charge, aromatics, active-site protection).
Tier 1: optional PDB Cα — radius of gyration, mean Cα–Cα distance in a shell (needs coords + active site index).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import math

from petase_design.sequence_utils import KD_HYDROPATHY, mutation_diff


@dataclass
class PhysicsBreakdown:
    mean_hydrophobicity: float
    net_charge_proxy: float
    aromatic_fraction: float
    active_site_violation: float
    radius_of_gyration: float | None
    mutation_count: int
    composite: float


def _mean_kd(seq: str) -> float:
    vals = [KD_HYDROPATHY.get(a, 0.0) for a in seq.upper()]
    return sum(vals) / max(len(vals), 1)


def _charge_proxy(seq: str) -> float:
    """Rough +1 K/R, -1 D/E at neutral pH; H +0.5."""
    q = 0.0
    for a in seq.upper():
        if a in "KR":
            q += 1.0
        elif a in "DE":
            q -= 1.0
        elif a == "H":
            q += 0.5
    return q


def _aromatic_fraction(seq: str) -> float:
    arom = sum(1 for a in seq.upper() if a in "FYW")
    return arom / max(len(seq), 1)


def active_site_penalty(
    wt: str,
    variant: str,
    protected_indices: Iterable[int],
) -> float:
    """0 = no protected residue changed; higher = more / harsher changes at protected sites."""
    prot = set(protected_indices)
    if not prot:
        return 0.0
    penalty = 0.0
    for i, a_wt, a_mut in mutation_diff(wt, variant):
        if i in prot:
            penalty += 2.0 + abs(KD_HYDROPATHY.get(a_mut, 0) - KD_HYDROPATHY.get(a_wt, 0)) * 0.1
    return penalty


def parse_pdb_ca_coords(pdb_path: Path) -> list[tuple[str, int, str, tuple[float, float, float]]]:
    """
    Minimal ATOM parser: CA atoms only.
    Returns list of (chain, resseq, resname, (x,y,z)).
    """
    coords: list[tuple[str, int, str, tuple[float, float, float]]] = []
    for line in pdb_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("ATOM"):
            continue
        if len(line) < 54:
            continue
        atom_name = line[12:16].strip()
        if atom_name != "CA":
            continue
        resname = line[17:20].strip()
        chain = line[21].strip() or "A"
        try:
            resseq = int(line[22:26])
        except ValueError:
            continue
        try:
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
        except ValueError:
            continue
        coords.append((chain, resseq, resname, (x, y, z)))
    return coords


def radius_of_gyration(coords: list[tuple[float, float, float]]) -> float:
    if not coords:
        return float("nan")
    n = len(coords)
    cx = sum(c[0] for c in coords) / n
    cy = sum(c[1] for c in coords) / n
    cz = sum(c[2] for c in coords) / n
    acc = 0.0
    for x, y, z in coords:
        acc += (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2
    return math.sqrt(acc / n)


def score_sequence_physics(
    wt: str,
    variant: str,
    *,
    protected_indices: Iterable[int] = (),
    structure_pdb: Path | None = None,
    weights: dict[str, float] | None = None,
) -> PhysicsBreakdown:
    """
    Higher composite ≈ more favorable for thermostability *proxy* (tunable, not experimental Tm).

    - Favors moderate hydrophobicity (not too buried charge on surface — crude).
    - Penalizes |charge_proxy| extremes slightly.
    - Slight reward for aromatic content (packing).
    - Penalizes mutations in protected_indices.
    - If structure_pdb given, lower Rg slightly rewarded (compactness proxy).
    """
    w = weights or {
        "hydrophobic_core_proxy": 0.35,
        "charge_balance": 0.15,
        "aromatic": 0.1,
        "active_site": 0.25,
        "compactness": 0.15,
    }

    mh = _mean_kd(variant)
    qc = _charge_proxy(variant)
    ar = _aromatic_fraction(variant)
    asp = active_site_penalty(wt, variant, protected_indices)
    n_mut = len(mutation_diff(wt, variant))

    # Normalize crude terms to roughly O(1)
    hydro_score = (mh + 2.0) / 6.0  # ~0..1
    hydro_score = max(0.0, min(1.0, hydro_score))
    charge_score = 1.0 - min(1.0, abs(qc) / 20.0)
    arom_score = min(1.0, ar / 0.25)
    active_score = 1.0 / (1.0 + asp)
    rg: float | None = None
    compact_score = 0.5
    if structure_pdb and structure_pdb.is_file():
        parsed = parse_pdb_ca_coords(structure_pdb)
        xyz = [t[3] for t in parsed]
        rg = radius_of_gyration(xyz)
        if not math.isnan(rg):
            # Smaller Rg → higher score, saturate
            compact_score = 1.0 / (1.0 + rg / 20.0)

    composite = (
        w["hydrophobic_core_proxy"] * hydro_score
        + w["charge_balance"] * charge_score
        + w.get("aromatic", 0.1) * arom_score
        + w["active_site"] * active_score
        + w.get("compactness", 0.15) * compact_score
        - 0.02 * n_mut
    )

    return PhysicsBreakdown(
        mean_hydrophobicity=mh,
        net_charge_proxy=qc,
        aromatic_fraction=ar,
        active_site_violation=asp,
        radius_of_gyration=rg,
        mutation_count=n_mut,
        composite=float(composite),
    )
