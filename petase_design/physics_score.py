"""
Physics-informed and geometry proxies for thermostability screening.

Tier 0: sequence-only (hydropathy, charge, aromatics, active-site protection).
Tier 1: optional PDB — Cα radius of gyration (compactness) + FreeSASA polar/apolar SASA when
``freesasa`` is installed (``petase_design/requirements-extras.txt``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import math

from petase_design.sequence_utils import KD_HYDROPATHY, mutation_diff
from petase_design.sasa_utils import compute_sasa_breakdown
from petase_design.openmm_energy import minimize_and_score


@dataclass
class PhysicsBreakdown:
    mean_hydrophobicity: float
    net_charge_proxy: float
    aromatic_fraction: float
    active_site_violation: float
    radius_of_gyration: float | None
    # FreeSASA (optional): total SASA Å²; apolar fraction = Apolar / (Polar+Apolar+Unknown)
    sasa_total_area: float | None
    apolar_sasa_fraction: float | None
    structure_confidence: float | None
    structural_viability_penalty: float | None
    openmm_total_energy_kj_mol: float | None
    openmm_energy_per_residue_kj_mol: float | None
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


def structure_viability_penalty(
    ca_atoms: list[tuple[str, int, str, tuple[float, float, float]]],
) -> tuple[float, float]:
    """
    Returns (penalty, confidence) in [0,1].
    Lower penalty / higher confidence are better.
    """
    if len(ca_atoms) < 20:
        return 1.0, 0.0
    xyz = [x[3] for x in ca_atoms]
    rg = radius_of_gyration(xyz)
    # Backbone continuity penalty from large C-alpha jumps.
    jumps = 0
    total_links = 0
    for i in range(1, len(ca_atoms)):
        ch0, r0, _n0, p0 = ca_atoms[i - 1]
        ch1, r1, _n1, p1 = ca_atoms[i]
        if ch0 != ch1:
            continue
        if abs(r1 - r0) > 2:
            continue
        total_links += 1
        d = math.dist(p0, p1)
        if d > 5.2 or d < 2.4:
            jumps += 1
    jump_frac = (jumps / total_links) if total_links else 1.0

    # Clash-like penalty among nearby residues.
    clashes = 0
    pairs = 0
    for i in range(len(xyz)):
        for j in range(i + 4, min(len(xyz), i + 36)):
            pairs += 1
            if math.dist(xyz[i], xyz[j]) < 2.2:
                clashes += 1
    clash_frac = (clashes / pairs) if pairs else 0.0

    rg_pen = 0.0
    if not math.isnan(rg):
        # Extremely compact or exploded traces are suspicious.
        if rg < 8.0:
            rg_pen = min(1.0, (8.0 - rg) / 8.0)
        elif rg > 42.0:
            rg_pen = min(1.0, (rg - 42.0) / 25.0)
    penalty = max(0.0, min(1.0, 0.5 * jump_frac + 0.3 * clash_frac + 0.2 * rg_pen))
    confidence = max(0.0, min(1.0, 1.0 - penalty))
    return penalty, confidence


def score_sequence_physics(
    wt: str,
    variant: str,
    *,
    protected_indices: Iterable[int] = (),
    structure_pdb: Path | None = None,
    weights: dict[str, float] | None = None,
    use_openmm: bool = False,
    openmm_platform: str = "CPU",
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
        "hydrophobic_core_proxy": 0.28,
        "charge_balance": 0.14,
        "aromatic": 0.08,
        "active_site": 0.23,
        "compactness": 0.13,
        "sasa_burial": 0.14,
        "structure_viability": 0.12,
        "openmm_stability": 0.10,
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
    sasa_total: float | None = None
    apolar_frac: float | None = None
    structure_confidence: float | None = None
    viability_penalty: float | None = None
    openmm_total: float | None = None
    openmm_per_res: float | None = None
    openmm_score = 0.5
    sasa_score = 0.5  # neutral if no structure / no freesasa

    if structure_pdb and structure_pdb.is_file():
        parsed = parse_pdb_ca_coords(structure_pdb)
        xyz = [t[3] for t in parsed]
        rg = radius_of_gyration(xyz)
        if not math.isnan(rg):
            # Smaller Rg → higher score, saturate
            compact_score = 1.0 / (1.0 + rg / 20.0)

        sb = compute_sasa_breakdown(structure_pdb)
        if sb is not None:
            sasa_total = sb.total_area
            apolar_frac = sb.apolar_fraction
            # More exposed apolar SASA → lower burial proxy (penalize high apolar fraction).
            sasa_score = max(0.0, min(1.0, 1.25 - 1.5 * apolar_frac))
        viability_penalty, structure_confidence = structure_viability_penalty(parsed)
        if use_openmm:
            em = minimize_and_score(structure_pdb, openmm_platform)
            if em.get("ok"):
                openmm_total = float(em.get("total_energy_kj_mol"))
                openmm_per_res = float(em.get("energy_per_residue_kj_mol"))
                # Lower (more negative) per-residue energy gets higher bounded score.
                openmm_score = max(0.0, min(1.0, 1.0 / (1.0 + math.exp((openmm_per_res + 8.0) / 12.0))))
            else:
                openmm_score = 0.35

    w_sasa = float(w.get("sasa_burial", 0.0))
    w_viability = float(w.get("structure_viability", 0.0))
    w_openmm = float(w.get("openmm_stability", 0.0))
    conf = 0.5 if structure_confidence is None else structure_confidence
    viol_pen = 0.0 if viability_penalty is None else viability_penalty

    composite = (
        w["hydrophobic_core_proxy"] * hydro_score
        + w["charge_balance"] * charge_score
        + w.get("aromatic", 0.08) * arom_score
        + w["active_site"] * active_score
        + w.get("compactness", 0.13) * compact_score
        + w_sasa * sasa_score
        + w_viability * conf
        + w_openmm * openmm_score
        - 0.15 * viol_pen
        - 0.02 * n_mut
    )

    return PhysicsBreakdown(
        mean_hydrophobicity=mh,
        net_charge_proxy=qc,
        aromatic_fraction=ar,
        active_site_violation=asp,
        radius_of_gyration=rg,
        sasa_total_area=sasa_total,
        apolar_sasa_fraction=apolar_frac,
        structure_confidence=structure_confidence,
        structural_viability_penalty=viability_penalty,
        openmm_total_energy_kj_mol=openmm_total,
        openmm_energy_per_residue_kj_mol=openmm_per_res,
        mutation_count=n_mut,
        composite=float(composite),
    )
