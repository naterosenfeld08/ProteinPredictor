"""
CASP-style structure comparison utilities for WT/mutant benchmarking.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np


AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "MSE": "M",
}


@dataclass
class CAAtom:
    chain_id: str
    residue_id: str
    aa: str
    xyz: tuple[float, float, float]


@dataclass
class StructuralMetrics:
    gdt_ts: float
    gdt_ha: float
    rms_ca: float
    tm_score: float
    p0_5: float
    p1: float
    p2: float
    p4: float
    p8: float
    matched_residues: int
    target_residues: int
    model_residues: int
    percent_coverage: float
    mean_ca_distance_a: float
    median_ca_distance_a: float


def _to_array(coords: list[tuple[float, float, float]]) -> np.ndarray:
    return np.array(coords, dtype=float)


def parse_pdb_ca_atoms(pdb_path: Path, *, chain_id: str | None = None) -> list[CAAtom]:
    atoms: list[CAAtom] = []
    for line in pdb_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("ATOM"):
            continue
        if len(line) < 54:
            continue
        atom_name = line[12:16].strip()
        if atom_name != "CA":
            continue
        chain = (line[21].strip() or "A")
        if chain_id and chain != chain_id:
            continue
        resname = line[17:20].strip().upper()
        aa = AA3_TO_1.get(resname, "X")
        try:
            resseq = int(line[22:26])
        except ValueError:
            continue
        icode = line[26].strip()
        residue_id = f"{resseq}{icode}" if icode else str(resseq)
        try:
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
        except ValueError:
            continue
        atoms.append(CAAtom(chain_id=chain, residue_id=residue_id, aa=aa, xyz=(x, y, z)))
    return atoms


def write_chain_only_pdb(in_path: Path, out_path: Path, chain_id: str) -> None:
    lines_out: list[str] = []
    for line in in_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith(("ATOM", "HETATM")):
            chain = (line[21].strip() or "A")
            if chain == chain_id:
                lines_out.append(line)
        elif line.startswith(("TER", "END")):
            lines_out.append(line)
    if not lines_out:
        raise ValueError(f"No atoms found for chain '{chain_id}' in {in_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines_out) + "\n", encoding="utf-8")


def _global_alignment_pairs(seq_a: str, seq_b: str) -> list[tuple[int, int]]:
    """
    Needleman-Wunsch alignment; returns index pairs of aligned residues.
    """
    match = 2
    mismatch = -1
    gap = -2
    n = len(seq_a)
    m = len(seq_b)
    score = [[0] * (m + 1) for _ in range(n + 1)]
    trace = [[0] * (m + 1) for _ in range(n + 1)]  # 0=diag 1=up 2=left

    for i in range(1, n + 1):
        score[i][0] = i * gap
        trace[i][0] = 1
    for j in range(1, m + 1):
        score[0][j] = j * gap
        trace[0][j] = 2

    for i in range(1, n + 1):
        ai = seq_a[i - 1]
        for j in range(1, m + 1):
            bj = seq_b[j - 1]
            s_diag = score[i - 1][j - 1] + (match if ai == bj else mismatch)
            s_up = score[i - 1][j] + gap
            s_left = score[i][j - 1] + gap
            best = s_diag
            t = 0
            if s_up > best:
                best = s_up
                t = 1
            if s_left > best:
                best = s_left
                t = 2
            score[i][j] = best
            trace[i][j] = t

    i, j = n, m
    pairs_rev: list[tuple[int, int]] = []
    while i > 0 or j > 0:
        t = trace[i][j]
        if i > 0 and j > 0 and t == 0:
            pairs_rev.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif i > 0 and (j == 0 or t == 1):
            i -= 1
        else:
            j -= 1
    pairs_rev.reverse()
    return pairs_rev


def _kabsch_superpose(model_xyz: np.ndarray, target_xyz: np.ndarray) -> np.ndarray:
    m_centroid = model_xyz.mean(axis=0)
    t_centroid = target_xyz.mean(axis=0)
    m0 = model_xyz - m_centroid
    t0 = target_xyz - t_centroid
    cov = m0.T @ t0
    u, _s, vt = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    diag = np.diag([1.0, 1.0, d])
    rot = vt.T @ diag @ u.T
    return (m0 @ rot) + t_centroid


def _gdt_percent(distances: np.ndarray, cutoff: float) -> float:
    if distances.size == 0:
        return 0.0
    return 100.0 * float(np.sum(distances <= cutoff)) / float(distances.size)


def _tm_score(distances: np.ndarray, l_target: int) -> float:
    if distances.size == 0 or l_target <= 0:
        return 0.0
    l = max(l_target, 16)
    d0 = max(0.5, 1.24 * ((l - 15) ** (1.0 / 3.0)) - 1.8)
    score = np.sum(1.0 / (1.0 + (distances / d0) ** 2)) / float(l_target)
    return float(score)


def compare_structures_ca(
    model_pdb: Path,
    target_pdb: Path,
    *,
    model_chain: str | None = None,
    target_chain: str | None = None,
) -> StructuralMetrics:
    """
    CASP-style C-alpha comparison with sequence-dependent alignment.
    """
    model_atoms = parse_pdb_ca_atoms(model_pdb, chain_id=model_chain)
    target_atoms = parse_pdb_ca_atoms(target_pdb, chain_id=target_chain)
    if len(model_atoms) < 10 or len(target_atoms) < 10:
        raise ValueError("Not enough CA atoms for robust comparison.")

    model_seq = "".join(a.aa for a in model_atoms)
    target_seq = "".join(a.aa for a in target_atoms)
    aligned_pairs = _global_alignment_pairs(model_seq, target_seq)
    filtered_pairs = [(i, j) for i, j in aligned_pairs if model_atoms[i].aa != "X" and target_atoms[j].aa != "X"]
    if len(filtered_pairs) < 10:
        raise ValueError("Insufficient aligned residue pairs after sequence mapping.")

    model_xyz = _to_array([model_atoms[i].xyz for i, _j in filtered_pairs])
    target_xyz = _to_array([target_atoms[j].xyz for _i, j in filtered_pairs])
    model_aligned = _kabsch_superpose(model_xyz, target_xyz)
    distances = np.linalg.norm(model_aligned - target_xyz, axis=1)

    p05 = _gdt_percent(distances, 0.5)
    p1 = _gdt_percent(distances, 1.0)
    p2 = _gdt_percent(distances, 2.0)
    p4 = _gdt_percent(distances, 4.0)
    p8 = _gdt_percent(distances, 8.0)
    gdt_ha = (p05 + p1 + p2 + p4) / 4.0
    gdt_ts = (p1 + p2 + p4 + p8) / 4.0
    rms_ca = float(np.sqrt(np.mean(distances**2)))
    tm = _tm_score(distances, len(target_atoms))
    matched = int(distances.size)
    coverage = 100.0 * matched / max(len(target_atoms), 1)

    return StructuralMetrics(
        gdt_ts=float(gdt_ts),
        gdt_ha=float(gdt_ha),
        rms_ca=rms_ca,
        tm_score=tm,
        p0_5=float(p05),
        p1=float(p1),
        p2=float(p2),
        p4=float(p4),
        p8=float(p8),
        matched_residues=matched,
        target_residues=len(target_atoms),
        model_residues=len(model_atoms),
        percent_coverage=float(coverage),
        mean_ca_distance_a=float(np.mean(distances)),
        median_ca_distance_a=float(np.median(distances)),
    )


def metrics_to_dict(metrics: StructuralMetrics) -> dict[str, Any]:
    return asdict(metrics)
