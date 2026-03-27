"""
Helpers for quick sequence-driven structure visualization in the GUI.

This is intentionally a lightweight visual model generator for demos:
- It identifies whether a sequence matches the bundled PETase WT.
- It can synthesize a pseudo-PDB (CA trace) from sequence so py3Dmol can render it.
"""

from __future__ import annotations

import math
from pathlib import Path


AA1_TO_AA3 = {
    "A": "ALA",
    "C": "CYS",
    "D": "ASP",
    "E": "GLU",
    "F": "PHE",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "K": "LYS",
    "L": "LEU",
    "M": "MET",
    "N": "ASN",
    "P": "PRO",
    "Q": "GLN",
    "R": "ARG",
    "S": "SER",
    "T": "THR",
    "V": "VAL",
    "W": "TRP",
    "Y": "TYR",
}


def sanitize_sequence(seq: str) -> str:
    return "".join(ch for ch in seq.upper() if ch.isalpha())


def load_fasta_sequence(path: Path) -> str | None:
    if not path.is_file():
        return None
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    seq = "".join(line.strip() for line in lines if line and not line.startswith(">"))
    return sanitize_sequence(seq)


def identify_sequence(seq: str, *, petase_wt_fasta: Path) -> dict[str, str]:
    clean = sanitize_sequence(seq)
    wt = load_fasta_sequence(petase_wt_fasta)
    if not clean:
        return {"label": "empty", "detail": "No amino acids provided."}
    if wt and clean == wt:
        return {"label": "petase_wt_exact", "detail": "Exact match to bundled PETase WT FASTA."}
    if wt and len(clean) == len(wt):
        same = sum(1 for a, b in zip(clean, wt) if a == b)
        ident = 100.0 * same / max(len(wt), 1)
        return {"label": "petase_like", "detail": f"Length match to PETase WT; identity ~{ident:.1f}%."}
    return {"label": "custom_sequence", "detail": f"Custom sequence ({len(clean)} aa)."}


def build_pseudo_pdb_from_sequence(seq: str, *, chain_id: str = "A") -> str:
    """
    Build a simple C-alpha trace in a loose helix-like path.
    This is for visual storytelling only, not structural accuracy.
    """
    clean = sanitize_sequence(seq)
    if not clean:
        return ""
    lines: list[str] = []
    atom_id = 1
    radius = 8.0
    rise = 1.5
    for i, aa in enumerate(clean, start=1):
        theta = i * 1.7
        x = radius * math.cos(theta)
        y = radius * math.sin(theta)
        z = rise * i
        resname = AA1_TO_AA3.get(aa, "GLY")
        lines.append(
            f"ATOM  {atom_id:5d}  CA  {resname:>3s} {chain_id}{i:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           C"
        )
        atom_id += 1
    lines.append("END")
    return "\n".join(lines) + "\n"
