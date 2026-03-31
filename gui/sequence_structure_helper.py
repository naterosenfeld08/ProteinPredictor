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
    """Keep letters only, uppercased (for loose user input)."""
    return "".join(ch for ch in seq.upper() if ch.isalpha())


def load_fasta_sequence(path: Path) -> str | None:
    """Return first FASTA sequence as a string, or ``None`` if the file is missing or empty."""
    if not path.is_file():
        return None
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    seq = "".join(line.strip() for line in lines if line and not line.startswith(">"))
    return sanitize_sequence(seq)


def identify_sequence(seq: str, *, petase_wt_fasta: Path) -> dict[str, str]:
    """
    Compare ``seq`` to the bundled PETase WT (length / identity) for UI labeling.

    Returns keys ``label`` (``petase_wt_exact`` | ``petase_like`` | ``custom_sequence`` | …)
    and ``detail`` (human-readable explanation).
    """
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


def _pdb_atom_line(
    *,
    serial: int,
    resname: str,
    chain: str,
    resseq: int,
    x: float,
    y: float,
    z: float,
) -> str:
    """
    One ATOM record (Cα only) in PDB-like fixed columns.
    Matches common minimal examples parsed reliably by 3Dmol.js.
    """
    ch = (chain or "A")[:1].upper()
    rn = resname.strip().upper()[:3].rjust(3)
    return (
        f"ATOM  {serial:5d}  CA  {rn} {ch}{resseq:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           C"
    )


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
    ch = (chain_id or "A")[:1]
    for i, aa in enumerate(clean, start=1):
        theta = i * 1.7
        x = radius * math.cos(theta)
        y = radius * math.sin(theta)
        z = rise * i
        resname = AA1_TO_AA3.get(aa, "GLY")
        lines.append(
            _pdb_atom_line(
                serial=atom_id,
                resname=resname,
                chain=ch,
                resseq=i,
                x=x,
                y=y,
                z=z,
            )
        )
        atom_id += 1
    lines.append("END")
    return "\n".join(lines) + "\n"
