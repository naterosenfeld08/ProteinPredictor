from __future__ import annotations

import re
from pathlib import Path

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

# Kyte & Doolittle (1982) — higher = more hydrophobic
KD_HYDROPATHY: dict[str, float] = {
    "A": 1.8,
    "R": -4.5,
    "N": -3.5,
    "D": -3.5,
    "C": 2.5,
    "Q": -3.5,
    "E": -3.5,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "L": 3.8,
    "K": -3.9,
    "M": 1.9,
    "F": 2.8,
    "P": -1.6,
    "S": -0.8,
    "T": -0.7,
    "W": -0.9,
    "Y": -1.3,
    "V": 4.2,
}


def load_fasta_sequence(path: Path) -> tuple[str, str]:
    """Return (header, sequence) from first record in FASTA."""
    text = path.read_text(encoding="utf-8")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise ValueError(f"Empty FASTA: {path}")
    header = lines[0].lstrip(">")
    seq_chunks: list[str] = []
    for ln in lines[1:]:
        if ln.startswith(">"):
            break
        seq_chunks.append(ln.upper())
    seq = "".join(seq_chunks)
    seq = re.sub(r"[^A-Z]", "", seq)
    bad = set(seq) - VALID_AA
    if bad:
        raise ValueError(f"Invalid residues in FASTA: {bad}")
    return header, seq


def apply_mutations(wt: str, mutations: list[tuple[int, str]]) -> str:
    """
    Apply 0-based (index, new_aa) mutations. Raises on invalid index or AA.
    """
    s = list(wt.upper())
    for i, aa in mutations:
        aa = aa.upper()
        if aa not in VALID_AA:
            raise ValueError(f"Invalid AA: {aa}")
        if i < 0 or i >= len(s):
            raise IndexError(f"Mutation index out of range: {i} (len={len(s)})")
        s[i] = aa
    return "".join(s)


def mutation_diff(wt: str, variant: str) -> list[tuple[int, str, str]]:
    """List (index, wt_aa, mut_aa) for differing positions."""
    if len(wt) != len(variant):
        raise ValueError("WT and variant length mismatch")
    out: list[tuple[int, str, str]] = []
    for i, (a, b) in enumerate(zip(wt.upper(), variant.upper())):
        if a != b:
            out.append((i, a, b))
    return out
