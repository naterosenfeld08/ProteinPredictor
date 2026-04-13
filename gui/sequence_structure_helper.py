"""
Helpers for quick sequence-driven structure visualization in the GUI.

This is intentionally a lightweight visual model generator for demos:
- It identifies whether a sequence matches the bundled PETase WT.
- It can synthesize a pseudo-PDB backbone from sequence so py3Dmol can render it.
"""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
import urllib.request


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

# Curated "common sequence -> known PDB" mappings for the GUI helper.
# These are intended as high-confidence shortcuts for demo/common inputs.
_KNOWN_SEQUENCE_STRUCTURES: tuple[dict[str, str], ...] = (
    {
        "label": "insulin_a_human",
        "name": "Human insulin A-chain",
        "sequence": "GIVEQCCTSICSLYQLENYCN",
        "pdb_id": "4INS",
        "chain": "A",
    },
    {
        "label": "insulin_b_human",
        "name": "Human insulin B-chain",
        "sequence": "FVNQHLCGSHLVEALYLVCGERGFFYTPKT",
        "pdb_id": "4INS",
        "chain": "B",
    },
    {
        "label": "lysozyme_c",
        "name": "Lysozyme C (human, mature/pro-peptide region)",
        "sequence": (
            "KVFERCELARTLKRLGMDGYRGISLANWMCLAKWESGYNTRATNYNAGDRSTDYGIFQINSRYWCNDGKTP"
            "GAVNACHLSCSALLQDNIADAVACAKRVVRDPQGIRAWVAWRNRCQNRDVRQYVQGCGV"
        ),
        "pdb_id": "1LZ1",
        "chain": "A",
    },
)


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


def _sequence_identity(a: str, b: str) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    same = sum(1 for x, y in zip(a, b) if x == y)
    return same / len(a)


def find_known_structure_match(seq: str) -> dict[str, str] | None:
    """
    Return a curated known-structure match for common sequences, if available.

    Match policy:
    - exact full-sequence match, OR
    - known sequence appears as a contiguous region (e.g., precursor includes mature chain), OR
    - equal-length near-exact match (>=95% identity).
    """
    clean = sanitize_sequence(seq)
    if not clean:
        return None
    for item in _KNOWN_SEQUENCE_STRUCTURES:
        known_seq = item["sequence"]
        if clean == known_seq:
            return {
                **item,
                "match_type": "exact",
                "match_detail": f"Exact sequence match to {item['name']}.",
            }
        if known_seq in clean:
            return {
                **item,
                "match_type": "contains_known_region",
                "match_detail": (
                    f"Input contains known {item['name']} sequence region "
                    f"({len(known_seq)} aa)."
                ),
            }
        if len(clean) == len(known_seq):
            ident = _sequence_identity(clean, known_seq)
            if ident >= 0.95:
                return {
                    **item,
                    "match_type": "high_identity",
                    "match_detail": (
                        f"High identity to {item['name']} "
                        f"({ident * 100:.1f}% over {len(clean)} aa)."
                    ),
                }
    return None


@lru_cache(maxsize=32)
def _download_text(url: str) -> str:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "petase-thermostability-benchmark-GUI/1"},
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=25) as resp:
        data = resp.read()
    return data.decode("utf-8", errors="replace")


def _extract_chain_pdb(pdb_text: str, chain_id: str) -> str:
    """Keep only ATOM/HETATM records from one chain (plus terminal records)."""
    ch = (chain_id or "").strip()[:1].upper()
    if not ch:
        return pdb_text
    out: list[str] = []
    for line in pdb_text.splitlines():
        if line.startswith(("ATOM", "HETATM", "TER")):
            record_chain = line[21:22].upper() if len(line) > 21 else ""
            if line.startswith("TER") or record_chain == ch:
                out.append(line)
        elif line.startswith(("HEADER", "TITLE", "COMPND", "SOURCE", "REMARK", "MODEL", "ENDMDL", "END")):
            out.append(line)
    if not any(line.startswith("ATOM") for line in out):
        return pdb_text
    if not out or out[-1] != "END":
        out.append("END")
    return "\n".join(out) + "\n"


def fetch_known_structure_pdb(match: dict[str, str]) -> tuple[str | None, str | None]:
    """
    Download known PDB for a curated match and optionally isolate the chain.

    Returns ``(pdb_text, error_message)``.
    """
    pdb_id = match.get("pdb_id", "").strip().upper()
    chain = match.get("chain", "").strip().upper()
    if not pdb_id:
        return None, "Missing PDB ID for known sequence match."
    urls = (
        f"https://files.rcsb.org/download/{pdb_id}.pdb",
        f"http://files.rcsb.org/download/{pdb_id}.pdb",
    )
    errs: list[str] = []
    pdb_text: str | None = None
    for url in urls:
        try:
            pdb_text = _download_text(url)
            break
        except Exception as exc:  # noqa: BLE001
            errs.append(f"{url} -> {type(exc).__name__}: {exc}")
    if not pdb_text:
        return None, "; ".join(errs[:2]) if errs else "Unknown fetch failure."
    if "ATOM" not in pdb_text and "HETATM" not in pdb_text:
        return None, f"Downloaded {pdb_id} but no ATOM/HETATM records found."
    if chain:
        pdb_text = _extract_chain_pdb(pdb_text, chain)
    return pdb_text, None


def _pdb_atom_line(
    *,
    serial: int,
    atom_name: str,
    resname: str,
    chain: str,
    resseq: int,
    x: float,
    y: float,
    z: float,
    element: str,
) -> str:
    """
    One ATOM record in PDB-like fixed columns.
    Matches common minimal examples parsed reliably by 3Dmol.js.
    """
    ch = (chain or "A")[:1].upper()
    rn = resname.strip().upper()[:3].rjust(3)
    an = atom_name.strip().upper()[:4].rjust(4)
    el = element.strip().upper()[:2].rjust(2)
    return (
        f"ATOM  {serial:5d} {an} {rn} {ch}{resseq:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           {el}"
    )


def build_pseudo_pdb_from_sequence(seq: str, *, chain_id: str = "A") -> str:
    """
    Build a lightweight pseudo-backbone (N/CA/C/O) along a loose helix-like path.
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
        # Minimal peptide-like backbone so cartoon rendering has expected atoms.
        atoms = (
            ("N", x - 1.20, y - 0.10, z - 0.55, "N"),
            ("CA", x, y, z, "C"),
            ("C", x + 1.20, y + 0.10, z + 0.55, "C"),
            ("O", x + 1.65, y + 0.55, z + 1.25, "O"),
        )
        for atom_name, ax, ay, az, elem in atoms:
            lines.append(
                _pdb_atom_line(
                    serial=atom_id,
                    atom_name=atom_name,
                    resname=resname,
                    chain=ch,
                    resseq=i,
                    x=ax,
                    y=ay,
                    z=az,
                    element=elem,
                )
            )
            atom_id += 1
    lines.append("END")
    return "\n".join(lines) + "\n"
