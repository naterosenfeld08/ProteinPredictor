from __future__ import annotations

"""
Locate ColabFold / localcolabfold batch outputs and convert mmCIF → PDB.

ColabFold versions differ: outputs may be ``ranked_0.pdb``, ``*_unrelaxed_*_rank_*.pdb``,
nested subfolders, or mmCIF-only. We collect all plausible structure files under the job
directory, score by filename heuristics, and pick the best match.
"""

import re
from pathlib import Path


def _is_noise_structure_file(path: Path) -> bool:
    """Filter coverage maps, logs, and other non-coordinate PDBs."""
    n = path.name.lower()
    if path.suffix.lower() not in (".pdb", ".cif", ".mmcif"):
        return True
    bad = (
        "coverage",
        "pae",
        "plddt",
        "confidence",
        "log",
        "readme",
        "citation",
        "config",
        ".done",
        "pickle",
        "a3m",
        "hhr",
    )
    return any(b in n for b in bad)


def _pdb_priority(path: Path) -> tuple[int, int, str]:
    """
    Lower tuple sorts earlier (better). Tier guesses ColabFold / AF batch naming.
    """
    n = path.name.lower()
    tier = 50
    # Best: explicit ranked_0 (avoid ranked_00 matching ranked_0 incorrectly)
    if re.search(r"ranked[_-]?0(?:[^0-9]|\.|$)", n) or n.startswith("ranked_0."):
        tier = 0
    elif re.search(r"rank[_-]?0(?:[^0-9]|\.|$)", n):
        tier = 1
    elif re.search(r"ranked[_-]?1(?:[^0-9]|\.|$)", n):
        tier = 2
    elif re.search(r"ranked[_-]?2(?:[^0-9]|\.|$)", n):
        tier = 3
    elif re.search(r"rank[_-]?1(?:[^0-9]|\.|$)", n) and "ranked_10" not in n:
        tier = 4
    elif "ranked" in n and path.suffix.lower() == ".pdb":
        tier = 8
    elif "relaxed" in n and path.suffix.lower() == ".pdb":
        tier = 10
    elif "unrelaxed" in n and path.suffix.lower() == ".pdb":
        tier = 12
    elif path.suffix.lower() == ".pdb":
        tier = 18
    else:
        tier = 40
    return (tier, len(path.parts), n)


def _cif_priority(path: Path) -> tuple[int, int, str]:
    n = path.name.lower()
    tier = 50
    if re.search(r"ranked[_-]?0(?:[^0-9]|\.|$)", n) or n.startswith("ranked_0."):
        tier = 0
    elif re.search(r"ranked[_-]?1(?:[^0-9]|\.|$)", n):
        tier = 2
    elif "ranked" in n:
        tier = 8
    elif path.suffix.lower() in (".cif", ".mmcif"):
        tier = 15
    return (tier, len(path.parts), n)


def _collect_pdb_candidates(search_root: Path) -> list[Path]:
    if not search_root.is_dir():
        return []
    out: list[Path] = []
    for p in search_root.rglob("*.pdb"):
        if p.is_file() and not _is_noise_structure_file(p):
            out.append(p)
    return out


def _collect_cif_candidates(search_root: Path) -> list[Path]:
    if not search_root.is_dir():
        return []
    out: list[Path] = []
    for ext in ("*.cif", "*.mmcif"):
        for p in search_root.rglob(ext):
            if p.is_file() and not _is_noise_structure_file(p):
                out.append(p)
    return out


def find_best_structure_pdb(search_root: Path) -> Path | None:
    """Pick the best PDB under ``search_root`` using ColabFold-style filename heuristics."""
    cands = _collect_pdb_candidates(search_root)
    if not cands:
        return None
    cands.sort(key=_pdb_priority)
    return cands[0]


def find_best_structure_cif(search_root: Path) -> Path | None:
    cands = _collect_cif_candidates(search_root)
    if not cands:
        return None
    cands.sort(key=_cif_priority)
    return cands[0]


def format_structure_discovery_report(search_root: Path, *, limit: int = 35) -> str:
    """
    Human-readable summary for ``colabfold.stderr.log`` when discovery fails or for debugging.
    """
    if not search_root.is_dir():
        return f"[structure discovery] search_root is not a directory: {search_root}\n"

    pdbs = _collect_pdb_candidates(search_root)
    cifs = _collect_cif_candidates(search_root)
    lines = [
        "[structure discovery] scanned (recursive) under:",
        f"  {search_root.resolve()}",
        f"  PDB candidates: {len(pdbs)}  CIF candidates: {len(cifs)}",
    ]
    if pdbs:
        lines.append("  PDB files (first %d, priority order):" % min(len(pdbs), limit))
        sorted_p = sorted(pdbs, key=_pdb_priority)
        for p in sorted_p[:limit]:
            try:
                rel = p.relative_to(search_root)
            except ValueError:
                rel = p
            lines.append(f"    {rel}")
    if cifs:
        lines.append("  CIF/mmCIF files (first %d):" % min(len(cifs), limit))
        sorted_c = sorted(cifs, key=_cif_priority)
        for p in sorted_c[:limit]:
            try:
                rel = p.relative_to(search_root)
            except ValueError:
                rel = p
            lines.append(f"    {rel}")
    if not pdbs and not cifs:
        lines.append("  (no .pdb / .cif structure files found — check ColabFold version & flags)")
    lines.append("")
    return "\n".join(lines)


# Backward-compatible names
def find_ranked_structure_pdb(search_root: Path) -> Path | None:
    return find_best_structure_pdb(search_root)


def find_ranked_structure_cif(search_root: Path) -> Path | None:
    return find_best_structure_cif(search_root)


def cif_to_pdb(cif_path: Path, pdb_out: Path) -> bool:
    """Write first model to PDB using Biopython. Returns False if Biopython missing."""
    try:
        from Bio.PDB import MMCIFParser, PDBIO, Select
    except ImportError:
        return False

    class FirstModelOnly(Select):
        def __init__(self) -> None:
            self._took = False

        def accept_model(self, model):
            if self._took:
                return False
            self._took = True
            return True

    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("cif", str(cif_path))
    pdb_out.parent.mkdir(parents=True, exist_ok=True)
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(pdb_out), FirstModelOnly())
    return pdb_out.is_file()
