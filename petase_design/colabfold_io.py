from __future__ import annotations

"""
Locate ColabFold batch outputs and optionally convert mmCIF → PDB for physics_score.py.
"""

from pathlib import Path


def _is_structure_pdb(path: Path) -> bool:
    n = path.name.lower()
    if "coverage" in n or "log" in n or "readme" in n:
        return False
    return path.suffix.lower() == ".pdb"


def find_ranked_structure_pdb(search_root: Path) -> Path | None:
    """
    Pick the best ColabFold structure file under search_root.

    Preference: ranked_0 pdb → other ranked → unrelaxed → any pdb.
    """
    if not search_root.is_dir():
        return None

    def rglob_sorted(pat: str) -> list[Path]:
        return sorted(p for p in search_root.rglob(pat) if p.is_file() and _is_structure_pdb(p))

    for pat in (
        "*ranked_0*.pdb",
        "*ranked_0.pdb",
        "*relax*ranked_0*.pdb",
        "*ranked*.pdb",
        "*unrelaxed*.pdb",
        "*.pdb",
    ):
        hits = rglob_sorted(pat)
        if hits:
            return hits[0]
    return None


def find_ranked_structure_cif(search_root: Path) -> Path | None:
    if not search_root.is_dir():
        return None
    for pat in ("*ranked_0*.cif", "*ranked_0.cif", "*ranked*.cif", "*.cif"):
        matches = sorted(p for p in search_root.rglob(pat) if p.is_file())
        if matches:
            return matches[0]
    return None


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
