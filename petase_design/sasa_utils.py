"""
Optional SASA via FreeSASA (install: pip install -r petase_design/requirements-extras.txt).

Used when a PDB path is available; if `freesasa` is missing or calculation fails,
callers should treat metrics as None and skip SASA terms in the composite score.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SasaBreakdown:
    """Å²; apolar_fraction = Apolar / (Polar + Apolar + Unknown) from FreeSASA classifier."""

    total_area: float
    polar_area: float
    apolar_area: float
    unknown_area: float

    @property
    def apolar_fraction(self) -> float:
        denom = self.polar_area + self.apolar_area + self.unknown_area
        if denom <= 0:
            return 0.0
        return self.apolar_area / denom


def compute_sasa_breakdown(pdb_path: Path) -> SasaBreakdown | None:
    """
    Run Lee–Richards (default) SASA on a PDB file.

    Returns None if freesasa is not installed, the file is invalid, or calculation fails.
    """
    try:
        import freesasa
    except ImportError:
        return None

    path = Path(pdb_path)
    if not path.is_file():
        return None

    try:
        structure = freesasa.Structure(str(path))
        result = freesasa.calc(structure)
    except Exception:
        return None

    try:
        areas = freesasa.classifyResults(result, structure)
    except Exception:
        return None

    polar = float(areas.get("Polar", 0.0) or 0.0)
    apolar = float(areas.get("Apolar", 0.0) or 0.0)
    unknown = float(areas.get("Unknown", 0.0) or 0.0)
    total = float(result.totalArea())

    if total <= 0 and polar + apolar + unknown <= 0:
        return None

    return SasaBreakdown(
        total_area=total,
        polar_area=polar,
        apolar_area=apolar,
        unknown_area=unknown,
    )
