from __future__ import annotations

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
DATA_DIR = PACKAGE_DIR / "data"
DEFAULT_WT_FASTA = DATA_DIR / "petase_6eqd_chainA_notag.fasta"

# Optional: copy active_site_indices_0based.example.txt → active_site_indices_0based.txt
ACTIVE_SITE_INDICES_FILE = DATA_DIR / "active_site_indices_0based.txt"

# Shell around active site (Å) for local stability proxies once structures exist
ACTIVE_SITE_SHELL_ANGSTROM = 10.0

# Composite score weights (tunable; keys must match petase_design.physics_score.score_sequence_physics)
WEIGHTS = {
    "hydrophobic_core_proxy": 0.35,
    "charge_balance": 0.15,
    "aromatic": 0.1,
    "active_site": 0.25,
    "compactness": 0.15,
}
