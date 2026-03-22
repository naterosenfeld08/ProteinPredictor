from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

from petase_design import config
from petase_design.mutagenesis import propose_random_mutations, variant_from_mutations
from petase_design.physics_score import score_sequence_physics
from petase_design.sequence_utils import load_fasta_sequence
from petase_design.structure_runner import NullStructureRunner, StructureRunner


def load_protected_indices(path: Path | None = None) -> list[int]:
    p = path or config.ACTIVE_SITE_INDICES_FILE
    if not p.is_file():
        return []
    out: list[int] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        out.append(int(line))
    return out


def run_design_cycles(
    *,
    wt_fasta: Path,
    n_cycles: int,
    mutations_per_variant: int,
    out_jsonl: Path,
    seed: int = 42,
    structure_runner: StructureRunner | None = None,
    work_root: Path | None = None,
) -> list[dict[str, Any]]:
    """
    Propose random variants, score with physics proxy, append one JSON object per line.
    """
    rng = random.Random(seed)
    _, wt = load_fasta_sequence(wt_fasta)
    protected = load_protected_indices()
    runner = structure_runner or NullStructureRunner()
    work_root = work_root or Path("petase_design_runs") / "structures"
    work_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    for t in range(n_cycles):
        muts = propose_random_mutations(
            wt, mutations_per_variant, rng=rng, protected_indices=protected
        )
        var = variant_from_mutations(wt, muts)
        job_id = f"gen{t:05d}"
        pdb = runner.predict(var, job_id, work_root / job_id)

        bd = score_sequence_physics(
            wt,
            var,
            protected_indices=protected,
            structure_pdb=pdb,
            weights=dict(config.WEIGHTS),
        )
        row = {
            "generation": t,
            "mutations": [{"index": i, "to": aa} for i, aa in muts],
            "sequence": var,
            "physics": asdict(bd),
            "structure_pdb": str(pdb) if pdb else None,
        }
        rows.append(row)
        with out_jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    return rows
