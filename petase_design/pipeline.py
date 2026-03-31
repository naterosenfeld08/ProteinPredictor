"""
PETase design **orchestration**: random mutations, optional two-stage ColabFold, JSONL logging.

Calls :func:`petase_design.physics_score.score_sequence_physics` for each proposal.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from petase_design import config
from petase_design.run_summary import write_run_summary_json
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
    structure_top_k: int | None = None,
) -> list[dict[str, Any]]:
    """
    Propose random variants and score with physics proxy.

    Modes:
      - Default: if structure_runner is ColabFold, each variant gets structure scoring.
      - Two-stage: if structure_top_k is set (>0), do cheap sequence-only scoring first for
        all variants, then run structure prediction only for the top-K by composite.
    """
    t_wall0 = time.time()
    rng = random.Random(seed)
    _, wt = load_fasta_sequence(wt_fasta)
    protected = load_protected_indices()
    runner = structure_runner or NullStructureRunner()
    work_root = work_root or Path("petase_design_runs") / "structures"
    work_root.mkdir(parents=True, exist_ok=True)

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    # Stage 1: cheap sequence-only scoring for every proposal.
    for t in range(n_cycles):
        muts = propose_random_mutations(
            wt, mutations_per_variant, rng=rng, protected_indices=protected
        )
        var = variant_from_mutations(wt, muts)
        bd = score_sequence_physics(
            wt,
            var,
            protected_indices=protected,
            structure_pdb=None,
            weights=dict(config.WEIGHTS),
        )
        row = {
            "generation": t,
            "job_id": f"gen{t:05d}",
            "mutations": [{"index": i, "to": aa} for i, aa in muts],
            "sequence": var,
            "physics": asdict(bd),
            "structure_pdb": None,
            "selected_for_structure": False,
        }
        rows.append(row)

    # Stage 2 (optional): structure only for top-K by cheap composite.
    use_two_stage = structure_top_k is not None and structure_top_k > 0
    if use_two_stage:
        k = min(int(structure_top_k), len(rows))
        ranked = sorted(
            enumerate(rows),
            key=lambda it: float((it[1].get("physics") or {}).get("composite", float("-inf"))),
            reverse=True,
        )
        chosen_idx = {idx for idx, _ in ranked[:k]}
    else:
        chosen_idx = set(range(len(rows)))

    # If runner is NullStructureRunner, leave structure fields null.
    if not isinstance(runner, NullStructureRunner):
        for idx, row in enumerate(rows):
            if idx not in chosen_idx:
                continue
            row["selected_for_structure"] = True
            job_id = str(row["job_id"])
            pdb = runner.predict(str(row["sequence"]), job_id, work_root / job_id)
            bd = score_sequence_physics(
                wt,
                str(row["sequence"]),
                protected_indices=protected,
                structure_pdb=pdb,
                weights=dict(config.WEIGHTS),
            )
            row["physics"] = asdict(bd)
            row["structure_pdb"] = str(pdb) if pdb else None

    with out_jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    t_wall1 = time.time()
    summary_meta = {
        "wt_fasta": str(wt_fasta),
        "n_cycles": n_cycles,
        "mutations_per_variant": mutations_per_variant,
        "seed": seed,
        "structure_top_k": structure_top_k,
    }
    write_run_summary_json(
        out_jsonl,
        rows,
        t0=t_wall0,
        t1=t_wall1,
        meta=summary_meta,
    )

    return rows
