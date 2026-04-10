"""
PETase design orchestration with optional policy mixing and Pareto archive guidance.
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
from petase_design.mutagenesis import (
    propose_random_mutations,
    propose_recombined_variant,
    propose_weighted_mutations,
    variant_from_mutations,
)
from petase_design.physics_score import score_sequence_physics
from petase_design.sequence_utils import load_fasta_sequence, mutation_diff
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
    policy_random_frac: float = 0.5,
    policy_adaptive_frac: float = 0.35,
    policy_recombine_frac: float = 0.15,
    archive_size: int = 24,
    use_pareto_archive: bool = True,
    use_openmm: bool = False,
    openmm_platform: str = "CPU",
    protected_indices_override: list[int] | None = None,
    region_mutation_budgets: list[tuple[int, int, int]] | None = None,
) -> list[dict[str, Any]]:
    """Generate, score, and optionally structure-rerank PETase variants."""
    t_wall0 = time.time()
    rng = random.Random(seed)
    _, wt = load_fasta_sequence(wt_fasta)
    protected = (
        sorted({int(i) for i in (protected_indices_override or [])})
        if protected_indices_override is not None
        else load_protected_indices()
    )
    runner = structure_runner or NullStructureRunner()
    work_root = work_root or Path("petase_design_runs") / "structures"
    work_root.mkdir(parents=True, exist_ok=True)

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    # Policy and archive state.
    policy_random_frac = max(0.0, float(policy_random_frac))
    policy_adaptive_frac = max(0.0, float(policy_adaptive_frac))
    policy_recombine_frac = max(0.0, float(policy_recombine_frac))
    total_mix = policy_random_frac + policy_adaptive_frac + policy_recombine_frac
    if total_mix <= 0:
        policy_random_frac, policy_adaptive_frac, policy_recombine_frac, total_mix = 1.0, 0.0, 0.0, 1.0
    policy_random_frac /= total_mix
    policy_adaptive_frac /= total_mix
    policy_recombine_frac /= total_mix
    position_weights = [1.0 for _ in wt]
    archive_ids: list[str] = []
    archive_sequences: dict[str, str] = {}
    seen_sequences: set[str] = {wt}
    region_mutation_budgets = list(region_mutation_budgets or [])

    def _safe_float(x: object, default: float = 0.0) -> float:
        try:
            if x is None:
                return default
            return float(x)
        except (TypeError, ValueError):
            return default

    def _novelty(seq: str) -> float:
        mut_frac = len(mutation_diff(wt, seq)) / max(len(wt), 1)
        # Penalize repeated proposals; reward broader exploration.
        duplicate_pen = 0.35 if seq in seen_sequences else 0.0
        return max(0.0, min(1.0, mut_frac * 4.0 - duplicate_pen))

    def _dominates(a: dict[str, float], b: dict[str, float]) -> bool:
        # maximize composite + novelty; minimize active_site_violation
        better_or_eq = (
            a["composite"] >= b["composite"]
            and a["novelty"] >= b["novelty"]
            and a["active_site_violation"] <= b["active_site_violation"]
        )
        strictly_better = (
            a["composite"] > b["composite"]
            or a["novelty"] > b["novelty"]
            or a["active_site_violation"] < b["active_site_violation"]
        )
        return better_or_eq and strictly_better

    def _compute_archive(current_rows: list[dict[str, Any]]) -> list[str]:
        if not current_rows:
            return []
        if not use_pareto_archive:
            ranked = sorted(
                current_rows,
                key=lambda r: _safe_float((r.get("physics") or {}).get("composite"), default=-1e9),
                reverse=True,
            )
            return [str(r.get("job_id", "")) for r in ranked[: max(1, archive_size)] if r.get("job_id")]

        feats: list[dict[str, float]] = []
        for row in current_rows:
            phys = row.get("physics") or {}
            feats.append(
                {
                    "composite": _safe_float(phys.get("composite"), default=-1e9),
                    "novelty": _safe_float(row.get("novelty_score"), default=0.0),
                    "active_site_violation": _safe_float(phys.get("active_site_violation"), default=1e9),
                }
            )
        dominated = [False] * len(current_rows)
        for i in range(len(current_rows)):
            if dominated[i]:
                continue
            for j in range(len(current_rows)):
                if i == j:
                    continue
                if _dominates(feats[j], feats[i]):
                    dominated[i] = True
                    break
        frontier = [idx for idx, is_dom in enumerate(dominated) if not is_dom]
        frontier.sort(
            key=lambda idx: (
                feats[idx]["active_site_violation"],
                -feats[idx]["composite"],
                -feats[idx]["novelty"],
            )
        )
        if len(frontier) > archive_size:
            frontier = frontier[:archive_size]
        return [
            str(current_rows[idx].get("job_id", ""))
            for idx in frontier
            if current_rows[idx].get("job_id")
        ]

    def _sample_parent_sequence() -> tuple[str, str | None]:
        if not archive_ids:
            return wt, None
        # Slightly biased toward top archive members while keeping exploration.
        ranks = list(range(1, len(archive_ids) + 1))
        weights = [1.0 / r for r in ranks]
        pick = rng.choices(archive_ids, weights=weights, k=1)[0]
        seq = archive_sequences.get(pick, wt)
        return seq, pick

    def _enforce_region_budgets(seq: str) -> tuple[str, int]:
        if not region_mutation_budgets:
            return seq, 0
        arr = list(seq)
        reverted = 0
        for start, end, max_mut in region_mutation_budgets:
            s = max(0, min(int(start), len(wt) - 1))
            e = max(s, min(int(end), len(wt) - 1))
            m = max(0, int(max_mut))
            changed = [i for i in range(s, e + 1) if arr[i] != wt[i]]
            if len(changed) <= m:
                continue
            rng.shuffle(changed)
            for i in changed[m:]:
                arr[i] = wt[i]
                reverted += 1
        return "".join(arr), reverted

    # Stage 1: cheap scoring with mixed generation policies.
    for t in range(n_cycles):
        roll = rng.random()
        policy = "random"
        parent_a_seq, parent_a_id = _sample_parent_sequence()
        parent_b_seq, parent_b_id = wt, None
        var = wt
        muts: list[tuple[int, str]] = []

        if (
            roll >= policy_random_frac + policy_adaptive_frac
            and len(archive_ids) >= 2
        ):
            policy = "recombine"
            pick_a = rng.choice(archive_ids)
            pick_b = rng.choice([x for x in archive_ids if x != pick_a])
            parent_a_seq = archive_sequences.get(pick_a, wt)
            parent_b_seq = archive_sequences.get(pick_b, wt)
            parent_a_id, parent_b_id = pick_a, pick_b
            var, _ = propose_recombined_variant(
                wt,
                parent_a_seq,
                parent_b_seq,
                int(mutations_per_variant),
                rng=rng,
                protected_indices=protected,
            )
            muts = [(i, aa_mut) for i, _aa_wt, aa_mut in mutation_diff(wt, var)]
        elif roll >= policy_random_frac:
            policy = "adaptive"
            muts = propose_weighted_mutations(
                parent_a_seq,
                int(mutations_per_variant),
                rng=rng,
                protected_indices=protected,
                position_weights=position_weights,
            )
            var = variant_from_mutations(parent_a_seq, muts)
        else:
            muts = propose_random_mutations(
                parent_a_seq,
                int(mutations_per_variant),
                rng=rng,
                protected_indices=protected,
            )
            var = variant_from_mutations(parent_a_seq, muts)

        var, reverted_by_budget = _enforce_region_budgets(var)
        if reverted_by_budget > 0:
            muts = [(i, aa_mut) for i, _aa_wt, aa_mut in mutation_diff(wt, var)]

        bd = score_sequence_physics(
            wt,
            var,
            protected_indices=protected,
            structure_pdb=None,
            weights=dict(config.WEIGHTS),
            use_openmm=False,
            openmm_platform=openmm_platform,
        )
        row = {
            "generation": t,
            "job_id": f"gen{t:05d}",
            "generator_policy": policy,
            "parent_ids": [x for x in (parent_a_id, parent_b_id) if x],
            "mutations": [{"index": i, "to": aa} for i, aa in muts],
            "sequence": var,
            "physics": asdict(bd),
            "novelty_score": _novelty(var),
            "structure_pdb": None,
            "selected_for_structure": False,
            "archive_member": False,
            "constraint_reverts": int(reverted_by_budget),
        }
        rows.append(row)
        seen_sequences.add(var)
        archive_sequences[str(row["job_id"])] = var

        # Adapt position priors from strong early winners.
        comp = _safe_float((row.get("physics") or {}).get("composite"), default=0.0)
        if comp > 0:
            for i, _aa_wt, _aa_mut in mutation_diff(wt, var):
                if 0 <= i < len(position_weights):
                    position_weights[i] += 0.03 + min(comp, 2.0) * 0.02
        if (t + 1) % 8 == 0:
            archive_ids = _compute_archive(rows)
            for r in rows:
                r["archive_member"] = str(r.get("job_id")) in set(archive_ids)

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
                use_openmm=use_openmm,
                openmm_platform=openmm_platform,
            )
            row["physics"] = asdict(bd)
            row["structure_pdb"] = str(pdb) if pdb else None
    archive_ids = _compute_archive(rows)
    archive_set = set(archive_ids)
    for r in rows:
        r["archive_member"] = str(r.get("job_id", "")) in archive_set

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
            "policy_mix": {
                "random": policy_random_frac,
                "adaptive": policy_adaptive_frac,
                "recombine": policy_recombine_frac,
            },
            "archive_size": int(archive_size),
            "use_pareto_archive": bool(use_pareto_archive),
            "protected_indices_count": len(protected),
            "region_mutation_budgets": [
                {"start": int(s), "end": int(e), "max_mut": int(m)}
                for s, e, m in region_mutation_budgets
            ],
    }
    write_run_summary_json(
        out_jsonl,
        rows,
        t0=t_wall0,
        t1=t_wall1,
        meta=summary_meta,
    )

    return rows
