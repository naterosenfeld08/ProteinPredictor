#!/usr/bin/env python3
"""
Run PETase design cycles in an isolated process so the Streamlit server stays responsive.

Invoked by ``gui/app.py`` with CLI args; writes JSONL and a full JSON result payload.
Do not import Streamlit here.
"""

from __future__ import annotations

import argparse
import json
import math
import shlex
import sys
import tempfile
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _safe_float(x: object, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except (TypeError, ValueError):
        return default


def _percentile_ranks(values: list[float], *, higher_is_better: bool) -> list[float]:
    """Return 0..1 percentile ranks for numeric values."""
    n = len(values)
    if n == 0:
        return []
    if n == 1:
        return [1.0]
    order = sorted(range(n), key=lambda i: values[i], reverse=higher_is_better)
    out = [0.0] * n
    denom = max(n - 1, 1)
    for rank, idx in enumerate(order):
        out[idx] = 1.0 - (rank / denom)
    return out


def _collect_ddg_survivors(
    rows: list[dict],
    *,
    budget_count: int,
    seed: int,
    lane_a_frac: float = 0.70,
    lane_b_frac: float = 0.20,
    lane_c_frac: float = 0.10,
) -> tuple[set[int], dict[int, str], dict[int, str]]:
    """
    Survivor policy:
      - Lane A: top cheap score (70%)
      - Lane B: diversity lane (20%)
      - Lane C: rescue lane (10%)
    """
    import random

    n = len(rows)
    if n == 0 or budget_count <= 0:
        return set(), {}, {}
    k = min(budget_count, n)
    cheap = [_safe_float((r.get("physics") or {}).get("composite"), default=-1e9) for r in rows]
    ranked = sorted(range(n), key=lambda i: cheap[i], reverse=True)

    k_a = min(int(round(k * lane_a_frac)), k)
    k_b = min(int(round(k * lane_b_frac)), k - k_a)
    k_c = max(0, k - k_a - k_b)

    selected: set[int] = set(ranked[:k_a])
    selected_by: dict[int, str] = {idx: "cheap_top" for idx in selected}
    rescue_reason: dict[int, str] = {}

    # Diversity lane over remaining variants.
    if k_b > 0:
        remaining = [idx for idx in ranked if idx not in selected]
        buckets: dict[int, list[int]] = {}
        for idx in remaining:
            seq = str(rows[idx].get("sequence", ""))
            # Deterministic bucket id (do not use Python's process-randomized hash()).
            b = sum((i + 1) * ord(ch) for i, ch in enumerate(seq)) % 11
            buckets.setdefault(b, []).append(idx)
        picks_b: list[int] = []
        keys = sorted(buckets.keys())
        ptr = {b: 0 for b in keys}
        while len(picks_b) < k_b and keys:
            progressed = False
            for b in list(keys):
                lst = buckets[b]
                i = ptr[b]
                if i < len(lst):
                    cand = lst[i]
                    ptr[b] = i + 1
                    if cand not in selected:
                        picks_b.append(cand)
                        progressed = True
                        if len(picks_b) >= k_b:
                            break
                else:
                    keys.remove(b)
            if not progressed:
                break
        for idx in picks_b:
            selected.add(idx)
            selected_by[idx] = "diversity_lane"

    # Rescue lane from near-threshold cheap region.
    if k_c > 0:
        rng = random.Random(seed + 17)
        top_cut = max(k_a, 1)
        band = [idx for idx in ranked[top_cut : min(len(ranked), top_cut + max(k * 2, 20))] if idx not in selected]
        if len(band) < k_c:
            band.extend(idx for idx in ranked if idx not in selected and idx not in band)
        rng.shuffle(band)
        for idx in band[:k_c]:
            selected.add(idx)
            selected_by[idx] = "rescue_lane"
            rescue_reason[idx] = "near-threshold rescue"

    return selected, selected_by, rescue_reason


def _pareto_layers(
    vectors: list[list[float]],
    maximize: list[bool],
) -> list[int]:
    """Return Pareto rank layer per vector (0 = frontier)."""
    n = len(vectors)
    if n == 0:
        return []

    def dominates(i: int, j: int) -> bool:
        a = vectors[i]
        b = vectors[j]
        ge_all = True
        gt_any = False
        for k, mx in enumerate(maximize):
            va = a[k] if mx else -a[k]
            vb = b[k] if mx else -b[k]
            if va < vb:
                ge_all = False
                break
            if va > vb:
                gt_any = True
        return ge_all and gt_any

    dominates_set: list[set[int]] = [set() for _ in range(n)]
    dominated_count = [0] * n
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if dominates(i, j):
                dominates_set[i].add(j)
            elif dominates(j, i):
                dominated_count[i] += 1

    layers = [-1] * n
    frontier = [i for i in range(n) if dominated_count[i] == 0]
    level = 0
    while frontier:
        next_frontier: list[int] = []
        for i in frontier:
            layers[i] = level
            for j in dominates_set[i]:
                dominated_count[j] -= 1
                if dominated_count[j] == 0:
                    next_frontier.append(j)
        level += 1
        frontier = next_frontier
    for i in range(n):
        if layers[i] < 0:
            layers[i] = level
    return layers


def main() -> int:
    sys.path.insert(0, str(REPO_ROOT))
    from gui.worker_env import configure_worker_runtime_env

    ap = argparse.ArgumentParser(description="PETase design loop (worker process)")
    ap.add_argument("--wt-fasta", type=Path, required=True)
    ap.add_argument("--cycles", type=int, required=True)
    ap.add_argument("--mutations-per-variant", type=int, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-jsonl", type=Path, required=True)
    ap.add_argument("--work-root", type=Path, required=True)
    ap.add_argument("--result-json", required=True, help="Write full rows list as JSON")
    ap.add_argument("--colabfold", action="store_true")
    ap.add_argument("--colabfold-bin", default="colabfold_batch")
    ap.add_argument("--num-recycle", type=int, default=3)
    ap.add_argument(
        "--structure-top-k",
        type=int,
        default=None,
        help="Two-stage mode: structure only for top-K by cheap composite",
    )
    ap.add_argument("--amber", action="store_true")
    ap.add_argument(
        "--colabfold-extra",
        default="",
        help="Extra shell string passed to shlex.split for colabfold_batch",
    )
    ap.add_argument(
        "--colabfold-overwrite",
        action="store_true",
        help="Pass --overwrite-existing-results to colabfold_batch",
    )
    ap.add_argument("--ddg-model", default="", help="Path to trained ddG model (.pkl) for hybrid rerank.")
    ap.add_argument("--ddg-survivor-pct", type=float, default=0.35, help="Fraction of variants passed to ddG stage.")
    ap.add_argument(
        "--ddg-embedding-model-type",
        default="both",
        choices=("both", "prot_t5", "esm2"),
    )
    ap.add_argument(
        "--ddg-no-composition",
        action="store_true",
        help="Disable composition features for ddG stage.",
    )
    ap.add_argument("--hybrid-cheap-weight", type=float, default=0.5)
    ap.add_argument("--hybrid-ddg-weight", type=float, default=0.5)
    ap.add_argument("--ddg-uncertainty-lambda", type=float, default=0.5)
    ap.add_argument("--policy-random-frac", type=float, default=0.50)
    ap.add_argument("--policy-adaptive-frac", type=float, default=0.35)
    ap.add_argument("--policy-recombine-frac", type=float, default=0.15)
    ap.add_argument("--archive-size", type=int, default=24)
    ap.add_argument("--no-pareto-archive", action="store_true")
    ap.add_argument("--openmm-stage", action="store_true")
    ap.add_argument("--openmm-platform", default="CPU")
    ap.add_argument("--objective-ddg-weight", type=float, default=0.35)
    ap.add_argument("--objective-physics-weight", type=float, default=0.25)
    ap.add_argument("--objective-structure-weight", type=float, default=0.15)
    ap.add_argument("--objective-novelty-weight", type=float, default=0.15)
    ap.add_argument("--objective-catalytic-safety-weight", type=float, default=0.10)
    ap.add_argument(
        "--protected-indices-json",
        default="",
        help="Optional JSON list of 0-based protected residue indices.",
    )
    ap.add_argument(
        "--region-budgets-json",
        default="",
        help="Optional JSON list of [start,end,max_mut] mutation budget triplets.",
    )
    args = ap.parse_args()
    if args.structure_top_k is not None and int(args.structure_top_k) <= 0:
        raise SystemExit("--structure-top-k must be a positive integer.")
    if float(args.policy_random_frac) + float(args.policy_adaptive_frac) + float(args.policy_recombine_frac) <= 0:
        raise SystemExit("Policy mix sum must be > 0.")
    if float(args.hybrid_cheap_weight) + float(args.hybrid_ddg_weight) <= 0:
        raise SystemExit("Hybrid cheap/ddG weights must sum to > 0.")
    if (
        float(args.objective_ddg_weight)
        + float(args.objective_physics_weight)
        + float(args.objective_structure_weight)
        + float(args.objective_novelty_weight)
        + float(args.objective_catalytic_safety_weight)
        <= 0
    ):
        raise SystemExit("Objective weight sum must be > 0.")
    if not (0.0 < float(args.ddg_survivor_pct) <= 1.0):
        raise SystemExit("--ddg-survivor-pct must be within (0, 1].")

    # PETase loop is mostly pure Python; env still helps if optional deps spawn threads.
    configure_worker_runtime_env()

    out_result = Path(args.result_json)

    t0 = time.time()
    try:
        from petase_design.pipeline import run_design_cycles
        from petase_design.run_summary import write_run_summary_json
        from petase_design.structure_runner import ColabFoldLocalRunner, NullStructureRunner
        from protein_baseline import predict_from_fasta

        if args.colabfold:
            extra: tuple[str, ...] = ()
            if args.colabfold_extra.strip():
                extra = tuple(shlex.split(args.colabfold_extra))
            runner = ColabFoldLocalRunner(
                binary=args.colabfold_bin,
                num_recycle=int(args.num_recycle),
                use_amber=bool(args.amber),
                overwrite_existing=bool(args.colabfold_overwrite),
                extra_args=extra,
            )
        else:
            runner = NullStructureRunner()

        args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        args.work_root.mkdir(parents=True, exist_ok=True)

        protected_indices_override: list[int] | None = None
        if str(args.protected_indices_json).strip():
            payload = json.loads(str(args.protected_indices_json))
            if not isinstance(payload, list):
                raise ValueError("--protected-indices-json must be a JSON list.")
            protected_indices_override = sorted({int(x) for x in payload})

        region_mutation_budgets: list[tuple[int, int, int]] = []
        if str(args.region_budgets_json).strip():
            payload = json.loads(str(args.region_budgets_json))
            if not isinstance(payload, list):
                raise ValueError("--region-budgets-json must be a JSON list.")
            for item in payload:
                if not isinstance(item, list) or len(item) != 3:
                    raise ValueError("Each region budget must be [start,end,max_mut].")
                s, e, m = int(item[0]), int(item[1]), int(item[2])
                region_mutation_budgets.append((s, e, m))

        rows = run_design_cycles(
            wt_fasta=args.wt_fasta,
            n_cycles=int(args.cycles),
            mutations_per_variant=int(args.mutations_per_variant),
            out_jsonl=args.out_jsonl,
            seed=int(args.seed),
            structure_runner=runner,
            work_root=args.work_root,
            structure_top_k=args.structure_top_k if args.colabfold else None,
            policy_random_frac=float(args.policy_random_frac),
            policy_adaptive_frac=float(args.policy_adaptive_frac),
            policy_recombine_frac=float(args.policy_recombine_frac),
            archive_size=max(4, int(args.archive_size)),
            use_pareto_archive=not bool(args.no_pareto_archive),
            use_openmm=bool(args.openmm_stage),
            openmm_platform=str(args.openmm_platform),
            protected_indices_override=protected_indices_override,
            region_mutation_budgets=region_mutation_budgets,
        )

        # Initialize hybrid fields.
        cheap_vals = [_safe_float((r.get("physics") or {}).get("composite"), default=-1e9) for r in rows]
        cheap_norm = _percentile_ranks(cheap_vals, higher_is_better=True)
        for i, row in enumerate(rows):
            row["cheap_score_norm"] = float(cheap_norm[i])
            row["ddg_pred"] = None
            row["ddg_uncertainty"] = None
            row["ddg_effective"] = None
            row["hybrid_score"] = float(cheap_norm[i])
            row["selected_by"] = "cheap_only"
            row["rescue_reason"] = ""
            row["objective_terms"] = {}
            row["objective_scalar"] = float(cheap_norm[i])
            row["pareto_rank"] = None
            conf = _safe_float((row.get("physics") or {}).get("structure_confidence"), default=1.0)
            row["structure_viable"] = bool(conf >= 0.15)
            if row.get("selected_for_structure") and not row["structure_viable"]:
                row["selected_by"] = "structure_failed"
                row["rescue_reason"] = "low structure confidence"

        ddg_model = Path(args.ddg_model).expanduser() if args.ddg_model else None
        if ddg_model and ddg_model.is_file() and rows:
            pct = max(0.01, min(1.0, float(args.ddg_survivor_pct)))
            budget = max(1, int(round(len(rows) * pct)))
            selected_idx, selected_by, rescue_reason = _collect_ddg_survivors(
                rows,
                budget_count=budget,
                seed=int(args.seed),
            )
            survivors = [
                i
                for i in sorted(selected_idx)
                if str(rows[i].get("sequence", "")).strip() and bool(rows[i].get("structure_viable", True))
            ]

            ddg_map: dict[str, tuple[float, float | None]] = {}
            if survivors:
                fasta_text = "".join(
                    f">{rows[i].get('job_id', f'row_{i}')}\n{rows[i].get('sequence', '')}\n"
                    for i in survivors
                )
                with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False, encoding="utf-8") as tf:
                    tf.write(fasta_text)
                    tf_path = Path(tf.name)
                try:
                    results = predict_from_fasta(
                        fasta_path=str(tf_path),
                        model_path=str(ddg_model),
                        embedding_model_type=str(args.ddg_embedding_model_type),
                        output_path=None,
                        use_composition_features=not bool(args.ddg_no_composition),
                    )
                finally:
                    tf_path.unlink(missing_ok=True)

                preds = (results or {}).get("predictions") or []
                for p in preds:
                    header = str(p.get("header", "")).strip()
                    if not header:
                        continue
                    ddg_map[header] = (
                        _safe_float(p.get("pred_value"), default=0.0),
                        float(p.get("uncertainty")) if p.get("uncertainty") is not None else None,
                    )

            idx_with_ddg: list[int] = []
            ddg_vals: list[float] = []
            unc_vals: list[float] = []
            for i in survivors:
                row = rows[i]
                row["selected_by"] = selected_by.get(i, "ddg_lane")
                row["rescue_reason"] = rescue_reason.get(i, "")
                job_id = str(row.get("job_id", ""))
                if job_id not in ddg_map:
                    continue
                pred, unc = ddg_map[job_id]
                row["ddg_pred"] = float(pred)
                row["ddg_uncertainty"] = float(unc) if unc is not None else None
                idx_with_ddg.append(i)
                ddg_vals.append(float(pred))
                unc_vals.append(float(unc) if unc is not None else 0.0)

            if idx_with_ddg:
                ddg_rank = _percentile_ranks(ddg_vals, higher_is_better=False)  # lower ddG is better
                unc_rank = _percentile_ranks(unc_vals, higher_is_better=False)  # lower uncertainty is better
                cw = max(0.0, float(args.hybrid_cheap_weight))
                dw = max(0.0, float(args.hybrid_ddg_weight))
                total = cw + dw
                if total <= 0:
                    cw, dw, total = 0.5, 0.5, 1.0
                cw /= total
                dw /= total
                lam = max(0.0, float(args.ddg_uncertainty_lambda))
                for j, i in enumerate(idx_with_ddg):
                    ddg_effective = max(0.0, min(1.0, ddg_rank[j] - lam * unc_rank[j]))
                    rows[i]["ddg_effective"] = float(ddg_effective)
                    rows[i]["hybrid_score"] = float(cw * float(rows[i]["cheap_score_norm"]) + dw * ddg_effective)

            for i, row in enumerate(rows):
                if i not in selected_idx:
                    row["selected_by"] = "not_in_ddg_budget"
                    row["rescue_reason"] = ""

        # Multi-objective schema + Pareto ranking (first-class) while preserving hybrid score.
        ddg_eff_vals = [_safe_float(r.get("ddg_effective"), default=0.0) for r in rows]
        phys_vals = [_safe_float((r.get("physics") or {}).get("composite"), default=-1e9) for r in rows]
        struct_vals = [_safe_float((r.get("physics") or {}).get("structure_confidence"), default=0.0) for r in rows]
        nov_vals = [_safe_float(r.get("novelty_score"), default=0.0) for r in rows]
        cat_pen = [
            _safe_float((r.get("physics") or {}).get("active_site_violation"), default=1e9)
            for r in rows
        ]
        phys_rank = _percentile_ranks(phys_vals, higher_is_better=True)
        cat_safety = [1.0 / (1.0 + max(0.0, p)) for p in cat_pen]

        ow_ddg = max(0.0, float(args.objective_ddg_weight))
        ow_phys = max(0.0, float(args.objective_physics_weight))
        ow_struct = max(0.0, float(args.objective_structure_weight))
        ow_nov = max(0.0, float(args.objective_novelty_weight))
        ow_safe = max(0.0, float(args.objective_catalytic_safety_weight))
        o_sum = ow_ddg + ow_phys + ow_struct + ow_nov + ow_safe
        if o_sum <= 0:
            ow_ddg = ow_phys = ow_struct = ow_nov = ow_safe = 0.2
            o_sum = 1.0
        ow_ddg /= o_sum
        ow_phys /= o_sum
        ow_struct /= o_sum
        ow_nov /= o_sum
        ow_safe /= o_sum

        vectors: list[list[float]] = []
        for i, row in enumerate(rows):
            terms = {
                "ddg_effective": float(max(0.0, min(1.0, ddg_eff_vals[i]))),
                "physics_composite_rank": float(max(0.0, min(1.0, phys_rank[i]))),
                "structure_confidence": float(max(0.0, min(1.0, struct_vals[i]))),
                "novelty_score": float(max(0.0, min(1.0, nov_vals[i]))),
                "catalytic_safety_score": float(max(0.0, min(1.0, cat_safety[i]))),
                "catalytic_safety_penalty": float(max(0.0, cat_pen[i])) if math.isfinite(cat_pen[i]) else 1e9,
            }
            scalar = (
                ow_ddg * terms["ddg_effective"]
                + ow_phys * terms["physics_composite_rank"]
                + ow_struct * terms["structure_confidence"]
                + ow_nov * terms["novelty_score"]
                + ow_safe * terms["catalytic_safety_score"]
            )
            row["objective_terms"] = terms
            row["objective_scalar"] = float(scalar)
            vectors.append(
                [
                    terms["ddg_effective"],
                    terms["physics_composite_rank"],
                    terms["structure_confidence"],
                    terms["novelty_score"],
                    terms["catalytic_safety_score"],
                ]
            )

        ranks = _pareto_layers(vectors, maximize=[True, True, True, True, True])
        for i, row in enumerate(rows):
            row["pareto_rank"] = int(ranks[i])

        rows = sorted(
            rows,
            key=lambda r: (
                int(r.get("pareto_rank") if r.get("pareto_rank") is not None else 9999),
                -_safe_float(r.get("objective_scalar"), default=-1e9),
                -_safe_float(r.get("hybrid_score"), default=-1e9),
            ),
        )

        # Keep JSONL and summary aligned with enriched rows.
        with args.out_jsonl.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        write_run_summary_json(
            args.out_jsonl,
            rows,
            t0=t0,
            t1=time.time(),
            meta={
                "wt_fasta": str(args.wt_fasta),
                "n_cycles": int(args.cycles),
                "mutations_per_variant": int(args.mutations_per_variant),
                "seed": int(args.seed),
                "structure_top_k": args.structure_top_k if args.colabfold else None,
                "hybrid_rerank": True,
                "ddg_model": str(ddg_model),
                "ddg_survivor_pct": float(args.ddg_survivor_pct),
                "policy_mix": {
                    "random": float(args.policy_random_frac),
                    "adaptive": float(args.policy_adaptive_frac),
                    "recombine": float(args.policy_recombine_frac),
                },
                "archive_size": int(args.archive_size),
                "use_pareto_archive": not bool(args.no_pareto_archive),
                "objective_weights": {
                    "ddg_effective": ow_ddg,
                    "physics_composite_rank": ow_phys,
                    "structure_confidence": ow_struct,
                    "novelty_score": ow_nov,
                    "catalytic_safety_score": ow_safe,
                },
                "openmm_stage": bool(args.openmm_stage),
                "openmm_platform": str(args.openmm_platform),
                "protected_indices_count": len(protected_indices_override or []),
                "region_mutation_budgets": [
                    {"start": int(s), "end": int(e), "max_mut": int(m)}
                    for s, e, m in region_mutation_budgets
                ],
            },
        )

        out_result.write_text(json.dumps(rows, default=str), encoding="utf-8")
        return 0
    except Exception:
        payload = {"ok": False, "error": traceback.format_exc()}
        out_result.write_text(json.dumps(payload), encoding="utf-8")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
