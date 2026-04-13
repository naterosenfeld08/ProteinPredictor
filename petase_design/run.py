#!/usr/bin/env python3
"""
CLI entry: PETase design loop (physics proxy, optional structure hook).

  python -m petase_design.run --cycles 100 --mutations 3 --out runs/petase_batch1.jsonl
"""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path

from petase_design import config
from petase_design.pipeline import run_design_cycles
from petase_design.structure_runner import ColabFoldLocalRunner, NullStructureRunner


def main() -> None:
    ap = argparse.ArgumentParser(description="PETase thermostability design loop (P0)")
    ap.add_argument("--wt-fasta", type=Path, default=config.DEFAULT_WT_FASTA)
    ap.add_argument("--cycles", type=int, default=50)
    ap.add_argument("--mutations", type=int, default=2, help="Random mutations per variant")
    ap.add_argument("--out", type=Path, default=Path("petase_design_runs") / "design_log.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--work-dir",
        type=Path,
        default=Path("petase_design_runs") / "structures",
        help="ColabFold / structure scratch directory",
    )
    ap.add_argument(
        "--colabfold",
        action="store_true",
        help="Run local colabfold_batch for each variant (slow; needs GPU + install)",
    )
    ap.add_argument(
        "--colabfold-bin",
        default="colabfold_batch",
        help="Path or name of colabfold_batch executable",
    )
    ap.add_argument("--num-recycle", type=int, default=3, help="ColabFold --num-recycle")
    ap.add_argument(
        "--structure-top-k",
        type=int,
        default=None,
        help=(
            "Two-stage mode: score all variants sequence-only first, then run ColabFold "
            "for top-K by cheap composite. Ignored unless --colabfold."
        ),
    )
    ap.add_argument(
        "--amber",
        action="store_true",
        help="Pass --amber to colabfold_batch (OpenMM relax; much slower)",
    )
    ap.add_argument(
        "--colabfold-arg",
        action="append",
        default=[],
        metavar="ARG",
        help="Extra argument to colabfold_batch (repeat flag per token)",
    )
    ap.add_argument(
        "--colabfold-extra",
        default="",
        help='Extra args as one shell string, e.g. --colabfold-extra "--max-msa 512:1024"',
    )
    ap.add_argument(
        "--colabfold-overwrite",
        action="store_true",
        help="Pass --overwrite-existing-results to colabfold_batch (re-run even if outputs exist)",
    )
    ap.add_argument("--policy-random-frac", type=float, default=0.50)
    ap.add_argument("--policy-adaptive-frac", type=float, default=0.35)
    ap.add_argument("--policy-recombine-frac", type=float, default=0.15)
    ap.add_argument("--archive-size", type=int, default=24)
    ap.add_argument("--no-pareto-archive", action="store_true")
    ap.add_argument("--openmm-stage", action="store_true")
    ap.add_argument("--openmm-platform", default="CPU")
    ap.add_argument("--protected-indices-json", default="")
    ap.add_argument("--region-budgets-json", default="")
    ap.add_argument(
        "--struct-benchmark-manifest",
        type=Path,
        default=None,
        help="Optional benchmark manifest JSON to score structure-predicted variants against experimental mutants",
    )
    ap.add_argument(
        "--struct-benchmark-include-controls",
        action="store_true",
        help="Also score predicted WT controls for calibration (slower)",
    )
    ap.add_argument("--struct-benchmark-gdt-ts-min", type=float, default=None)
    ap.add_argument("--struct-benchmark-coverage-min", type=float, default=None)
    ap.add_argument(
        "--struct-benchmark-weight",
        type=float,
        default=0.0,
        help="Optional rank bonus weight from benchmark GDT-TS (0 disables blending)",
    )
    args = ap.parse_args()

    if not args.wt_fasta.is_file():
        raise SystemExit(f"WT FASTA not found: {args.wt_fasta}")
    if args.structure_top_k is not None and args.structure_top_k <= 0:
        raise SystemExit("--structure-top-k must be a positive integer.")
    if args.struct_benchmark_weight < 0:
        raise SystemExit("--struct-benchmark-weight must be >= 0.")
    if args.struct_benchmark_gdt_ts_min is not None and not (0.0 <= float(args.struct_benchmark_gdt_ts_min) <= 100.0):
        raise SystemExit("--struct-benchmark-gdt-ts-min must be within [0,100].")
    if args.struct_benchmark_coverage_min is not None and not (0.0 <= float(args.struct_benchmark_coverage_min) <= 100.0):
        raise SystemExit("--struct-benchmark-coverage-min must be within [0,100].")
    if args.struct_benchmark_manifest is not None and not args.struct_benchmark_manifest.is_file():
        raise SystemExit(f"Benchmark manifest not found: {args.struct_benchmark_manifest}")
    protected_indices_override: list[int] | None = None
    if str(args.protected_indices_json).strip():
        payload = json.loads(str(args.protected_indices_json))
        if not isinstance(payload, list):
            raise SystemExit("--protected-indices-json must be a JSON list.")
        protected_indices_override = sorted({int(x) for x in payload})
    region_mutation_budgets: list[tuple[int, int, int]] = []
    if str(args.region_budgets_json).strip():
        payload = json.loads(str(args.region_budgets_json))
        if not isinstance(payload, list):
            raise SystemExit("--region-budgets-json must be a JSON list.")
        for item in payload:
            if not isinstance(item, list) or len(item) != 3:
                raise SystemExit("Each region budget must be [start,end,max_mut].")
            region_mutation_budgets.append((int(item[0]), int(item[1]), int(item[2])))

    runner = None
    if args.colabfold:
        extra = list(args.colabfold_arg)
        if args.colabfold_extra.strip():
            extra.extend(shlex.split(args.colabfold_extra))
        runner = ColabFoldLocalRunner(
            binary=args.colabfold_bin,
            num_recycle=args.num_recycle,
            use_amber=args.amber,
            overwrite_existing=bool(args.colabfold_overwrite),
            extra_args=tuple(extra),
        )
    else:
        runner = NullStructureRunner()

    if args.colabfold:
        print(
            f"ColabFold mode: {args.cycles} job(s). "
            "Progress streams to stderr; this can take a long time on CPU.",
            flush=True,
        )
        if args.structure_top_k is not None:
            print(
                f"Two-stage mode enabled: structure for top {args.structure_top_k} / {args.cycles} variants.",
                flush=True,
            )
    run_design_cycles(
        wt_fasta=args.wt_fasta,
        n_cycles=args.cycles,
        mutations_per_variant=args.mutations,
        out_jsonl=args.out,
        seed=args.seed,
        structure_runner=runner,
        work_root=args.work_dir,
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
        struct_benchmark_manifest=args.struct_benchmark_manifest,
        struct_benchmark_include_controls=bool(args.struct_benchmark_include_controls),
        struct_benchmark_gdt_ts_min=(float(args.struct_benchmark_gdt_ts_min) if args.struct_benchmark_gdt_ts_min is not None else None),
        struct_benchmark_coverage_min=(float(args.struct_benchmark_coverage_min) if args.struct_benchmark_coverage_min is not None else None),
        struct_benchmark_weight=float(args.struct_benchmark_weight),
    )
    print(f"Wrote {args.cycles} variants to {args.out}", flush=True)
    print(f"Run summary: {args.out.parent / 'run_summary.json'}", flush=True)
    print("Tip: fill petase_design/data/active_site_indices_0based.txt to protect catalytic pocket.")


if __name__ == "__main__":
    main()
