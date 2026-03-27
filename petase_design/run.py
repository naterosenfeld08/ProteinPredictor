#!/usr/bin/env python3
"""
CLI entry: PETase design loop (physics proxy, optional structure hook).

  python -m petase_design.run --cycles 100 --mutations 3 --out runs/petase_batch1.jsonl
"""

from __future__ import annotations

import argparse
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
    args = ap.parse_args()

    if not args.wt_fasta.is_file():
        raise SystemExit(f"WT FASTA not found: {args.wt_fasta}")
    if args.structure_top_k is not None and args.structure_top_k <= 0:
        raise SystemExit("--structure-top-k must be a positive integer.")

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
    )
    print(f"Wrote {args.cycles} variants to {args.out}", flush=True)
    print(f"Run summary: {args.out.parent / 'run_summary.json'}", flush=True)
    print("Tip: fill petase_design/data/active_site_indices_0based.txt to protect catalytic pocket.")


if __name__ == "__main__":
    main()
