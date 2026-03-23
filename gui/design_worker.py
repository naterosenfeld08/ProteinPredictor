#!/usr/bin/env python3
"""
Run PETase design cycles in an isolated process (for Streamlit GUI).
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


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
    ap.add_argument("--amber", action="store_true")
    ap.add_argument(
        "--colabfold-extra",
        default="",
        help="Extra shell string passed to shlex.split for colabfold_batch",
    )
    args = ap.parse_args()

    # PETase loop is mostly pure Python; env still helps if optional deps spawn threads.
    configure_worker_runtime_env()

    out_result = Path(args.result_json)

    try:
        from petase_design.pipeline import run_design_cycles
        from petase_design.structure_runner import ColabFoldLocalRunner, NullStructureRunner

        if args.colabfold:
            extra: tuple[str, ...] = ()
            if args.colabfold_extra.strip():
                extra = tuple(shlex.split(args.colabfold_extra))
            runner = ColabFoldLocalRunner(
                binary=args.colabfold_bin,
                num_recycle=int(args.num_recycle),
                use_amber=bool(args.amber),
                extra_args=extra,
            )
        else:
            runner = NullStructureRunner()

        args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        args.work_root.mkdir(parents=True, exist_ok=True)

        rows = run_design_cycles(
            wt_fasta=args.wt_fasta,
            n_cycles=int(args.cycles),
            mutations_per_variant=int(args.mutations_per_variant),
            out_jsonl=args.out_jsonl,
            seed=int(args.seed),
            structure_runner=runner,
            work_root=args.work_root,
        )
        out_result.write_text(json.dumps(rows, default=str), encoding="utf-8")
        return 0
    except Exception:
        payload = {"ok": False, "error": traceback.format_exc()}
        out_result.write_text(json.dumps(payload), encoding="utf-8")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
