#!/usr/bin/env python3
"""
Repeatable multi-seed smoke benchmark for PETase design ranking stability.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path


def _read_json(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Unexpected payload type from worker: {type(payload).__name__}")
    return payload


def _top_job_ids(rows: list[dict], k: int) -> list[str]:
    out: list[str] = []
    for r in rows[:k]:
        out.append(str(r.get("job_id", "")))
    return [x for x in out if x]


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    uni = len(a | b)
    return inter / max(uni, 1)


def run_once(args: argparse.Namespace, seed: int) -> dict:
    repo = Path(args.repo_root).resolve()
    worker = repo / "gui" / "design_worker.py"
    if not worker.is_file():
        raise FileNotFoundError(worker)

    with tempfile.TemporaryDirectory(prefix=f"pp_smoke_{seed}_") as td:
        tmp = Path(td)
        out_jsonl = tmp / "run.jsonl"
        out_result = tmp / "result.json"
        work_root = tmp / "work"
        cmd = [
            sys.executable,
            str(worker),
            "--wt-fasta",
            str(Path(args.wt_fasta).resolve()),
            "--cycles",
            str(int(args.cycles)),
            "--mutations-per-variant",
            str(int(args.mutations_per_variant)),
            "--seed",
            str(int(seed)),
            "--out-jsonl",
            str(out_jsonl),
            "--work-root",
            str(work_root),
            "--result-json",
            str(out_result),
            "--ddg-model",
            str(Path(args.ddg_model).resolve()),
            "--ddg-survivor-pct",
            str(float(args.ddg_survivor_pct)),
            "--ddg-embedding-model-type",
            str(args.ddg_embedding_model_type),
            "--hybrid-cheap-weight",
            str(float(args.hybrid_cheap_weight)),
            "--hybrid-ddg-weight",
            str(float(args.hybrid_ddg_weight)),
            "--ddg-uncertainty-lambda",
            str(float(args.ddg_uncertainty_lambda)),
            "--policy-random-frac",
            str(float(args.policy_random_frac)),
            "--policy-adaptive-frac",
            str(float(args.policy_adaptive_frac)),
            "--policy-recombine-frac",
            str(float(args.policy_recombine_frac)),
            "--archive-size",
            str(int(args.archive_size)),
        ]
        if args.no_pareto_archive:
            cmd.append("--no-pareto-archive")
        if args.openmm_stage:
            cmd.append("--openmm-stage")
            cmd.extend(["--openmm-platform", str(args.openmm_platform)])

        subprocess.run(cmd, cwd=str(repo), check=True)
        rows = _read_json(out_result)
        policy_counts: dict[str, int] = {}
        lane_counts: dict[str, int] = {}
        for r in rows:
            p = str(r.get("generator_policy", "unknown"))
            policy_counts[p] = policy_counts.get(p, 0) + 1
            s = str(r.get("selected_by", "unknown"))
            lane_counts[s] = lane_counts.get(s, 0) + 1
        topk = _top_job_ids(rows, int(args.top_k))
        return {
            "seed": int(seed),
            "n_rows": len(rows),
            "topk": topk,
            "policy_counts": policy_counts,
            "lane_counts": lane_counts,
            "pareto_frontier": sum(1 for r in rows if int(r.get("pareto_rank") or 9999) == 0),
        }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="PETase design multi-seed smoke benchmark")
    ap.add_argument("--repo-root", default=".")
    ap.add_argument(
        "--wt-fasta",
        default="petase_design/data/petase_6eqd_chainA_notag.fasta",
    )
    ap.add_argument("--ddg-model", required=True)
    ap.add_argument("--cycles", type=int, default=50)
    ap.add_argument("--mutations-per-variant", type=int, default=2)
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--ddg-survivor-pct", type=float, default=0.35)
    ap.add_argument("--ddg-embedding-model-type", default="both", choices=("both", "prot_t5", "esm2"))
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
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    seeds = [int(x.strip()) for x in str(args.seeds).split(",") if x.strip()]
    runs = [run_once(args, s) for s in seeds]
    if not runs:
        print("No runs executed.")
        return 1

    print("=== Smoke Benchmark Summary ===")
    for r in runs:
        print(
            f"seed={r['seed']} n={r['n_rows']} frontier={r['pareto_frontier']} "
            f"policy_counts={r['policy_counts']} lane_counts={r['lane_counts']}"
        )

    overlaps: list[float] = []
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            a = set(runs[i]["topk"])
            b = set(runs[j]["topk"])
            overlaps.append(_jaccard(a, b))
    if overlaps:
        print(
            f"top{args.top_k}_jaccard_mean={statistics.mean(overlaps):.3f} "
            f"min={min(overlaps):.3f} max={max(overlaps):.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
