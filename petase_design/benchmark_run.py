#!/usr/bin/env python3
"""
Automated WT/mutant structural benchmark runner.

Usage:
  python -m petase_design.benchmark_run --colabfold --max-enzymes 3
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import shlex
from typing import Any
from urllib.request import urlretrieve

from petase_design.struct_benchmark_discovery import (
    BenchmarkPair,
    DiscoveryReport,
    discover_wt_mutant_pairs,
    write_discovery_manifest,
)
from petase_design.struct_benchmark_metrics import (
    build_calibration_profile,
    compare_structures_ca,
    metrics_to_dict,
    write_chain_only_pdb,
)
from petase_design.structure_runner import ColabFoldLocalRunner, NullStructureRunner, StructureRunner


def _utc_now() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _download_pdb(pdb_id: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{pdb_id.lower()}.pdb"
    if out_path.is_file():
        return out_path
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    urlretrieve(url, out_path)
    return out_path


def _score_pair(
    *,
    pair: BenchmarkPair,
    runner: StructureRunner,
    exp_dir: Path,
    pred_dir: Path,
    include_controls: bool,
    wt_pred_cache: dict[str, Path],
    gdt_ts_min: float | None,
    coverage_min: float | None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "job_id": f"{pair.enzyme_key}:{pair.wt_pdb_id}:{pair.mut_pdb_id}:{pair.mutation_code}",
        "generated_at": _utc_now(),
        "status": "ok",
        "enzyme_key": pair.enzyme_key,
        "enzyme_label": pair.enzyme_label,
        "uniprot_id": pair.uniprot_id,
        "mutation_code": pair.mutation_code,
        "mutation_raw": pair.mutation_raw,
        "seq_identity": pair.seq_identity,
        "wt": {
            "pdb_id": pair.wt_pdb_id,
            "entity_id": pair.wt_entity_id,
            "chain_id": pair.wt_chain_id,
            "resolution_a": pair.wt_resolution_a,
        },
        "mutant": {
            "pdb_id": pair.mut_pdb_id,
            "entity_id": pair.mut_entity_id,
            "chain_id": pair.mut_chain_id,
            "resolution_a": pair.mut_resolution_a,
        },
    }
    metrics: dict[str, Any] = {}
    row["metrics"] = metrics

    try:
        wt_full = _download_pdb(pair.wt_pdb_id, exp_dir)
        mut_full = _download_pdb(pair.mut_pdb_id, exp_dir)

        wt_chain = exp_dir / f"{pair.wt_pdb_id.lower()}_{pair.wt_chain_id}.pdb"
        mut_chain = exp_dir / f"{pair.mut_pdb_id.lower()}_{pair.mut_chain_id}.pdb"
        if not wt_chain.is_file():
            write_chain_only_pdb(wt_full, wt_chain, pair.wt_chain_id)
        if not mut_chain.is_file():
            write_chain_only_pdb(mut_full, mut_chain, pair.mut_chain_id)

        pair_pred_dir = pred_dir / f"{pair.wt_pdb_id.lower()}_{pair.mut_pdb_id.lower()}_{pair.mutation_code}"
        pred_mut = runner.predict(pair.mut_sequence, f"mut_{pair.mut_pdb_id.lower()}", pair_pred_dir / "mutant")
        row["paths"] = {
            "experimental_wt_pdb": str(wt_chain),
            "experimental_mutant_pdb": str(mut_chain),
            "predicted_mutant_pdb": str(pred_mut) if pred_mut else None,
        }
        if pred_mut is None:
            raise RuntimeError("Prediction failed for mutant sequence.")

        m_main = compare_structures_ca(pred_mut, mut_chain, model_chain=None, target_chain=None)
        metrics["predicted_mutant_vs_experimental_mutant"] = metrics_to_dict(m_main)
        metrics["experimental_wt_vs_experimental_mutant"] = metrics_to_dict(
            compare_structures_ca(wt_chain, mut_chain, model_chain=None, target_chain=None)
        )

        if include_controls:
            wt_cache_key = f"{pair.wt_pdb_id}:{pair.wt_chain_id}"
            if wt_cache_key not in wt_pred_cache:
                wt_pred = runner.predict(pair.wt_sequence, f"wt_{pair.wt_pdb_id.lower()}", pair_pred_dir / "wt")
                if wt_pred is None:
                    raise RuntimeError("Prediction failed for WT control sequence.")
                wt_pred_cache[wt_cache_key] = wt_pred
            wt_pred_path = wt_pred_cache[wt_cache_key]
            row["paths"]["predicted_wt_pdb"] = str(wt_pred_path)

            metrics["predicted_wt_vs_experimental_wt"] = metrics_to_dict(
                compare_structures_ca(wt_pred_path, wt_chain, model_chain=None, target_chain=None)
            )
            metrics["predicted_wt_vs_experimental_mutant"] = metrics_to_dict(
                compare_structures_ca(wt_pred_path, mut_chain, model_chain=None, target_chain=None)
            )
        row["calibration"] = build_calibration_profile(
            metrics,
            gdt_ts_min=gdt_ts_min,
            coverage_min=coverage_min,
        )
    except Exception as e:
        row["status"] = "error"
        row["error"] = str(e)
    return row


def _flatten_row_for_csv(row: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {
        "job_id": row.get("job_id"),
        "status": row.get("status"),
        "enzyme_key": row.get("enzyme_key"),
        "enzyme_label": row.get("enzyme_label"),
        "uniprot_id": row.get("uniprot_id"),
        "mutation_code": row.get("mutation_code"),
        "wt_pdb_id": ((row.get("wt") or {}).get("pdb_id")),
        "mut_pdb_id": ((row.get("mutant") or {}).get("pdb_id")),
        "wt_chain_id": ((row.get("wt") or {}).get("chain_id")),
        "mut_chain_id": ((row.get("mutant") or {}).get("chain_id")),
        "seq_identity": row.get("seq_identity"),
        "error": row.get("error"),
    }
    metrics = row.get("metrics") or {}
    for block_name, vals in metrics.items():
        if isinstance(vals, dict):
            for k, v in vals.items():
                flat[f"{block_name}.{k}"] = v
    return flat


def _write_results(rows: list[dict[str, Any]], out_dir: Path, discovery: DiscoveryReport) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "benchmark_results.jsonl"
    csv_path = out_dir / "benchmark_results.csv"
    summary_path = out_dir / "benchmark_summary.json"

    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    flat_rows = [_flatten_row_for_csv(r) for r in rows]
    fieldnames: list[str] = []
    key_union: set[str] = set()
    for r in flat_rows:
        key_union.update(r.keys())
    fieldnames = sorted(key_union)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in flat_rows:
            w.writerow(r)

    ok_rows = [r for r in rows if r.get("status") == "ok"]
    gdt_values: list[float] = []
    for r in ok_rows:
        block = ((r.get("metrics") or {}).get("predicted_mutant_vs_experimental_mutant") or {})
        try:
            gdt_values.append(float(block.get("gdt_ts")))
        except (TypeError, ValueError):
            pass
    summary = {
        "schema_version": "2026-04-10.struct-benchmark.run.v1",
        "generated_at": _utc_now(),
        "counts": {
            "pairs_total": len(rows),
            "pairs_ok": len(ok_rows),
            "pairs_error": len(rows) - len(ok_rows),
            "discovery_exclusions": len(discovery.exclusions),
        },
        "discovery_stats": discovery.stats,
        "metrics_summary": {
            "main_gdt_ts_mean": (sum(gdt_values) / len(gdt_values)) if gdt_values else None,
            "main_gdt_ts_min": min(gdt_values) if gdt_values else None,
            "main_gdt_ts_max": max(gdt_values) if gdt_values else None,
        },
        "files": {
            "benchmark_results_jsonl": str(jsonl_path),
            "benchmark_results_csv": str(csv_path),
            "discovery_manifest_json": str(out_dir / "discovery_manifest.json"),
        },
        "top_cases_by_gdt_ts": sorted(
            [
                {
                    "job_id": r.get("job_id"),
                    "mutation_code": r.get("mutation_code"),
                    "gdt_ts": ((r.get("metrics") or {}).get("predicted_mutant_vs_experimental_mutant") or {}).get("gdt_ts"),
                }
                for r in ok_rows
            ],
            key=lambda x: float(x.get("gdt_ts") or -1.0),
            reverse=True,
        )[:10],
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _build_runner(args: argparse.Namespace) -> StructureRunner:
    if not bool(args.colabfold):
        return NullStructureRunner()
    extra = list(args.colabfold_arg)
    if str(args.colabfold_extra).strip():
        extra.extend(shlex.split(str(args.colabfold_extra)))
    return ColabFoldLocalRunner(
        binary=str(args.colabfold_bin),
        num_recycle=int(args.num_recycle),
        use_amber=bool(args.amber),
        overwrite_existing=bool(args.colabfold_overwrite),
        extra_args=tuple(extra),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Automated WT/mutant structural benchmark")
    ap.add_argument("--out-dir", type=Path, default=Path("struct_benchmark_runs") / datetime.now().strftime("%Y%m%d_%H%M%S"))
    ap.add_argument("--manifest-json", type=Path, default=None, help="Optional existing discovery manifest to reuse")
    ap.add_argument("--max-entries", type=int, default=800)
    ap.add_argument("--max-enzymes", type=int, default=3)
    ap.add_argument("--max-pairs-per-enzyme", type=int, default=3)
    ap.add_argument("--max-cases", type=int, default=0, help="0 means no explicit cap")
    ap.add_argument("--resolution-max", type=float, default=2.8)
    ap.add_argument("--min-seq-identity", type=float, default=0.95)
    ap.add_argument("--max-len-delta-frac", type=float, default=0.05)
    ap.add_argument("--skip-controls", action="store_true", help="Skip WT control comparisons to save runtime")
    ap.add_argument("--gdt-ts-min", type=float, default=None, help="Optional quality threshold for calibration flags")
    ap.add_argument("--coverage-min", type=float, default=None, help="Optional coverage threshold for calibration flags")
    ap.add_argument(
        "--disable-sequence-diff-fallback",
        action="store_true",
        help="Disable discovery fallback that infers single-point mutants from sequence diffs",
    )

    ap.add_argument("--colabfold", action="store_true", help="Run local colabfold_batch predictions")
    ap.add_argument("--colabfold-bin", default="colabfold_batch")
    ap.add_argument("--num-recycle", type=int, default=3)
    ap.add_argument("--amber", action="store_true")
    ap.add_argument("--colabfold-overwrite", action="store_true")
    ap.add_argument("--colabfold-arg", action="append", default=[], metavar="ARG")
    ap.add_argument("--colabfold-extra", default="")
    args = ap.parse_args()
    if args.gdt_ts_min is not None and not (0.0 <= float(args.gdt_ts_min) <= 100.0):
        raise SystemExit("--gdt-ts-min must be within [0,100].")
    if args.coverage_min is not None and not (0.0 <= float(args.coverage_min) <= 100.0):
        raise SystemExit("--coverage-min must be within [0,100].")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    exp_dir = out_dir / "experimental_structures"
    pred_dir = out_dir / "predicted_structures"

    if args.manifest_json is not None and args.manifest_json.is_file():
        payload = json.loads(args.manifest_json.read_text(encoding="utf-8"))
        pairs = [BenchmarkPair(**p) for p in (payload.get("pairs") or [])]
        report = DiscoveryReport(
            pairs=pairs,
            exclusions=list(payload.get("exclusions") or []),
            stats=dict(payload.get("stats") or {}),
        )
    else:
        report = discover_wt_mutant_pairs(
            max_entries=int(args.max_entries),
            max_enzymes=int(args.max_enzymes),
            max_pairs_per_enzyme=int(args.max_pairs_per_enzyme),
            resolution_max_a=float(args.resolution_max),
            min_seq_identity=float(args.min_seq_identity),
            max_len_delta_frac=float(args.max_len_delta_frac),
            enable_sequence_diff_fallback=not bool(args.disable_sequence_diff_fallback),
        )
    write_discovery_manifest(report, out_dir / "discovery_manifest.json")

    pairs = list(report.pairs)
    if args.max_cases and args.max_cases > 0:
        pairs = pairs[: int(args.max_cases)]
    if not pairs:
        raise SystemExit("No benchmark pairs discovered. Check filters or provide --manifest-json.")

    runner = _build_runner(args)
    wt_pred_cache: dict[str, Path] = {}
    rows: list[dict[str, Any]] = []
    for i, pair in enumerate(pairs, start=1):
        print(f"[struct-benchmark] case {i}/{len(pairs)}: {pair.wt_pdb_id} -> {pair.mut_pdb_id} ({pair.mutation_code})", flush=True)
        row = _score_pair(
            pair=pair,
            runner=runner,
            exp_dir=exp_dir,
            pred_dir=pred_dir,
            include_controls=not bool(args.skip_controls),
            wt_pred_cache=wt_pred_cache,
            gdt_ts_min=float(args.gdt_ts_min) if args.gdt_ts_min is not None else None,
            coverage_min=float(args.coverage_min) if args.coverage_min is not None else None,
        )
        rows.append(row)

    _write_results(rows, out_dir, report)
    print(f"[struct-benchmark] wrote outputs under: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
