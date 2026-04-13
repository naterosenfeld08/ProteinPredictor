#!/usr/bin/env python3
"""
Fast smoke validation for structural benchmark workflow.

This uses a tiny mock `colabfold_batch` script so we can verify CLI + artifact
schema quickly without running real AlphaFold inference.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile


def _write_mock_colabfold(path: Path) -> None:
    path.write_text(
        """#!/usr/bin/env python3
import sys
from pathlib import Path

def main():
    if len(sys.argv) < 3:
        raise SystemExit("usage: mock_colabfold_batch query.fasta out_dir [args...]")
    fasta = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    seq = ""
    for ln in fasta.read_text(encoding="utf-8").splitlines():
        if ln.startswith(">"):
            continue
        seq += ln.strip().upper()
    n = max(24, len(seq))
    lines = []
    for i in range(1, n + 1):
        x = float(i) * 1.25
        y = 0.3 if i % 2 else -0.3
        z = 0.0
        lines.append(f"ATOM  {i:5d}  CA  ALA A{i:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           C")
    lines.extend(["TER", "END"])
    (out_dir / "ranked_0.pdb").write_text("\\n".join(lines) + "\\n", encoding="utf-8")

if __name__ == "__main__":
    main()
""",
        encoding="utf-8",
    )
    path.chmod(0o755)


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="struct_benchmark_smoke_") as td:
        tdp = Path(td)
        out_dir = tdp / "run_out"
        mock_bin = tdp / "mock_colabfold_batch.py"
        manifest_path = tdp / "manifest.json"
        _write_mock_colabfold(mock_bin)

        manifest = {
            "schema_version": "smoke.v1",
            "stats": {"source": "smoke_manifest"},
            "exclusions": [],
            "pairs": [
                {
                    "enzyme_key": "smoke_ubq",
                    "enzyme_label": "smoke_ubq",
                    "uniprot_id": None,
                    "wt_pdb_id": "1UBQ",
                    "wt_entity_id": "1",
                    "wt_chain_id": "A",
                    "wt_sequence": "A" * 76,
                    "wt_resolution_a": 1.8,
                    "mut_pdb_id": "1UBQ",
                    "mut_entity_id": "1",
                    "mut_chain_id": "A",
                    "mut_sequence": "A" * 76,
                    "mut_resolution_a": 1.8,
                    "mutation_code": "A1V",
                    "mutation_raw": "A1V",
                    "seq_identity": 1.0,
                }
            ],
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        cmd = [
            sys.executable,
            "-m",
            "petase_design.benchmark_run",
            "--manifest-json",
            str(manifest_path),
            "--out-dir",
            str(out_dir),
            "--colabfold",
            "--colabfold-bin",
            str(mock_bin),
            "--max-cases",
            "1",
            "--gdt-ts-min",
            "40",
            "--coverage-min",
            "80",
        ]
        proc = subprocess.run(cmd, cwd=str(repo_root), text=True, capture_output=True)
        _assert(proc.returncode == 0, f"benchmark_run failed\nstdout={proc.stdout}\nstderr={proc.stderr}")

        summary_path = out_dir / "benchmark_summary.json"
        jsonl_path = out_dir / "benchmark_results.jsonl"
        csv_path = out_dir / "benchmark_results.csv"
        manifest_out_path = out_dir / "discovery_manifest.json"
        _assert(summary_path.is_file(), "missing benchmark_summary.json")
        _assert(jsonl_path.is_file(), "missing benchmark_results.jsonl")
        _assert(csv_path.is_file(), "missing benchmark_results.csv")
        _assert(manifest_out_path.is_file(), "missing discovery_manifest.json")

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        _assert("counts" in summary and "metrics_summary" in summary, "summary missing required sections")
        _assert(int((summary.get("counts") or {}).get("pairs_ok") or 0) >= 1, "expected at least one ok pair")

        rows = [json.loads(x) for x in jsonl_path.read_text(encoding="utf-8").splitlines() if x.strip()]
        _assert(len(rows) == 1, "expected one row in JSONL")
        row0 = rows[0]
        _assert(row0.get("status") == "ok", "expected status=ok")
        calib = row0.get("calibration") or {}
        _assert("main_gdt_ts" in calib and "passes_thresholds" in calib, "missing calibration fields")
        print("struct_benchmark_smoke: PASS")


if __name__ == "__main__":
    main()
