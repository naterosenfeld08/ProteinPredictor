"""
Summaries for PETase design runs: compact JSON metadata written next to JSONL logs.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _sequence_preview(seq: str | None, *, head: int = 48) -> tuple[int, str]:
    if not seq:
        return 0, ""
    s = str(seq).strip()
    n = len(s)
    if n <= head:
        return n, s
    return n, s[:head] + "…"


def _mutations_compact(muts: list[dict[str, Any]] | None) -> str:
    if not muts:
        return ""
    parts: list[str] = []
    for m in muts:
        idx = m.get("index")
        to = m.get("to")
        if idx is None or to is None:
            continue
        parts.append(f"{idx}{to}")
    return ";".join(parts)


def build_run_summary(
    rows: list[dict[str, Any]],
    *,
    out_jsonl: Path,
    wall_seconds: float,
    started_at_iso: str,
    ended_at_iso: str,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    n_variants = len(rows)
    n_with_structure = 0
    n_with_sasa = 0
    for row in rows:
        if row.get("structure_pdb"):
            n_with_structure += 1
        phys = row.get("physics") or {}
        if phys.get("sasa_total_area") is not None:
            n_with_sasa += 1

    ranked = sorted(
        rows,
        key=lambda r: float((r.get("physics") or {}).get("composite", float("-inf"))),
        reverse=True,
    )
    top: list[dict[str, Any]] = []
    for i, row in enumerate(ranked[:10], start=1):
        phys = row.get("physics") or {}
        sp = row.get("structure_pdb")
        sp_short = Path(str(sp)).name if sp else None
        seq = row.get("sequence")
        slen, sprev = _sequence_preview(seq if isinstance(seq, str) else None)
        top.append(
            {
                "rank": i,
                "job_id": row.get("job_id"),
                "generation": row.get("generation"),
                "composite": phys.get("composite"),
                "mutation_count": phys.get("mutation_count"),
                "mutations_compact": _mutations_compact(row.get("mutations")),
                "has_structure": bool(sp),
                "structure_pdb_basename": sp_short,
                "sasa_total_area": phys.get("sasa_total_area"),
                "sequence_length": slen,
                "sequence_preview": sprev,
            }
        )

    out: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": ended_at_iso,
        "out_jsonl": str(out_jsonl),
        "counts": {
            "n_variants": n_variants,
            "n_with_structure": n_with_structure,
            "n_with_sasa": n_with_sasa,
        },
        "runtime": {
            "seconds_wall": round(wall_seconds, 3),
            "started_at": started_at_iso,
            "ended_at": ended_at_iso,
        },
        "top_variants": top,
    }
    if meta:
        out["run"] = meta
    return out


def write_run_summary_json(
    out_jsonl: Path,
    rows: list[dict[str, Any]],
    *,
    t0: float,
    t1: float,
    meta: dict[str, Any] | None = None,
) -> Path:
    """Write ``run_summary.json`` next to the design JSONL."""
    started = datetime.fromtimestamp(t0, tz=timezone.utc).isoformat()
    ended = datetime.fromtimestamp(t1, tz=timezone.utc).isoformat()
    payload = build_run_summary(
        rows,
        out_jsonl=out_jsonl,
        wall_seconds=float(t1 - t0),
        started_at_iso=started,
        ended_at_iso=ended,
        meta=meta,
    )
    path = out_jsonl.parent / "run_summary.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path
