#!/usr/bin/env python3
"""
Run PETase design cycles in an isolated process so the Streamlit server stays responsive.

Invoked by ``gui/app.py`` with CLI args; writes JSONL and a full JSON result payload.
Do not import Streamlit here.
"""

from __future__ import annotations

import argparse
import json
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
    args = ap.parse_args()
    if args.structure_top_k is not None and int(args.structure_top_k) <= 0:
        raise SystemExit("--structure-top-k must be a positive integer.")

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

        rows = run_design_cycles(
            wt_fasta=args.wt_fasta,
            n_cycles=int(args.cycles),
            mutations_per_variant=int(args.mutations_per_variant),
            out_jsonl=args.out_jsonl,
            seed=int(args.seed),
            structure_runner=runner,
            work_root=args.work_root,
            structure_top_k=args.structure_top_k if args.colabfold else None,
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

        ddg_model = Path(args.ddg_model).expanduser() if args.ddg_model else None
        if ddg_model and ddg_model.is_file() and rows:
            pct = max(0.01, min(1.0, float(args.ddg_survivor_pct)))
            budget = max(1, int(round(len(rows) * pct)))
            selected_idx, selected_by, rescue_reason = _collect_ddg_survivors(
                rows,
                budget_count=budget,
                seed=int(args.seed),
            )
            survivors = [i for i in sorted(selected_idx) if str(rows[i].get("sequence", "")).strip()]

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

            rows = sorted(rows, key=lambda r: _safe_float(r.get("hybrid_score"), default=-1e9), reverse=True)

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
