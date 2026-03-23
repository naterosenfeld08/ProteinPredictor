#!/usr/bin/env python3
"""
Run ΔΔG prediction in an isolated process.

Streamlit stays responsive while this script loads torch/transformers and embeds.
Invoked by gui/app.py — do not import Streamlit here.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    sys.path.insert(0, str(REPO_ROOT))
    from gui.worker_env import configure_worker_runtime_env, limit_torch_threads

    ap = argparse.ArgumentParser(description="Predict ΔΔG (worker process)")
    ap.add_argument("--fasta", required=True, help="Path to FASTA file")
    ap.add_argument("--model", required=True, help="Path to model .pkl")
    ap.add_argument(
        "--embedding-model-type",
        default="both",
        choices=("both", "prot_t5", "esm2"),
    )
    ap.add_argument(
        "--no-composition",
        action="store_true",
        help="Disable composition features (must match training)",
    )
    ap.add_argument(
        "--out",
        required=True,
        help="Write JSON: either predict_from_fasta results or {\"ok\":false,\"error\":...}",
    )
    args = ap.parse_args()

    # Before torch / transformers / tokenizers (macOS SIGSEGV / tokenizer thread issues).
    configure_worker_runtime_env()
    import torch

    limit_torch_threads()

    out_path = Path(args.out)

    try:
        from protein_baseline import predict_from_fasta

        results = predict_from_fasta(
            fasta_path=args.fasta,
            model_path=args.model,
            embedding_model_type=args.embedding_model_type,
            output_path=None,
            use_composition_features=not args.no_composition,
        )
        if results is None:
            raise RuntimeError(
                "predict_from_fasta returned None (internal bug: expected a results dict)."
            )
        out_path.write_text(json.dumps(results, default=str), encoding="utf-8")
        return 0
    except Exception:
        payload = {"ok": False, "error": traceback.format_exc()}
        out_path.write_text(json.dumps(payload), encoding="utf-8")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
