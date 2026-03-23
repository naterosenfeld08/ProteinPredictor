"""
Apply environment + PyTorch thread limits **before** heavy imports.

Used by predict_worker / design_worker subprocesses. Prevents common macOS
crashes (SIGSEGV / exit code -11) when Hugging Face tokenizers use background
threads together with PyTorch and duplicated OpenMP runtimes.
"""

from __future__ import annotations

import os


def configure_worker_runtime_env() -> None:
    """setdefault so users can override in the shell if needed."""
    for key, val in (
        ("TOKENIZERS_PARALLELISM", "false"),
        ("OMP_NUM_THREADS", "1"),
        ("MKL_NUM_THREADS", "1"),
        ("OPENBLAS_NUM_THREADS", "1"),
        ("VECLIB_MAXIMUM_THREADS", "1"),
        ("NUMEXPR_NUM_THREADS", "1"),
        # Multiple libs ship OpenMP on macOS; this avoids hard crashes for many setups.
        ("KMP_DUPLICATE_LIB_OK", "TRUE"),
    ):
        os.environ.setdefault(key, val)


def limit_torch_threads() -> None:
    """Call immediately after `import torch`, before model work."""
    import torch

    torch.set_num_threads(1)
    if hasattr(torch, "set_num_interop_threads"):
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
