from __future__ import annotations

"""
Structure prediction: local ColabFold (`colabfold_batch`) or skip (null runner).
"""

import os
import shutil
import subprocess
import sys
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from petase_design.colabfold_io import (
    cif_to_pdb,
    find_best_structure_cif,
    find_best_structure_pdb,
    format_structure_discovery_report,
)


class StructureRunner(ABC):
    @abstractmethod
    def predict(self, sequence: str, job_id: str, work_dir: Path) -> Path | None:
        """Write structure to work_dir; return path to best PDB or None on failure."""


class ColabFoldPlaceholder(StructureRunner):
    """Deprecated — use ColabFoldLocalRunner."""

    def predict(self, sequence: str, job_id: str, work_dir: Path) -> Path | None:
        raise NotImplementedError("Use ColabFoldLocalRunner (see --colabfold on petase_design.run).")


@dataclass
class ColabFoldLocalRunner(StructureRunner):
    """
    Run local ``colabfold_batch`` (ColabFold / LocalColabFold install).

    Typical install: https://github.com/YoshitakaMo/localcolabfold or pip colabfold in conda env.
    Ensure ``colabfold_batch`` is on PATH, or set ``binary`` to a full path.

    Command shape::
        colabfold_batch query.fasta out_dir --num-recycle 3 [extra_args...]
    """

    binary: str = "colabfold_batch"
    num_recycle: int = 3
    use_amber: bool = False
    #: Passes ``--overwrite-existing-results`` to ``colabfold_batch`` (recompute even if cached).
    overwrite_existing: bool = False
    extra_args: tuple[str, ...] = field(default_factory=tuple)
    env: dict[str, str] | None = None

    def predict(self, sequence: str, job_id: str, work_dir: Path) -> Path | None:
        work_dir.mkdir(parents=True, exist_ok=True)
        fasta = work_dir / f"{job_id}.fasta"
        fasta.write_text(f">{job_id}\n{sequence.strip()}\n", encoding="utf-8")

        stderr_log = work_dir / "colabfold.stderr.log"
        exe_path = Path(self.binary)
        exe_ok = exe_path.is_file() or bool(shutil.which(self.binary))
        if not exe_ok:
            # Always write the same log name so `tail .../colabfold.stderr.log` works after --colabfold.
            stderr_log.write_text(
                "[preflight] colabfold_batch was not executed — executable missing or not on PATH.\n"
                f"  --colabfold-bin: {self.binary!r}\n"
                f"  Path(...).is_file(): {exe_path.is_file()}\n"
                f"  shutil.which({self.binary!r}): {shutil.which(self.binary)!r}\n"
                f"  shutil.which('colabfold_batch'): {shutil.which('colabfold_batch')!r}\n\n"
                "Fix: run `conda activate <your-colabfold-env>` then `which colabfold_batch`, "
                "or pass the full path to the binary that exists.\n",
                encoding="utf-8",
            )
            return None

        cmd: list[str] = [
            self.binary,
            str(fasta.resolve()),
            str(work_dir.resolve()),
            "--num-recycle",
            str(self.num_recycle),
        ]
        if self.use_amber:
            cmd.append("--amber")
        if self.overwrite_existing:
            cmd.append("--overwrite-existing-results")
        cmd.extend(self.extra_args)

        run_env = os.environ.copy()
        if self.env:
            run_env.update(self.env)
        # Helps child Python processes line-buffer to our pipe when possible.
        run_env.setdefault("PYTHONUNBUFFERED", "1")

        def _append_log(extra: str) -> None:
            prev = ""
            if stderr_log.is_file():
                prev = stderr_log.read_text(encoding="utf-8", errors="replace")
            stderr_log.write_text(prev + extra, encoding="utf-8")

        returncode = -1
        try:
            # Stream stdout+stderr: ColabFold can run 30+ minutes on Mac CPU; buffering everything
            # with subprocess.run(capture_output=True) looks like a hang with "no output".
            with stderr_log.open("w", encoding="utf-8") as logf:
                logf.write(
                    f"CMD: {' '.join(cmd)}\n"
                    f"cwd={work_dir.resolve()}\n\n"
                    f"=== Live stream (echoed to stderr) ===\n"
                )
                logf.flush()
                print(
                    f"\n[petase_design] ColabFold {job_id}: started "
                    f"(long run on CPU — progress streams below; log: {stderr_log})\n",
                    file=sys.stderr,
                    flush=True,
                )
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(work_dir),
                    env=run_env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                assert proc.stdout is not None

                stop_hb = threading.Event()
                t0 = time.monotonic()

                def _heartbeat() -> None:
                    # ColabFold often prints nothing for many minutes during JAX / AF2 on CPU.
                    while not stop_hb.wait(120):
                        elapsed = int(time.monotonic() - t0)
                        print(
                            f"[petase_design] ColabFold {job_id}: still running (~{elapsed}s); "
                            "long gaps with no new lines are normal on CPU between model logs.\n",
                            file=sys.stderr,
                            flush=True,
                        )

                hb = threading.Thread(target=_heartbeat, daemon=True)
                hb.start()
                try:
                    for line in proc.stdout:
                        logf.write(line)
                        logf.flush()
                        print(line, end="", file=sys.stderr, flush=True)
                finally:
                    stop_hb.set()
                returncode = int(proc.wait())
                logf.write(f"\n=== end stream ===\nreturncode={returncode}\n")
                logf.flush()
                print(
                    f"\n[petase_design] ColabFold {job_id}: finished returncode={returncode}\n",
                    file=sys.stderr,
                    flush=True,
                )
        except OSError as e:
            stderr_log.write_text(f"Failed to run colabfold_batch: {e}\nCMD: {cmd}\n", encoding="utf-8")
            return None

        pdb = find_best_structure_pdb(work_dir)
        if pdb is not None and returncode == 0:
            return pdb

        cif = find_best_structure_cif(work_dir)
        if cif is not None and returncode == 0:
            converted = work_dir / f"{job_id}_ranked0_from_cif.pdb"
            if cif_to_pdb(cif, converted):
                return converted
            _append_log(
                "\nFound mmCIF but could not convert to PDB (install biopython: "
                "pip install -r petase_design/requirements-extras.txt).\n"
            )

        # Salvage: some ColabFold builds exit non-zero after writing models; if a PDB exists, use it.
        if pdb is not None and returncode != 0:
            _append_log(
                f"\n[warning] colabfold_batch returncode={returncode} but a PDB was found; "
                f"using {pdb.name} for scoring. Verify structure quality.\n"
            )
            return pdb
        if cif is not None and returncode != 0:
            converted = work_dir / f"{job_id}_ranked0_from_cif.pdb"
            if cif_to_pdb(cif, converted):
                _append_log(
                    f"\n[warning] returncode={returncode}; converted mmCIF to PDB for scoring.\n"
                )
                return converted

        _append_log(format_structure_discovery_report(work_dir))
        return None


class NullStructureRunner(StructureRunner):
    """Skip structure prediction (sequence-only physics)."""

    def predict(self, sequence: str, job_id: str, work_dir: Path) -> Path | None:
        work_dir.mkdir(parents=True, exist_ok=True)
        return None
