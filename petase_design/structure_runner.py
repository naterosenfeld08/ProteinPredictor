from __future__ import annotations

"""
Structure prediction: local ColabFold (`colabfold_batch`) or skip (null runner).
"""

import os
import shutil
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from petase_design.colabfold_io import (
    cif_to_pdb,
    find_ranked_structure_cif,
    find_ranked_structure_pdb,
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
    extra_args: tuple[str, ...] = field(default_factory=tuple)
    env: dict[str, str] | None = None

    def predict(self, sequence: str, job_id: str, work_dir: Path) -> Path | None:
        work_dir.mkdir(parents=True, exist_ok=True)
        fasta = work_dir / f"{job_id}.fasta"
        fasta.write_text(f">{job_id}\n{sequence.strip()}\n", encoding="utf-8")

        exe_ok = Path(self.binary).is_file() or shutil.which(self.binary)
        if not exe_ok:
            log = work_dir / "colabfold_error.txt"
            log.write_text(
                f"Executable not found: {self.binary}\n"
                "Install LocalColabFold or add colabfold_batch to PATH.\n",
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
        cmd.extend(self.extra_args)

        run_env = os.environ.copy()
        if self.env:
            run_env.update(self.env)

        stderr_log = work_dir / "colabfold.stderr.log"
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(work_dir.parent),
                env=run_env,
                capture_output=True,
                text=True,
                timeout=None,
            )
            stderr_log.write_text(
                f"CMD: {' '.join(cmd)}\n"
                f"returncode={proc.returncode}\n\n=== STDERR ===\n{proc.stderr}\n\n=== STDOUT ===\n{proc.stdout}\n",
                encoding="utf-8",
            )
        except OSError as e:
            stderr_log.write_text(f"Failed to run colabfold_batch: {e}\nCMD: {cmd}\n", encoding="utf-8")
            return None

        if proc.returncode != 0:
            return None

        pdb = find_ranked_structure_pdb(work_dir)
        if pdb is not None:
            return pdb

        cif = find_ranked_structure_cif(work_dir)
        if cif is not None:
            converted = work_dir / f"{job_id}_ranked0_from_cif.pdb"
            if cif_to_pdb(cif, converted):
                return converted
            stderr_log.write_text(
                stderr_log.read_text(encoding="utf-8")
                + "\nFound mmCIF but could not convert to PDB (pip install biopython).\n",
                encoding="utf-8",
            )

        return None


class NullStructureRunner(StructureRunner):
    """Skip structure prediction (sequence-only physics)."""

    def predict(self, sequence: str, job_id: str, work_dir: Path) -> Path | None:
        work_dir.mkdir(parents=True, exist_ok=True)
        return None
