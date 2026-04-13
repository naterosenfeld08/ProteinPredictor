# Automated WT/Mutant Structural Benchmark

This workflow discovers enzymes with both experimental WT and mutant X-ray structures, predicts mutant structures with local ColabFold, and compares predictions against experiment with CASP-style metrics.

Primary metric:
- `GDT_TS = (P1 + P2 + P4 + P8) / 4`

Secondary metrics:
- `GDT_HA = (P0.5 + P1 + P2 + P4) / 4`
- `RMS_CA`
- approximate `TM-score`
- residue coverage (`matched_residues / target_residues`)

## Prerequisites

From repo root:

```bash
pip install -r requirements.txt
```

For structure prediction with ColabFold, install and validate `colabfold_batch` (see `docs/COLABFOLD_LOCAL.md`):

```bash
which colabfold_batch
colabfold_batch --help
```

## Run

```bash
python -m petase_design.benchmark_run \
  --colabfold \
  --max-enzymes 3 \
  --max-pairs-per-enzyme 2 \
  --max-entries 800 \
  --resolution-max 2.8 \
  --out-dir struct_benchmark_runs/run1
```

Useful flags:

| Flag | Meaning |
|------|---------|
| `--manifest-json path.json` | Reuse an existing discovery manifest instead of querying RCSB again |
| `--max-cases N` | Limit benchmark to first `N` discovered pairs |
| `--skip-controls` | Skip WT control comparisons to reduce runtime |
| `--min-seq-identity` | Tighten/relax WT↔mutant sequence comparability in discovery |
| `--max-len-delta-frac` | Maximum relative sequence length mismatch allowed for pairing |
| `--colabfold-bin` | Path/name of `colabfold_batch` |

## Outputs

Under `--out-dir`:

- `discovery_manifest.json`
  - selected WT/mutant pairs
  - discovery stats
  - exclusion reasons for rejected entries
- `benchmark_results.jsonl`
  - one record per benchmark case
- `benchmark_results.csv`
  - flattened table for spreadsheet/statistical analysis
- `benchmark_summary.json`
  - run-level counts and aggregate GDT-TS summary
- `experimental_structures/`
  - downloaded PDBs and extracted chain files
- `predicted_structures/`
  - ColabFold outputs used for scoring

## Notes on Interpretation

- `GDT-TS` is less sensitive than RMSD to local outliers and is the primary ranking metric in this workflow.
- Crystal conditions, unresolved loops, alternate conformers, and chain selection can impact scores.
- WT vs mutant experimental differences are biological/contextual; low WT↔mutant similarity does not always indicate prediction failure.
- Use `percent_coverage` and `matched_residues` to detect partial comparisons.

## Current Scope

- Discovery currently filters to single-protein-entity X-ray entries with parseable mutation annotations.
- Mutant inclusion is currently limited to single missense-like annotations.
- CASP-like metrics are implemented on C-alpha superposition with sequence-dependent residue mapping.
