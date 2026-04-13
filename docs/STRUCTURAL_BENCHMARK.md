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
| `--disable-sequence-diff-fallback` | Turn off fallback that infers single-point mutants from WT↔mutant sequence diffs |
| `--gdt-ts-min` | Optional quality threshold used in calibration flags |
| `--coverage-min` | Optional coverage threshold used in calibration flags |

## Design CLI Integration

Design runs can now attach structural benchmark evidence to each structure-scored variant when a benchmark manifest is available.

```bash
python -m petase_design.run \
  --colabfold \
  --cycles 20 \
  --mutations 2 \
  --struct-benchmark-manifest struct_benchmark_runs/real_colabfold_pilot/pilot_manifest.json \
  --struct-benchmark-gdt-ts-min 50 \
  --struct-benchmark-coverage-min 95 \
  --struct-benchmark-weight 0.20 \
  --out petase_design_runs/design_with_benchmark.jsonl
```

Key integration flags:

| Flag | Meaning |
|------|---------|
| `--struct-benchmark-manifest path.json` | Benchmark reference pairs used to evaluate matching design variants |
| `--struct-benchmark-include-controls` | Also compute predicted WT control blocks (slower) |
| `--struct-benchmark-gdt-ts-min` | Marks low-quality benchmark matches in calibration flags |
| `--struct-benchmark-coverage-min` | Marks low-coverage comparisons in calibration flags |
| `--struct-benchmark-weight` | Blends benchmark GDT signal into archive ranking score |

## Outputs

Under `--out-dir`:

- `discovery_manifest.json`
  - selected WT/mutant pairs
  - discovery stats
  - exclusion reasons for rejected entries
- `benchmark_results.jsonl`
  - one record per benchmark case
  - includes `calibration.*` fields (main score, baseline deltas, quality flags)
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

- Discovery uses a strict metadata path first, then a fallback that infers single-point mutants from sequence diffs inside UniProt-grouped entries when mutation annotations are missing/noisy.
- Mutant inclusion remains restricted to single missense-like changes for benchmark pairing.
- CASP-like metrics are implemented on C-alpha superposition with sequence-dependent residue mapping.
- Design-loop integration writes `struct_benchmark.*` blocks only for variants with predicted structures and matching manifest mutation codes.
