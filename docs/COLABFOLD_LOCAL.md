# Local ColabFold for `petase_design`

The design loop can call **`colabfold_batch`** for each variant when you pass **`--colabfold`** to `python -m petase_design.run`.

## Install (pick one path)

### A) LocalColabFold (common on Mac / Linux)

Follow the installer for your OS:

- **https://github.com/YoshitakaMo/localcolabfold**

After install, ensure `colabfold_batch` is on your `PATH` (often inside the conda env you created). Test:

```bash
which colabfold_batch
colabfold_batch --help
```

### B) Upstream ColabFold in conda

See **https://github.com/sokrypton/ColabFold** — install into a dedicated environment; dependencies include CUDA for reasonable speed.

## GPU

Structure prediction is slow on CPU. Use a machine with **NVIDIA GPU + working CUDA** in the same environment that provides `colabfold_batch`.

## MSA: public server vs local

By default ColabFold may query the **public MSA server**. That is fine for a few runs; **large batches** should use **`colabfold_search` + local databases** (see ColabFold wiki “Local MSA generation”) to avoid rate limits and to keep sequences on-prem.

## Run with this repo

From the project root:

```bash
# Sequence-only (fast)
python -m petase_design.run --cycles 10 --mutations 2 --out petase_design_runs/log.jsonl

# With local ColabFold (slow: one structure per variant)
python -m petase_design.run --colabfold --cycles 3 --mutations 1 --out petase_design_runs/log_cf.jsonl
```

Useful flags:

| Flag | Purpose |
|------|---------|
| `--colabfold-bin /path/to/colabfold_batch` | Non-standard install location |
| `--num-recycle N` | Passed as `--num-recycle` (default 3) |
| `--amber` | Adds `--amber` (OpenMM relax; slower) |
| `--colabfold-arg ARG` | Repeat for each extra CLI token |
| `--colabfold-extra '...'` | One shell-quoted string of extra args (`shlex.split`) |

Each job writes under `--work-dir` (default `petase_design_runs/structures/<job_id>/`):

- Input: `<job_id>.fasta`
- Log: `colabfold.stderr.log` (stdout/stderr + return code)
- If PDB isn’t found but mmCIF is: install **`biopython`** (`pip install -r petase_design/requirements-extras.txt`) for automatic CIF→PDB conversion for the physics compactness term.

## Troubleshooting

1. **`colabfold_batch` not found** — activate the conda env where ColabFold lives, or pass **`--colabfold-bin`**.
2. **`returncode != 0`** — read `colabfold.stderr.log` in that job folder.
3. **No PDB after success** — ColabFold version may only emit **mmCIF**; install **biopython** or upgrade ColabFold; check `colabfold_io.find_ranked_structure_*` patterns vs your files.
4. **MSA / rate limit errors** — reduce `--cycles`, add delay between jobs, or switch to **local `colabfold_search`**.
