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

### C) `conda: command not found` on macOS

That means **no conda is installed**, or your **zsh was never initialized** for conda.

1. **Install a conda base** (pick one):
   - **Homebrew:** `brew install --cask miniforge`  
     Then **close and reopen Terminal**, or run the “Next steps” line `brew` prints (often `conda init zsh` using the Miniforge path).
   - **Manual:** download **Miniforge** from [conda-forge.org/miniforge](https://github.com/conda-forge/miniforge#miniforge3), run the installer, then:
     ```bash
     /Users/YOU/miniforge3/bin/conda init zsh
     ```
     Open a **new** terminal tab.

2. **Verify:**
   ```bash
   which conda
   conda --version
   ```

3. **Then** follow [LocalColabFold](https://github.com/YoshitakaMo/localcolabfold) to create an env that provides **`colabfold_batch`**.

Until `which colabfold_batch` prints a real path, **`--colabfold` in this repo cannot run** — you will get **`structure_pdb: null`** and (with current code) a **`colabfold.stderr.log`** under `petase_design_runs/structures/<job_id>/` whose **`[preflight]`** section explains the missing binary.

**Why `tail .../colabfold.stderr.log` says “No such file”:** either you never ran from this clone with `--colabfold` after that job folder was created, the job dir is on another machine, or `petase_design_runs/` was deleted — it is **gitignored**, so a fresh `git clone` starts with **no** run artifacts. After a real `--colabfold` run, check:

```bash
ls -la petase_design_runs/structures/
```

## GPU

Structure prediction is slow on CPU. Use a machine with **NVIDIA GPU + working CUDA** in the same environment that provides `colabfold_batch`.

**“No output” while running:** older code buffered all ColabFold stdout until exit. Current `ColabFoldLocalRunner` **streams** ColabFold’s merged stdout/stderr to **your terminal (stderr)** line-by-line and to **`colabfold.stderr.log`**. On **Mac CPU**, a ~300-residue protein can take **tens of minutes to hours**. ColabFold itself often prints **nothing for many minutes** while JAX runs (between lines like `recycle=0 pLDDT=…` and the next) — that is **normal**, not a freeze. A **heartbeat** every 2 minutes from `petase_design` reminds you the job is still alive. The line `Wrote N variants …` appears only **after** each job’s ColabFold step finishes.

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
- Log: `colabfold.stderr.log` — always created when `--colabfold` is on: either **`[preflight]`** (binary missing / not on `PATH`) or full subprocess **stdout/stderr** + return code + optional **`[structure discovery]`**.
- If PDB isn’t found but mmCIF is: install **`biopython`** (`pip install -r petase_design/requirements-extras.txt`) for automatic CIF→PDB conversion for the physics compactness term.

## Troubleshooting

1. **`colabfold_batch` not found** — activate the conda env where ColabFold lives, or pass **`--colabfold-bin`**.
2. **`returncode != 0`** — read `colabfold.stderr.log` in that job folder.
3. **No PDB after success** — ColabFold version may only emit **mmCIF**; install **biopython** or upgrade ColabFold. The job’s **`colabfold.stderr.log`** now ends with a **`[structure discovery]`** section listing every **`.pdb` / `.cif`** found under that job directory (recursive) so you can compare filenames to what your ColabFold build emits. Discovery uses **`colabfold_io.find_best_structure_*`** heuristics (`ranked_0`, `unrelaxed_*_rank_*`, etc.).
4. **Non-zero exit but files exist** — some installs return an error after writing models; we **may still use** a found PDB/mmCIF and log a **`[warning]`** in `colabfold.stderr.log`. Verify models manually if that happens.
5. **MSA / rate limit errors** — reduce `--cycles`, add delay between jobs, or switch to **local `colabfold_search`**.
