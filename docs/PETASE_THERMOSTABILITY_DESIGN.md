# PETase thermostability — design loop (plan)

Goal: **Option C** — a generation loop that proposes **theoretical PETase variants**, scores them with **structure + physics-informed** proxies for **thermostability**, and keeps a **Pareto archive** of candidates for AF validation / lab testing.

## Biology anchor

- **IsPETase** (Ideonella sakaiensis) is the canonical PET-degrading enzyme; many structures exist (e.g. **PDB 6EQD**).
- **Thermostability** in engineering usually trades off **activity / flexibility at the active site** vs **global rigidity / core packing / surface charge / loop stabilization**.
- WT reference sequence for this repo: `petase_design/data/petase_6eqd_chainA_notag.fasta` (6EQD chain A, **His-tag stripped**). You may later switch to **mature-only** (signal peptide removed) numbering used in papers — document offsets if you compare to literature mutations.

## Loop (high level)

```
  ┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
  │ Propose seq │ ──► │ Predict structure │ ──► │ Physics + ML score │
  │ (mutations) │     │ (ColabFold / AF2)  │     │ (composite)        │
  └─────────────┘     └──────────────────┘     └─────────┬─────────┘
         ▲                                                │
         │         ┌──────────────────┐                  │
         └──────── │ Update archive /   │ ◄────────────────┘
                   │ select parents   │
                   └──────────────────┘
```

1. **Propose:** start from **single- and multi-site** mutations on WT (later: fragment crossover, LM proposals).
2. **Structure:** **ColabFold** or local **AlphaFold2** — same protocol for all candidates (reproducibility).
3. **Physics layer (this is what makes “weak AF-only” stronger):**
   - **Tier 0 (implemented stub):** geometry proxies — compactness, local contact density near active-site shell (user-defined residues), clash proxy from van der Waals overlap counts (crude).
   - **Tier 1:** **SASA** (`freesasa`, optional install) — **implemented:** polar/apolar classification via `freesasa.classifyResults` on ranked PDBs; feeds `apolar_sasa_fraction` + `sasa_total_area` in `PhysicsBreakdown` and the composite (`petase_design/sasa_utils.py`).
   - **Tier 2:** **Molecular mechanics** — short **OpenMM** minimization + energy components (electrostatics, bonded strain) on a fixed FF (AMBER/CHARMM via OpenFF pipeline is possible but heavier).
   - **Tier 3:** **Rosetta / FoldX**-class ΔΔG or stability metrics (licensing + install separate).
4. **ML layer (your existing stack):** optional **ΔΔG head** or **ranking** using PLM embeddings + composition (+ later structural descriptors) — trained on FireProt is **not** PET-specific; use as a **weak prior** or retrain on PETase mutants when you have data.

## What “physics” means here (concrete)

| Signal | Role for thermostability | Tool direction |
|--------|-------------------------|----------------|
| Core packing / voids | Better packed cores ↑ Tm | Rosetta packstat / geometric proxies |
| H-bond network | Stabilizing without over-rigidifying active site | H-bond counting (MDAnalysis / Rosetta) |
| Electrostatics | Surface charge optimization, salt bridges | OpenMM GBSA or Poisson-Boltzmann (APBS) later |
| Flexibility | Too flexible → melts; too rigid → no activity | RMSF from short MD OR pLDDT/PAE from AF as cheap proxy |
| Metal / catalytic geometry | PETase chemistry — don’t break catalytic triad / substrate channel | Distance restraints, penalty in composite score |

## Implementation phases in this repo

| Phase | Deliverable |
|-------|-------------|
| **P0** | `petase_design/` package: load WT, mutate, composite **sequence + stub structure** scoring, JSONL logging, Pareto archive |
| **P1** | **ColabFold local** — `ColabFoldLocalRunner` + `--colabfold` on `petase_design.run` (see `docs/COLABFOLD_LOCAL.md`) |
| **P2** | **Done (this repo):** **freesasa** optional dep (`petase_design/requirements-extras.txt`); SASA polar/apolar terms in `physics_score` when PDB path is valid |
| **P3** | **OpenMM** energy minimization + component breakdown |
| **P4** | Active-site **restraints** + literature mutation priors (e.g. ThermoPET, known stabilizing sites) |

### P0 usage (now)

```bash
cd /path/to/petase-thermostability-benchmark
python -m petase_design.run --cycles 100 --mutations 3 --out petase_design_runs/log.jsonl
```

Optional: copy `petase_design/data/active_site_indices_0based.example.txt` to `active_site_indices_0based.txt` and list 0-based indices to protect from mutation.

Optional biopython / freesasa: `pip install -r petase_design/requirements-extras.txt` (for later tiers).

## Ethics / safety

Engineered PETases are **environmental biocatalysts**; still treat **dual-use** and **release** thoughtfully if you ever move beyond *in silico* design.

## References to carry forward

- PETase discovery / structure landscape: search **IsPETase**, **PDB 6EQD**, **thermostable PETase** variants in literature.
- ColabFold: `https://github.com/sokrypton/ColabFold`
- OpenMM: `http://openmm.org/`
