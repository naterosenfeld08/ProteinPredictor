# Limitations & scientific priors

## FireProt ΔΔG head (embedding → scalar)

1. **Training task:** Models map **fixed PLM embeddings** (+ optional composition) to **experimental ΔΔG** for **single amino-acid mutations** curated in **FireProtDB**, across **many unrelated proteins**.

2. **Not PET-specific:** Unless you **retrain or fine-tune** on **PETase (or enzyme) mutant data**, the output is a **generic stability prior** on the FireProt scale — **not** a predictor of PET hydrolysis, **Tm**, or **activity**.

3. **Sequence-only inference:** Pasting a **full sequence** does **not** define a mutant–wild-type pair. The number is **“what the regressor outputs on this embedding”** — useful for **relative ranking** under a **fixed pipeline**, not as a measured thermodynamic quantity for that construct.

4. **“Uncertainty” from Random Forest:** Inter-tree **variance** indicates **model disagreement** in embedding space. It is **not** calibrated experimental error. Wide intervals often mean **don’t trust the sign** tightly.

5. **PETase design loop:** The **physics composite** (sequence + optional structure + optional SASA) is a **cheap screen** for thermostability-flavored hypotheses. It does **not** replace MD, Rosetta, or lab assays.

When publishing or reporting to mentors, lead with these caveats before citing specific numeric ΔΔG values.
