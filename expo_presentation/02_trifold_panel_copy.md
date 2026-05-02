# Trifold Panel Copy (Scientific Research Expo)

Use this file as print-ready content for a standard trifold board.

---

## Left Panel: Background and Methods

### Problem
Protein design workflows increasingly depend on predicted structures, but benchmark confidence is often weak when WT/mutant metadata is incomplete or noisy.

### Research Question
How reliably can we pair experimental WT/mutant structures and quantify predicted mutant structure accuracy in a way that improves design-stage ranking decisions?

### Hypothesis
If we combine CASP-style structural metrics with robust WT/mutant pair discovery fallback logic, then structural benchmark outputs can become calibration signals for design ranking.

### Method Summary
1. Query RCSB for high-quality X-ray structures (single-protein entity).
2. Build WT/mutant pairs using strict metadata first.
3. Apply sequence-diff fallback when mutation annotations are missing/noisy.
4. Predict mutant structures (ColabFold).
5. Compare predicted mutant vs experimental mutant using:
   - GDT-TS (primary)
   - GDT-HA
   - RMS_CA
   - TM-score
   - coverage
6. Add calibration controls:
   - predicted WT vs experimental WT
   - predicted WT vs experimental mutant
   - experimental WT vs experimental mutant
7. Integrate benchmark/calibration signals into design ranking outputs.

### Figure Placeholder (Left)
- Figure L1: Pipeline diagram (`03_figure_pack/pipeline_flow.png`)

---

## Center Panel: Results (Primary Visual)

### Discovery Reliability Snapshot
- Query sample size: 300 entries
- Candidate entries: 232
- Selected WT/mutant pairs: 3
- Selected enzymes: 3
- Fallback-recovered pairs: 3
- Title-promoted unresolved mutants: 15

### Recovered Pair Examples
- 8C3X -> 8R43 (H110N)
- 9HDW -> 9HDS (T99N)
- 2OV0 -> 2QDV (M51A)

### Real Pilot Benchmark Case
Case: 2OV0:A (WT) -> 2QDV:A (Mutant), M51A

Predicted mutant vs experimental mutant:
- GDT-TS: 54.76
- GDT-HA: 30.00
- RMS_CA: 2.93 A
- TM-score: 0.655
- Coverage: 99.06%

Baseline context (experimental WT vs experimental mutant):
- GDT-TS: 99.29
- GDT-HA: 93.57
- RMS_CA: 0.45 A
- TM-score: 0.977

### Center Takeaway Statement
The benchmark pipeline works end-to-end and yields interpretable structural calibration signals, while showing that model-vs-experiment agreement can differ substantially from experimental WT/mutant similarity.

### Figure Placeholders (Center)
- Figure C1: Discovery reliability chart (`03_figure_pack/discovery_reliability_bar.png`)
- Figure C2: Pilot vs baseline comparison (`03_figure_pack/pilot_vs_baseline_metrics.png`)

---

## Right Panel: Interpretation, Impact, and Next Steps

### Interpretation
- Discovery fallback materially improves practical pair recovery under real metadata noise.
- Pilot results show moderate predicted-vs-experimental agreement despite very high experimental WT/mutant structural similarity.
- This validates thresholded, calibration-aware interpretation rather than single-score ranking.

### Integration Impact (Implemented)
- Benchmark metrics standardized and logged.
- Calibration deltas/quality flags added.
- Design CLI includes benchmark manifest/threshold/weight controls.
- Run summary includes benchmark coverage and aggregate structural metrics.

### Limitations
- RCSB metadata quality remains variable.
- CPU-only ColabFold is slow for larger studies.
- Pilot is n=1 for real end-to-end case; broader benchmarking is still required.

### Next Research Steps
1. Scale benchmark set to 10-50 pairs across enzymes.
2. Define and validate pass/fail policy bands for GDT-TS and coverage.
3. Quantify design ranking stability with vs without calibration controls.
4. Add optional GUI-facing benchmark dashboard once CLI stability is confirmed.

### Figure Placeholder (Right)
- Figure R1: Integration output map (`03_figure_pack/integration_outputs_map.png`)

### Footer
- Repo: https://github.com/naterosenfeld08/petase-thermostability-benchmark
- Contact: Nate Rosenfeld
