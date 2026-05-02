# Rehearsal and Q&A Packet

## A) Short Public Pitch (45-60s)

"We built a structural benchmarking workflow for protein design. It automatically finds WT/mutant structure pairs, predicts mutant structures, and compares predictions to experimental structures using CASP-style metrics like GDT-TS. We also solved a key reliability issue by adding a sequence-diff fallback when mutation metadata is noisy. In our pilot, the framework ran end-to-end and produced calibrated ranking signals now integrated into design outputs."

## B) Judge-Depth Pitch (2.5-3 minutes)

1. **Context**  
   Model-only ranking can overstate confidence when benchmark coverage is weak.

2. **Method**  
   - RCSB pairing with strict filters  
   - sequence-diff fallback for unresolved mutation annotations  
   - ColabFold mutant prediction  
   - structural comparison vs experiment: GDT-TS, GDT-HA, RMS_CA, TM-score, coverage  
   - calibration controls and delta metrics

3. **Evidence**  
   - reliability snapshot recovered non-zero practical pairs  
   - pilot: GDT-TS 54.76 (pred vs exp mutant) vs 99.29 (exp WT vs exp mutant baseline)

4. **Integration**  
   Benchmark fields now propagate into JSONL/run_summary and can influence ranking.

5. **Limitations + next step**  
   n=1 pilot for real run; scale to 10-50 benchmark pairs and validate threshold policy.

## C) Likely Judge Questions + Strong Answers

### Q1: Why GDT-TS and not only RMSD?
A: GDT-TS is less dominated by outlier regions and is a CASP-standard global agreement metric. We still report RMS_CA and TM-score for complementary interpretation.

### Q2: How do you handle noisy or missing mutation metadata?
A: We use strict metadata first, then sequence-diff inference within UniProt-grouped candidates to recover single-point mutants when annotation fields are incomplete.

### Q3: Is one pilot case enough?
A: No. The pilot validates pipeline operability and interpretability. We explicitly treat it as proof-of-pipeline and plan scaling to a larger benchmark panel.

### Q4: How do results affect decisions?
A: We produce calibration fields and quality flags that can be blended into ranking, so candidates are prioritized using structural evidence rather than only one model score.

### Q5: What are your biggest error sources?
A: Metadata quality variance, sequence/chain mapping complexity, and runtime constraints for structure prediction on CPU.

## D) Practice Routine (recommended)

1. 3 full runs of 60s public pitch.
2. 3 full runs of 3-minute judge pitch with timer.
3. 2 mock Q&A rounds (10 questions each).
4. Final no-notes dry run at the board with monitor on.
