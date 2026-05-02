# Monitor Demo Script (90-120 seconds)

Use this for looping booth demos on the monitor while you narrate.

## Setup
- Monitor displays figures and summary slides.
- Laptop remains operator console (repo, scripts, fallback screenshots).

## Slide / Screen Order (Enhanced Visual Loop)

1. `03_figure_pack/discovery_funnel.png`
2. `03_figure_pack/pymol_superposition_overview.png`
3. `03_figure_pack/metric_delta_lollipop.png`
4. `03_figure_pack/pilot_vs_baseline_radar.png`
5. `03_figure_pack/integration_stage_impact.png`
6. `03_figure_pack/metric_bridge_animation.gif`
7. one screenshot of JSONL/run_summary output from a benchmark-enabled run

## Spoken Script (2-minute version)

### Segment A (20-25s): Problem and Goal
"Protein design uses predicted structures, but confidence is often unclear when WT/mutant metadata is messy. Our goal was to create a benchmark layer that directly improves ranking decisions."

### Segment B (20-25s): Reliability Innovation
"We start with strict WT/mutant pairing, then apply a sequence-diff fallback when mutation annotations are incomplete. In our realistic snapshot, this recovered all selected pairs."

### Segment C (20-25s): 3D Structural Context
"This PyMOL overlay shows experimental WT, experimental mutant, and predicted mutant after alignment, with the M51A region highlighted to connect benchmark metrics to actual local structure."

### Segment D (20-25s): Pilot Structural Findings
"In a real ColabFold pilot case, predicted-vs-experimental mutant agreement was moderate (GDT-TS 54.76), while experimental WT-vs-mutant similarity was high (GDT-TS 99.29). This shows why calibration controls are necessary."

### Segment E (20-25s): Integration Impact
"We integrated these benchmark and calibration signals into design CLI outputs and ranking logic, plus run-level reporting. So benchmark evidence now informs prioritization, not just post-hoc analysis."

## Fallback No-Internet Script (60-90 seconds)

If live tooling/network fails, show pre-generated PNGs and say:
- "These are frozen outputs from the validated run artifacts."
- "The methodology and outputs are reproducible from scripts in this repo."
- "The next phase is scaling pair count and validating threshold policy."

## Demo Operator Checklist
- Keep enhanced figures + GIF in one folder and open in presentation order.
- Keep `findings_summary.txt` open for exact numbers.
- Keep one JSONL or `run_summary.json` snippet ready for judge technical questions.
