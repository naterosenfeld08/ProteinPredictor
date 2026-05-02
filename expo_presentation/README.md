# Trifold Expo Package

This folder contains a complete, research-expo-ready presentation package based on the structural benchmark integration work.

## Contents

- `01_storyline.txt` - final title, central claim, and one-line takeaway.
- `02_trifold_panel_copy.md` - print-ready text for left/center/right board panels.
- `03_figure_pack/` - figure data, figure-generation script, and generated PNGs for print/monitor.
- `04_monitor_demo_script.md` - 2-minute rotating demo flow and fallback flow.
- `05_rehearsal_and_qa.md` - short pitch, judge-depth pitch, and Q&A prep.
- `06_event_readiness_checklist.md` - print, hardware, and booth setup checklist.

## Recommended use order

1. Print panel text from `02_trifold_panel_copy.md`.
2. Generate/print figures from `03_figure_pack/*.png`.
3. Load monitor talking points from `04_monitor_demo_script.md`.
4. Rehearse with `05_rehearsal_and_qa.md`.
5. Final pre-event checks with `06_event_readiness_checklist.md`.

## Validation commands

```bash
python3 expo_presentation/03_figure_pack/generate_figures.py
python3 expo_presentation/03_figure_pack/generate_advanced_visuals.py
python3 expo_presentation/preflight_check.py --include-advanced
```

For full structural rendering validation (requires PyMOL):

```bash
cd expo_presentation/03_figure_pack
pymol -cq pymol_render_pilot_case.pml
cd ../..
python3 expo_presentation/preflight_check.py --include-pymol
```
