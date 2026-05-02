# Advanced Visuals Guide

This guide adds high-impact visuals for both the board and monitor.

## 1) Generate upgraded chart pack

From repo root:

```bash
python3 expo_presentation/03_figure_pack/generate_advanced_visuals.py
```

Outputs:
- `discovery_funnel.png`
- `metric_delta_lollipop.png`
- `pilot_vs_baseline_radar.png`
- `integration_stage_impact.png`
- `metric_bridge_animation.gif`

## 2) Generate PyMOL structure renders

Install PyMOL if needed (recommended: `pymol-open-source`).

Run:

```bash
cd expo_presentation/03_figure_pack
pymol -cq pymol_render_pilot_case.pml
```

Outputs:
- `pymol_superposition_overview.png`
- `pymol_mutation_site_closeup.png`
- `pymol_predicted_confidence_style.png`

## 3) Suggested placement

- **Board left panel:** `pymol_superposition_overview.png` + short methods paragraph.
- **Board center panel:** `metric_delta_lollipop.png` and `pilot_vs_baseline_radar.png`.
- **Board right panel:** `integration_stage_impact.png` + limitations/next-steps bullets.
- **Monitor loop:** `metric_bridge_animation.gif` + PyMOL closeup figure.
