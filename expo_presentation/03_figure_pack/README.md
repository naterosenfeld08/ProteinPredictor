# Figure Pack

Files in this folder are intended for trifold printing and monitor display.

## Source
- `figure_data.json` - locked numeric values used for generated plots.
- `pipeline_flow.mmd` - mermaid flow diagram source for method panel.

## Generated figures
- `discovery_reliability_bar.png`
- `pilot_vs_baseline_metrics.png`
- `integration_outputs_map.png`
- `discovery_funnel.png`
- `metric_delta_lollipop.png`
- `pilot_vs_baseline_radar.png`
- `integration_stage_impact.png`
- `metric_bridge_animation.gif`

## Regenerate

```bash
python3 generate_figures.py
python3 generate_advanced_visuals.py
```

## PyMOL figures

- `pymol_render_pilot_case.pml` generates structure overlays and mutation closeups.
- See `ADVANCED_VISUALS_GUIDE.md` for command details and placement suggestions.
