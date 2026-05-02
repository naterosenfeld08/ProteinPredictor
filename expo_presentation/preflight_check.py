#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


MIN_W = 1200
MIN_H = 700


def _check_file(path: Path) -> bool:
    if not path.exists():
        print(f"[MISSING] {path}")
        return False
    if path.suffix.lower() == ".png":
        w, h = _read_png_size(path)
        if w < MIN_W or h < MIN_H:
            print(f"[LOW_RES] {path} ({w}x{h})")
            return False
        print(f"[OK] {path} ({w}x{h})")
        return True
    print(f"[OK] {path}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate expo presentation assets.")
    parser.add_argument(
        "--include-advanced",
        action="store_true",
        help="Also validate advanced figure pack assets (PNG/GIF).",
    )
    parser.add_argument(
        "--include-pymol",
        action="store_true",
        help="Require PyMOL-rendered PNG outputs in addition to advanced assets.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    required = [
        root / "01_storyline.txt",
        root / "02_trifold_panel_copy.md",
        root / "03_figure_pack" / "discovery_reliability_bar.png",
        root / "03_figure_pack" / "pilot_vs_baseline_metrics.png",
        root / "03_figure_pack" / "integration_outputs_map.png",
        root / "04_monitor_demo_script.md",
        root / "05_rehearsal_and_qa.md",
        root / "06_event_readiness_checklist.md",
    ]
    advanced_required = [
        root / "03_figure_pack" / "discovery_funnel.png",
        root / "03_figure_pack" / "metric_delta_lollipop.png",
        root / "03_figure_pack" / "pilot_vs_baseline_radar.png",
        root / "03_figure_pack" / "integration_stage_impact.png",
        root / "03_figure_pack" / "metric_bridge_animation.gif",
    ]
    pymol_required = [
        root / "03_figure_pack" / "pymol_superposition_overview.png",
        root / "03_figure_pack" / "pymol_mutation_site_closeup.png",
        root / "03_figure_pack" / "pymol_predicted_confidence_style.png",
    ]

    if args.include_advanced:
        required.extend(advanced_required)
    if args.include_pymol:
        required.extend(advanced_required)
        required.extend(pymol_required)

    ok = all(_check_file(p) for p in required)
    print("\nPreflight status:", "PASS" if ok else "FAIL")


def _read_png_size(path: Path) -> tuple[int, int]:
    raw = path.read_bytes()
    if len(raw) < 24 or raw[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"Invalid PNG file: {path}")
    width = int.from_bytes(raw[16:20], "big")
    height = int.from_bytes(raw[20:24], "big")
    return width, height


if __name__ == "__main__":
    main()
