#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    here = Path(__file__).resolve().parent
    data = json.loads((here / "figure_data.json").read_text(encoding="utf-8"))

    _discovery_chart(data, here / "discovery_reliability_bar.png")
    _pilot_vs_baseline_chart(data, here / "pilot_vs_baseline_metrics.png")
    _integration_outputs_map(data, here / "integration_outputs_map.png")


def _discovery_chart(data: dict, out_path: Path) -> None:
    snap = data["discovery_snapshot"]
    labels = [
        "Searched\nentries",
        "Candidate\nentries",
        "Selected\npairs",
        "Fallback\npairs",
        "Promoted\nmutants",
    ]
    values = [
        snap["searched_entries"],
        snap["candidate_entries"],
        snap["selected_pairs"],
        snap["fallback_pairs"],
        snap["title_promoted_mutants"],
    ]
    colors = ["#4E79A7", "#59A14F", "#E15759", "#F28E2B", "#76B7B2"]

    plt.figure(figsize=(9, 5))
    bars = plt.bar(labels, values, color=colors)
    plt.title("Discovery Reliability Snapshot (Fallback Enabled)", fontsize=14, weight="bold")
    plt.ylabel("Count")
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    for b, v in zip(bars, values):
        plt.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v}", ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def _pilot_vs_baseline_chart(data: dict, out_path: Path) -> None:
    pilot = data["pilot_case"]["predicted_vs_experimental_mutant"]
    base = data["pilot_case"]["experimental_wt_vs_experimental_mutant"]

    metrics = ["GDT-TS", "GDT-HA", "TM-score", "Coverage"]
    pilot_vals = [pilot["gdt_ts"], pilot["gdt_ha"], pilot["tm_score"] * 100.0, pilot["coverage"]]
    base_vals = [base["gdt_ts"], base["gdt_ha"], base["tm_score"] * 100.0, base["coverage"]]

    x = range(len(metrics))
    width = 0.36

    plt.figure(figsize=(9, 5))
    plt.bar([i - width / 2 for i in x], pilot_vals, width=width, label="Predicted mutant vs Exp mutant", color="#E15759")
    plt.bar([i + width / 2 for i in x], base_vals, width=width, label="Exp WT vs Exp mutant", color="#4E79A7")
    plt.xticks(list(x), metrics)
    plt.ylim(0, 105)
    plt.ylabel("Percent-like scale (TM-score x100)")
    plt.title("Pilot Case Structural Metrics: Model vs Baseline", fontsize=14, weight="bold")
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def _integration_outputs_map(data: dict, out_path: Path) -> None:
    items = data["integration_components"]

    plt.figure(figsize=(10, 5))
    y = list(range(len(items)))
    plt.barh(y, [1] * len(items), color="#59A14F")
    plt.yticks(y, items)
    plt.xticks([])
    plt.xlim(0, 1.2)
    plt.title("Implemented Integration Components", fontsize=14, weight="bold")
    for i in y:
        plt.text(1.03, i, "Implemented", va="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


if __name__ == "__main__":
    main()
