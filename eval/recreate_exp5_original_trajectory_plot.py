"""
Recreate eval/results/exp5_original/trajectory_plot.png from archived metrics.

Usage:
    python eval/recreate_exp5_original_trajectory_plot.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
ARCHIVE_JSON = ROOT / "eval" / "results" / "archive" / "metrics_iterations_2026-06-28_16-42-31_full_first_valid.json"

_INST_COLOURS = {
    "piano":  "#2196F3",
    "violin": "#FF5722",
    "oboe":   "#4CAF50",
    "drums":  "#9C27B0",
    "guitar": "#FF9800",
}
_DEFAULT_INST_COLOUR = "#888888"


def _plot_metric(
    ax: Any,
    trajectory: List[Dict[str, Any]],
    overall_metric: str,
    title: str,
    per_inst_metric: Optional[str] = None,
) -> None:
    iterations = [step["iteration"] for step in trajectory]
    values = [step.get(overall_metric) for step in trajectory]

    ax.plot(iterations, values, marker="o", linewidth=2, color="black", label="overall", zorder=3)

    if per_inst_metric:
        # Collect all instrument names present across all trajectory steps
        all_instruments: List[str] = []
        for step in trajectory:
            per_inst = step.get(per_inst_metric) or {}
            for inst in per_inst:
                if inst not in all_instruments:
                    all_instruments.append(inst)

        for inst in sorted(all_instruments):
            colour = _INST_COLOURS.get(inst, _DEFAULT_INST_COLOUR)
            inst_values = [
                (step.get(per_inst_metric) or {}).get(inst) for step in trajectory
            ]
            ax.plot(
                iterations,
                inst_values,
                marker="s",
                linewidth=1.4,
                linestyle="--",
                color=colour,
                alpha=0.8,
                label=inst,
            )

        # Also plot the per-inst mean if available
        mean_metric = per_inst_metric.replace("_per_inst", "_per_inst_mean")
        mean_values = [step.get(mean_metric) for step in trajectory]
        if any(v is not None for v in mean_values):
            ax.plot(
                iterations,
                mean_values,
                marker="^",
                linewidth=1.6,
                linestyle="-.",
                color="gray",
                alpha=0.7,
                label="inst mean",
            )

        ax.legend(fontsize=7, loc="upper right")

    ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(overall_metric)
    ax.grid(True, alpha=0.3)


def main() -> None:
    with ARCHIVE_JSON.open("r", encoding="utf-8") as f:
        data = json.load(f)

    out_png = ROOT / "eval" / "results" / "exp5" / f"trajectory_plot_{ARCHIVE_JSON.stem}.png"

    trajectory = data["llm_clap"]["trajectory"]

    # (overall_metric, title, per_inst_metric or None)
    metric_specs = [
        ("clap_overall",       "CLAP Overall",         None),
        ("mmd_dsp_overall",    "MMD DSP Overall",       "mmd_dsp_per_inst"),
        ("mmd_fxenc_emb_overall", "MMD FXEnc Emb Overall", "mmd_fxenc_emb_per_inst"),
        ("mmd_clap_emb_overall",  "MMD CLAP Emb Overall",  "mmd_clap_emb_per_inst"),
        ("mmd_vggish_overall",    "MMD VGGish Overall",    "mmd_vggish_per_inst"),
        ("mmd_afx_rep_overall",   "MMD AFx-Rep Overall",   "mmd_afx_rep_per_inst"),
    ]

    ncols = 2
    nrows = (len(metric_specs) + 1) // ncols
    fig, axs = plt.subplots(nrows, ncols, figsize=(14, nrows * 3.5), dpi=150)
    flat_axes = list(axs.flat)

    for ax, (overall_metric, title, per_inst_metric) in zip(flat_axes, metric_specs):
        _plot_metric(ax, trajectory, overall_metric, title, per_inst_metric)

    for ax in flat_axes[len(metric_specs):]:
        ax.axis("off")

    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png)
    plt.close(fig)

    print(f"Recreated plot at: {out_png}")


if __name__ == "__main__":
    main()
