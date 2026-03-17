"""
eval/report.py — Phase 6: Generate plots from experiment results.

Functions:
    plot_exp2_trajectory  — one PNG per word pair showing MMD trajectory
"""

from __future__ import annotations

import json
import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

import eval.config as cfg

# ── Colours & styles ──────────────────────────────────────────────────────────
_COLOURS = {"piano": "#2196F3", "violin": "#FF5722"}   # blue / orange
_OURS_STYLE = {"linewidth": 1.8, "alpha": 0.9}
_BASE_STYLE = {"linewidth": 1.5, "linestyle": "--", "alpha": 0.7}

# x-position for the "start" label (before iteration 0) and "end" (after last iter)
_START_X = -50
_END_X = cfg.N_GRAD_ITER  # 1000


def _step_to_x(iter_num):
    """Map iter_num (int or None from JSON) to an x-axis value."""
    if iter_num is None:
        return None          # resolved per-step below
    return iter_num


def _build_xs(iter_nums):
    """Convert iter_nums list (with leading/trailing None) to x values."""
    xs = []
    for i, v in enumerate(iter_nums):
        if v is not None:
            xs.append(v)
        elif i == 0:
            xs.append(_START_X)
        else:
            xs.append(_END_X)
    return xs


def plot_exp2_trajectory(
    results: Dict,
    output_dir: str = None,
) -> None:
    """Plot MMD trajectory for OURS vs BASELINE, one PNG per word pair.

    Args:
        results:    dict returned by run_exp2() (or loaded from exp2_trajectory.json).
        output_dir: directory to save PNGs. Defaults to cfg.RESULTS_DIR/exp2.
    """
    if output_dir is None:
        output_dir = os.path.join(cfg.RESULTS_DIR, "exp2")
    os.makedirs(output_dir, exist_ok=True)

    for pair_key, pair_data in results["pairs"].items():
        word_A, word_B = pair_key.split("_to_")

        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
        fig.suptitle(f"{word_A} → {word_B}", fontsize=13, fontweight="bold")

        for ax, target_word, side in zip(
            axes, [word_A, word_B], ["mmd_toward_A", "mmd_toward_B"]
        ):
            ax.set_title(f"MMD toward  '{target_word}'", fontsize=11)
            ax.set_xlabel("Gradient descent iteration")
            ax.set_ylabel("MMD")

            for instrument, inst_data in pair_data.items():
                colour = _COLOURS.get(instrument, "#888888")

                # ── OURS trajectory ──────────────────────────────────────────
                ours = inst_data.get("ours")
                if ours:
                    xs = _build_xs(ours["iter_nums"])
                    ys = ours[side]
                    ax.plot(xs, ys, color=colour, label=f"OURS {instrument}",
                            **_OURS_STYLE)

                # ── BASELINE horizontal line ─────────────────────────────────
                baseline = inst_data.get("baseline")
                if baseline:
                    y_val = baseline[side]
                    ax.axhline(y_val, color=colour, label=f"BASELINE {instrument}",
                               **_BASE_STYLE)

            ax.legend(fontsize=8)
            ax.set_xlim(_START_X - 10, _END_X + 10)

            # Mark x=0 (start of gradient descent)
            ax.axvline(0, color="gray", linewidth=0.8, linestyle=":", alpha=0.5)
            ax.set_xticks([-50, 0, 200, 400, 600, 800, 1000])
            ax.set_xticklabels(["init", "0", "200", "400", "600", "800", "1000"])

        plt.tight_layout()
        out_path = os.path.join(output_dir, f"{pair_key}.png")
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"[report] saved → {out_path}")
