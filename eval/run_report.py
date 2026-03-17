"""
eval/run_report.py — Generate all plots from saved experiment results.

Usage:
    python eval/run_report.py

Reads eval/results/exp2_trajectory.json.
Saves PNGs to eval/results/exp2/{pair_key}.png.
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import eval.config as cfg
from eval.report import plot_exp2_trajectory

if __name__ == "__main__":
    json_path = os.path.join(cfg.RESULTS_DIR, "exp2_trajectory.json")
    if not os.path.exists(json_path):
        print(f"ERROR: {json_path} not found — run eval/run_exp2.py first")
        sys.exit(1)

    with open(json_path) as f:
        results = json.load(f)

    plot_exp2_trajectory(results)
    print("Done.")
