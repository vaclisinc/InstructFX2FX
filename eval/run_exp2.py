"""
eval/run_exp2.py — Run Experiment 2 (Trajectory MMD) end-to-end.

Usage:
    python eval/run_exp2.py

Reads all config from eval/config.py.
Requires system outputs already on disk (run eval/run_exp1.py first).
Results saved to eval/results/exp2_trajectory.json.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env", override=False)

import eval.config as cfg
from eval.exp2_trajectory import run_exp2

if __name__ == "__main__":
    print(f"pairs  : {len(cfg.WORD_PAIRS)}")
    print(f"instr  : {cfg.INSTRUMENTS}")
    print()

    results = run_exp2()

    n_pairs = len(results.get("pairs", {}))
    print(f"\n=== DONE: {n_pairs} pairs saved to eval/results/exp2_trajectory.json ===")
