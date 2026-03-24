"""
eval/run_metrics.py — Run evaluation metrics.

Usage:
    python eval/run_metrics.py
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

import eval.config as cfg
from metrics.run_metrics import run_metrics_against_gt, run_metrics_for_comparison
from datetime import datetime
import json

if __name__ == "__main__":
    gt_folder = cfg.GT_BANK_DIR_ONESHOT
    system_outputs = cfg.SYSTEM_RESULTS_DIR

    # Example: compare exp3 LLM and CLAP outputs against GT
    exp3_llm = os.path.join(system_outputs, "exp3", "llm")
    exp3_clap = os.path.join(system_outputs, "exp3", "clap_loss")

    # ── Metrics against GT ───────────────────────────────────────────────
    gt_results = run_metrics_against_gt(
        gt_folder=gt_folder,
        test_folders={
            "LLM Init": exp3_llm,
            "CLAP Loss": exp3_clap,
        },
    )

    # ── Comparative metrics (no GT) ──────────────────────────────────────
    comp_results = run_metrics_for_comparison(
        technique_folders={
            "LLM Init": exp3_llm,
            "CLAP Loss": exp3_clap,
        },
    )

    # Create results directory with timestamp
    results_dir = Path(ROOT) / "eval" / "results" / "general"
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"results_{timestamp}.json"

    # Combine results and save
    combined_results = {
        "timestamp": datetime.now().isoformat(),
        "gt_metrics": gt_results,
        "comparative_metrics": comp_results,
    }

    with open(results_file, "w") as f:
        json.dump(combined_results, f, indent=2)

    print(f"Results saved to {results_file}")
