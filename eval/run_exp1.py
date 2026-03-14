"""
eval/run_exp1.py — Run Experiment 1 (Sequential MMD) end-to-end.

Usage:
    python eval/run_exp1.py

Reads all config from eval/config.py.
Results saved to eval/results/exp1_ours.json and exp1_baseline.json.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env", override=False)

import eval.config as cfg
from embeddings.clap import CLAPWrapper
from experimentation.experiment import Method
from llms.llmclient import LLMClient

from eval.exp1_sequential import run_exp1

if __name__ == "__main__":
    print(f"device : {cfg.DEVICE}")
    print(f"pairs  : {len(cfg.WORD_PAIRS)}")
    print(f"instr  : {cfg.INSTRUMENTS}")
    print(f"dry    : { {k: len(v) for k, v in cfg.DRY_PATHS_BY_INSTRUMENT.items()} }")
    print(f"n_iter : {cfg.N_GRAD_ITER}")
    print()

    print("Loading models …")
    llm  = LLMClient()
    clap = CLAPWrapper(device=cfg.DEVICE)
    print("Models loaded.\n")

    results = run_exp1(
        methods=[Method.InstructFX2FX, Method.LLM_LLM],
        method_names=["ours", "baseline"],
        llm_client=llm,
        clap=clap,
    )

    print("\n=== FINAL RESULTS ===")
    for r in results:
        print(f"  {r['method']}: overall_avg MMD = {r['overall_avg']:.6f}")
