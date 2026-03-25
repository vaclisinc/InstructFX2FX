"""
eval/run_exp1.py — Run Experiment 1 (Sequential MMD) end-to-end.

Usage:
    python eval/run_exp5.py

Reads all config from eval/config.py.
Results saved to eval/results/exp5.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env", override=False)

import glob

import eval.config as cfg
from embeddings.clap import CLAPWrapper
from runners.runners import Method
from llms.llmclient import LLMClient
from eval.exp.exp5_llm_init import run_exp5

if __name__ == "__main__":
    print(f"device : {cfg.DEVICE}")
    print(f"pairs  : {len(cfg.WORD_PAIRS)}")
    print(f"instr  : {cfg.INSTRUMENTS}")
    dry_paths = {i: sorted(glob.glob(os.path.join(cfg.DRY_AUDIO_DIR, i, "*.wav"))) for i in cfg.INSTRUMENTS}
    print(f"dry    : { {k: len(v) for k, v in dry_paths.items()} }")
    print(f"n_iter : {cfg.N_GRAD_ITER}")
    print()

    print("Loading models …")
    llm  = LLMClient()
    clap = CLAPWrapper(device=cfg.DEVICE)
    print("Models loaded.\n")

    results = run_exp5(
        llm_client=llm,
        clap=clap,
        words=cfg._WORDS_PCA_SOCIAL_EQ,
        device = cfg.DEVICE,
        n_clap_calls = 100,
        effects = ['eq']
    )

    print("\n=== FINAL RESULTS ===")
    print(results)
