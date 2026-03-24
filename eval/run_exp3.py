"""
eval/run_exp3.py — Run Experiment 3 (Timbre PCA) end-to-end.

Usage:
    python eval/exp/run_exp3.py

Reads all config from eval/config.py.
Results saved to eval/system_outputs/exp3/.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env", override=False)

import eval.config as cfg
from embeddings.clap import CLAPWrapper
from llms.llmclient import LLMClient

from typing import List

from eval.exp.exp3_pca import run_exp3

_WORDS_PCA_SOCIAL_EQ: List[str] = [
    "warm", "soft", "harsh", "calm", "loud", "bright", "heavy", "happy",
]

if __name__ == "__main__":
    print(f"device : {cfg.DEVICE}")
    print(f"instr  : {cfg.INSTRUMENTS}")
    print()

    print("Loading models …")
    llm  = LLMClient()
    clap = CLAPWrapper(device=cfg.DEVICE)
    print("Models loaded.\n")

    results = run_exp3(
        llm_client=llm,
        clap=clap,
        fx_type="eq",
        device=cfg.DEVICE,
        words = _WORDS_PCA_SOCIAL_EQ,
    )

    print("\n=== EXP 3 COMPLETE ===")
    for source, info in results.get("sources", {}).items():
        ev = info.get("explained_variance", [])
        pc1 = f"{ev[0]*100:.1f}%" if ev else "?"
        pc2 = f"{ev[1]*100:.1f}%" if len(ev) > 1 else "?"
        print(f"  {source}: PC1={pc1}, PC2={pc2}")
