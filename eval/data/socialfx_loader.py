"""
eval/data/socialfx_loader.py — Load GT parameter lists from HuggingFace SocialFX.

Datasets used:
  seungheondoh/socialfx-gen-eval  — maps text words to sample IDs (columns: input, output)
  seungheondoh/socialfx-original  — full dataset with param_values (columns: id, text,
                                     param_values, param_keys, extra)

Usage:
    ids = load_word_ids("warm", "eq")           # [str, ...]
    params = load_params_by_id("eq_0")          # [float] × 40
    all_params = load_all_params_for_word("warm")  # [[float] × 40, ...]
"""

from __future__ import annotations

import functools
from typing import List

from datasets import load_dataset


# ── Lazy dataset cache (avoid re-downloading within a process) ──────────────

@functools.lru_cache(maxsize=4)
def _get_gen_eval_dataset(fx_type: str):
    """Cache-load socialfx-gen-eval for a given split (eq / reverb)."""
    return load_dataset("seungheondoh/socialfx-gen-eval", split=fx_type)


@functools.lru_cache(maxsize=4)
def _get_original_dataset(fx_type: str):
    """Cache-load socialfx-original for a given split (eq / reverb)."""
    return load_dataset("seungheondoh/socialfx-original", split=fx_type)


# ── Public API ──────────────────────────────────────────────────────────────

def load_word_ids(word: str, fx_type: str = "eq") -> List[str]:
    """
    Get all sample IDs for a word from seungheondoh/socialfx-gen-eval.

    The gen-eval dataset has columns `input` (text/word) and `output` (list of
    sample IDs). There is one row per word; `output` is a list of ID strings.

    Input:
        word: str       e.g. "warm"
        fx_type: str    "eq" or "reverb"

    Output:
        ids: List[str]  e.g. ["eq_0", "eq_6", "eq_10", ...]
    """
    ds = _get_gen_eval_dataset(fx_type)
    for row in ds:
        if row["input"] == word:
            output = row["output"]
            # output is a list of ID strings (one row per word)
            if isinstance(output, list):
                return [str(s) for s in output]
            # fallback: single string
            return [str(output)]
    raise ValueError(
        f"No sample IDs found for word='{word}' in socialfx-gen-eval/{fx_type}. "
        f"Check that the word is in the dataset."
    )


def load_params_by_id(sample_id: str, fx_type: str = "eq") -> List[float]:
    """
    Fetch the 40-value EQ parameter array for one sample from
    seungheondoh/socialfx-original.

    Input:
        sample_id: str   e.g. "eq_0"
        fx_type: str     "eq" or "reverb"

    Output:
        params: List[float]   length 40  (raw Audealize EQ curve values)
    """
    ds = _get_original_dataset(fx_type)
    for row in ds:
        if row["id"] == sample_id:
            params = row["param_values"]
            if not isinstance(params, list):
                params = list(params)
            return [float(v) for v in params]
    raise KeyError(f"Sample ID '{sample_id}' not found in socialfx-original/{fx_type}.")


def load_all_params_for_word(word: str, fx_type: str = "eq") -> List[List[float]]:
    """
    Load all available EQ parameter arrays for a word.
    Calls load_word_ids → load_params_by_id for each ID.

    Input:
        word: str
        fx_type: str

    Output:
        params_list: List[List[float]]   e.g. 74 arrays for "warm", each length 40
    """
    ids = load_word_ids(word, fx_type)
    return [load_params_by_id(sample_id, fx_type) for sample_id in ids]
