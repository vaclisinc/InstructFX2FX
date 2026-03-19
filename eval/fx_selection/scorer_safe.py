"""
FX Selection Scorer (SAFE-only)
===============================
GT = normalized distribution derived ONLY from the SAFE dataset, filtered to FX with
>= min_threshold (default 10%).

Score per word = sum of GT percentages for each predicted FX.

Usage:
    from scorer_safe import score, score_all
    score("warm", {"eq"})         # → 0.9398 (SAFE-only)
    score("echo", {"rev"})        # → 1.0
"""

from __future__ import annotations

from typing import Iterable

# Import the SAFE dataset from the mixed scorer so the numbers stay in sync.
# Support both:
# - package import:   from eval.fx_selection import scorer_safe
# - script import:    from scorer_safe import ...
try:
    from .scorer import ALL_FX, MIN_THRESHOLD, _SAFE
except ImportError:  # pragma: no cover
    from scorer import ALL_FX, MIN_THRESHOLD, _SAFE


def _norm(counts: dict) -> dict[str, float]:
    padded = {fx: counts.get(fx, 0) for fx in ALL_FX}
    t = sum(padded.values())
    return {fx: padded[fx] / t if t > 0 else 0.0 for fx in ALL_FX}


def _build() -> dict[str, dict[str, float]]:
    return {w: _norm(_SAFE[w]) for w in _SAFE}


GROUND_TRUTH = _build()


def get_gt(word: str, min_threshold: float = MIN_THRESHOLD):
    """Get GT distribution for a word, filtered to FX >= min_threshold."""
    word = word.lower().strip()
    if word not in GROUND_TRUTH:
        return None
    raw = GROUND_TRUTH[word]
    return {fx: round(p, 4) for fx, p in raw.items() if p >= min_threshold}


def score(word: str, pred_fx: Iterable[str], min_threshold: float = MIN_THRESHOLD):
    """
    Score one prediction.

    Args:
        word:          descriptor word
        pred_fx:       iterable of predicted FX, e.g. {"eq", "rev"}
        min_threshold: ignore FX below this in GT (default 0.10)

    Returns:
        dict with word, score, gt (filtered distribution), pred
    """
    word = word.lower().strip()
    if word not in GROUND_TRUTH:
        return {"word": word, "score": None, "error": f"'{word}' not found"}

    # Filter GT to FX above threshold
    raw = GROUND_TRUTH[word]
    gt = {fx: p for fx, p in raw.items() if p >= min_threshold}

    # Score = sum of GT weight for each predicted FX that's in the filtered GT
    total = sum(gt.get(fx, 0.0) for fx in set(pred_fx))

    return {
        "word": word,
        "score": round(total, 4),
        "gt": {fx: round(p, 4) for fx, p in gt.items()},
        "pred": sorted(set(pred_fx)),
    }


def score_all(predictions: dict[str, Iterable[str]], min_threshold: float = MIN_THRESHOLD, verbose: bool = True):
    """
    Score a dict of {word: set_of_fx}. Prints per-word results.

    Returns:
        dict with per_word results and avg_score
    """
    results = []

    if verbose:
        print(f"\n{'Word':>12s}  {'GT (>=10%)':>35s}  {'Pred':>15s}  {'Score':>6s}")
        print("-" * 75)

    for word, pred_fx in sorted(predictions.items()):
        r = score(word, pred_fx, min_threshold)
        if r.get("error"):
            if verbose:
                print(f"  {word:>12s}  NOT FOUND")
            continue
        results.append(r)

        if verbose:
            gt_str = "  ".join(
                f"{fx}:{p:.0%}" for fx, p in sorted(r["gt"].items(), key=lambda x: -x[1])
            )
            pred_str = ", ".join(r["pred"])
            print(f"  {word:>12s}  {gt_str:>35s}  {pred_str:>15s}  {r['score']:6.2f}")

    n = len(results)
    avg = round(sum(r["score"] for r in results) / n, 4) if n else 0

    if verbose:
        print(f"\n  {n} words scored | Avg score: {avg:.3f}")

    return {"n_words": n, "avg_score": avg, "per_word": results}


def get_words():
    return sorted(GROUND_TRUTH.keys())

