"""
Compare FX selection scorers.

Runs the same test vectors through:
- scorer       (mixed SAFE + SOCIALFX GT)
- scorer_safe  (SAFE-only GT)

This is meant for quick sanity checks and side-by-side comparison.
"""

from __future__ import annotations

from typing import Iterable

import scorer as mixed
import scorer_safe as safe


def _norm(counts: dict) -> dict[str, float]:
    padded = {fx: counts.get(fx, 0) for fx in mixed.ALL_FX}
    t = sum(padded.values())
    return {fx: (padded[fx] / t if t > 0 else 0.0) for fx in mixed.ALL_FX}


def _filtered_gt(gt: dict[str, float], min_threshold: float):
    return {fx: round(p, 4) for fx, p in gt.items() if p >= min_threshold}


def _fmt_gt(gt):
    if gt is None:
        return "None"
    if not gt:
        return "{}"
    items = ", ".join(f"{k}:{v:.4f}" for k, v in sorted(gt.items(), key=lambda kv: (-kv[1], kv[0])))
    return "{" + items + "}"


def _score(mod, word: str, pred: Iterable[str]):
    r = mod.score(word, set(pred))
    if r.get("error"):
        return None
    return r["score"]


def main():
    print("=== Assertions (basic correctness) ===")

    # SAFE-only scorer should only contain SAFE words.
    safe_words = set(safe.get_words())
    mixed_words = set(mixed.get_words())
    assert safe_words.issubset(mixed_words), "SAFE words should be subset of mixed words"

    # SAFE-only GT should match the raw SAFE counts (normalized + thresholded).
    for w in safe_words:
        expected = _filtered_gt(_norm(mixed._SAFE[w]), mixed.MIN_THRESHOLD)
        got = safe.get_gt(w, mixed.MIN_THRESHOLD)
        assert got == expected, f"SAFE GT mismatch for '{w}': got {got}, expected {expected}"

    # Words that exist only in SOCIALFX should be missing in SAFE-only.
    social_only = set(mixed._SOCIALFX.keys()) - set(mixed._SAFE.keys())
    if social_only:
        sample = sorted(list(social_only))[:5]
        for w in sample:
            assert safe.get_gt(w) is None, f"Expected SAFE-only to not contain '{w}'"

    print("OK\n")

    print("=== Side-by-side GT (>=10%) ===")
    words = ["warm", "echo", "bright", "room", "punch", "air", "dark", "clear", "soft", "church", "hall"]
    for w in words:
        m_gt = mixed.get_gt(w)
        s_gt = safe.get_gt(w)
        print(f"{w:>10s} | mixed { _fmt_gt(m_gt):<45s} | safe { _fmt_gt(s_gt) }")

    print("\n=== Side-by-side scores ===")
    tests = [
        ("warm", {"eq"}),
        ("warm", {"rev"}),
        ("warm", {"eq", "rev"}),
        ("warm", {"eq", "rev", "comp"}),
        ("echo", {"rev"}),
        ("echo", {"eq"}),
        ("bright", {"eq"}),
        ("room", {"rev"}),
        ("room", {"rev", "eq"}),
        ("punch", {"comp"}),
        ("punch", {"comp", "eq"}),
        ("air", {"eq"}),
        ("air", {"eq", "rev"}),
        ("dark", {"eq"}),
        ("dark", {"eq", "rev"}),
        ("clear", {"eq"}),
        ("clear", {"eq", "rev", "comp"}),
        ("church", {"rev"}),
        ("hall", {"rev"}),
    ]
    for word, pred in tests:
        m = _score(mixed, word, pred)
        s = _score(safe, word, pred)
        pred_str = ",".join(sorted(pred))
        m_str = "NOT FOUND" if m is None else f"{m:.4f}"
        s_str = "NOT FOUND" if s is None else f"{s:.4f}"
        print(f"{word:>10s} pred={pred_str:<14s} | mixed {m_str:>9s} | safe {s_str:>9s}")


if __name__ == "__main__":
    main()

