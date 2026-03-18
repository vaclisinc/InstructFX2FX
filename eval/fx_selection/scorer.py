"""
FX Selection Scorer
====================
GT = normalized distribution, filtered to FX with >= min_threshold (default 10%).
Score per word = sum of GT percentages for each predicted FX.

Usage:
    from scorer import score, score_all
    score("warm", {"eq"})         # → 0.493
    score("warm", {"eq", "rev"})  # → 0.906
    score("echo", {"rev"})        # → 1.0
"""

_SAFE = {
    "warm":{"comp":9,"dist":26,"eq":542,"rev":5},"bright":{"comp":4,"dist":5,"eq":521,"rev":1},
    "punch":{"comp":27,"dist":1,"eq":6,"rev":0},"room":{"comp":1,"dist":0,"eq":2,"rev":30},
    "air":{"comp":0,"dist":0,"eq":18,"rev":13},"crunch":{"comp":0,"dist":27,"eq":0,"rev":2},
    "smooth":{"comp":15,"dist":3,"eq":2,"rev":2},"vocal":{"comp":16,"dist":1,"eq":4,"rev":1},
    "clear":{"comp":3,"dist":0,"eq":18,"rev":0},"subtle":{"comp":6,"dist":4,"eq":1,"rev":10},
    "bass":{"comp":3,"dist":4,"eq":13,"rev":0},"fuzz":{"comp":1,"dist":17,"eq":1,"rev":0},
    "nice":{"comp":12,"dist":0,"eq":4,"rev":2},"full":{"comp":3,"dist":0,"eq":9,"rev":4},
    "boom":{"comp":2,"dist":2,"eq":9,"rev":2},"crisp":{"comp":1,"dist":3,"eq":11,"rev":0},
    "sofa":{"comp":15,"dist":0,"eq":0,"rev":0},"soft":{"comp":5,"dist":1,"eq":4,"rev":5},
    "big":{"comp":1,"dist":0,"eq":1,"rev":11},"clean":{"comp":1,"dist":0,"eq":11,"rev":1},
    "thin":{"comp":1,"dist":0,"eq":12,"rev":0},"box":{"comp":1,"dist":0,"eq":8,"rev":3},
    "deep":{"comp":3,"dist":1,"eq":6,"rev":2},"tight":{"comp":7,"dist":0,"eq":4,"rev":1},
    "drum":{"comp":3,"dist":0,"eq":2,"rev":6},"gentle":{"comp":6,"dist":2,"eq":1,"rev":2},
    "thick":{"comp":2,"dist":2,"eq":6,"rev":1},"crushed":{"comp":7,"dist":2,"eq":1,"rev":0},
    "damp":{"comp":1,"dist":1,"eq":1,"rev":7},"harsh":{"comp":1,"dist":4,"eq":5,"rev":0},
    "low":{"comp":0,"dist":0,"eq":10,"rev":0},"presence":{"comp":2,"dist":0,"eq":8,"rev":0},
    "space":{"comp":0,"dist":0,"eq":1,"rev":9},"tin":{"comp":0,"dist":2,"eq":7,"rev":1},
    "acoustic":{"comp":4,"dist":2,"eq":3,"rev":0},"comp":{"comp":9,"dist":0,"eq":0,"rev":0},
    "dream":{"comp":1,"dist":0,"eq":0,"rev":8},"flat":{"comp":5,"dist":1,"eq":3,"rev":0},
    "hall":{"comp":0,"dist":0,"eq":0,"rev":9},"kick":{"comp":4,"dist":1,"eq":4,"rev":0},
    "loud":{"comp":6,"dist":2,"eq":1,"rev":0},"present":{"comp":3,"dist":0,"eq":6,"rev":0},
    "sharp":{"comp":2,"dist":1,"eq":4,"rev":2},"small":{"comp":0,"dist":0,"eq":0,"rev":9},
    "bite":{"comp":0,"dist":0,"eq":8,"rev":0},"click":{"comp":1,"dist":0,"eq":7,"rev":0},
    "cut":{"comp":2,"dist":0,"eq":6,"rev":0},"dark":{"comp":0,"dist":0,"eq":4,"rev":4},
    "echo":{"comp":0,"dist":0,"eq":0,"rev":8},"glue":{"comp":8,"dist":0,"eq":0,"rev":0},
}

_SOCIALFX = {
    "echo":{"comp":118,"eq":0,"rev":2278},"loud":{"comp":261,"eq":21,"rev":1026},
    "tin":{"comp":89,"eq":28,"rev":1095},"low":{"comp":92,"eq":16,"rev":1046},
    "war":{"comp":147,"eq":60,"rev":930},"warm":{"comp":135,"eq":59,"rev":863},
    "church":{"comp":8,"eq":0,"rev":1025},"big":{"comp":55,"eq":1,"rev":878},
    "spacious":{"comp":62,"eq":0,"rev":793},"distant":{"comp":29,"eq":2,"rev":817},
    "deep":{"comp":31,"eq":6,"rev":750},"muffle":{"comp":85,"eq":4,"rev":545},
    "muffled":{"comp":81,"eq":4,"rev":538},"hall":{"comp":7,"eq":0,"rev":577},
    "clear":{"comp":126,"eq":8,"rev":433},"ring":{"comp":24,"eq":7,"rev":506},
    "soft":{"comp":102,"eq":26,"rev":405},"far":{"comp":9,"eq":0,"rev":464},
    "bass":{"comp":43,"eq":1,"rev":417},"distort":{"comp":62,"eq":0,"rev":380},
    "echoing":{"comp":12,"eq":0,"rev":403},"large":{"comp":17,"eq":2,"rev":377},
    "drum":{"comp":11,"eq":1,"rev":312},"hollow":{"comp":14,"eq":2,"rev":307},
    "smooth":{"comp":13,"eq":9,"rev":255},"metal":{"comp":23,"eq":2,"rev":245},
    "sharp":{"comp":55,"eq":7,"rev":195},"full":{"comp":70,"eq":1,"rev":266},
    "room":{"comp":33,"eq":0,"rev":299},"nice":{"comp":30,"eq":3,"rev":296},
    "high":{"comp":37,"eq":4,"rev":278},"strong":{"comp":40,"eq":1,"rev":275},
    "pleasant":{"comp":32,"eq":4,"rev":257},"old":{"comp":18,"eq":36,"rev":228},
    "sad":{"comp":3,"eq":21,"rev":299},
}

ALL_FX = ["comp", "dist", "eq", "rev"]
MIN_THRESHOLD = 0.10  # FX below this are ignored from GT


def _norm(counts):
    padded = {fx: counts.get(fx, 0) for fx in ALL_FX}
    t = sum(padded.values())
    return {fx: padded[fx] / t if t > 0 else 0.0 for fx in ALL_FX}


def _build():
    gt = {}
    for w in set(list(_SAFE) + list(_SOCIALFX)):
        s, x = w in _SAFE, w in _SOCIALFX
        if s and x:
            a, b = _norm(_SAFE[w]), _norm(_SOCIALFX[w])
            gt[w] = {fx: (a[fx] + b[fx]) / 2 for fx in ALL_FX}
        elif s:
            gt[w] = _norm(_SAFE[w])
        else:
            gt[w] = _norm(_SOCIALFX[w])
    return gt


GROUND_TRUTH = _build()


def get_gt(word, min_threshold=MIN_THRESHOLD):
    """Get GT distribution for a word, filtered to FX >= min_threshold."""
    word = word.lower().strip()
    if word not in GROUND_TRUTH:
        return None
    raw = GROUND_TRUTH[word]
    return {fx: round(p, 4) for fx, p in raw.items() if p >= min_threshold}


def score(word, pred_fx, min_threshold=MIN_THRESHOLD):
    """
    Score one prediction.

    Args:
        word:          descriptor word
        pred_fx:       set of predicted FX, e.g. {"eq", "rev"}
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
    # Predicting an FX that was filtered out (below threshold) adds nothing
    total = sum(gt.get(fx, 0.0) for fx in pred_fx)

    return {
        "word": word,
        "score": round(total, 4),
        "gt": {fx: round(p, 4) for fx, p in gt.items()},
        "pred": sorted(pred_fx),
    }


def score_all(predictions, min_threshold=MIN_THRESHOLD, verbose=True):
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
            gt_str = "  ".join(f"{fx}:{p:.0%}" for fx, p in sorted(r["gt"].items(), key=lambda x: -x[1]))
            pred_str = ", ".join(r["pred"])
            print(f"  {word:>12s}  {gt_str:>35s}  {pred_str:>15s}  {r['score']:6.2f}")

    n = len(results)
    avg = round(sum(r["score"] for r in results) / n, 4) if n else 0

    if verbose:
        print(f"\n  {n} words scored | Avg score: {avg:.3f}")

    return {"n_words": n, "avg_score": avg, "per_word": results}


def get_words():
    return sorted(GROUND_TRUTH.keys())