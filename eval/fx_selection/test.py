from scorer import score, score_all, get_gt

print("=== GT distributions (filtered to >=10%) ===\n")
for w in ["warm", "echo", "bright", "room", "punch", "air", "dark", "clear", "soft"]:
    gt = get_gt(w)
    print(f"  {w:>10s}  {gt}")

print("\n\n=== Single word scores ===\n")
tests = [
    ("warm",   {"eq"}),
    ("warm",   {"rev"}),
    ("warm",   {"eq", "rev"}),
    ("warm",   {"eq", "rev", "comp"}),   # comp is <10%, adds nothing
    ("echo",   {"rev"}),
    ("echo",   {"eq"}),                   # eq is 0% for echo, adds nothing
    ("bright", {"eq"}),
    ("room",   {"rev"}),
    ("room",   {"rev", "eq"}),            # eq is <10% for room, adds nothing
    ("punch",  {"comp"}),
    ("punch",  {"comp", "eq"}),
    ("air",    {"eq"}),
    ("air",    {"eq", "rev"}),
    ("dark",   {"eq"}),
    ("dark",   {"eq", "rev"}),
    ("clear",  {"eq"}),
    ("clear",  {"eq", "rev", "comp"}),
]
for word, pred in tests:
    r = score(word, pred)
    print(f"  {word:>10s}  pred={str(sorted(pred)):>25s}  score={r['score']:.3f}")

print("\n\n=== Batch test ===")
score_all({
    "warm": {"eq"},
    "bright": {"eq"},
    "echo": {"rev"},
    "hall": {"rev"},
    "punch": {"comp"},
    "fuzz": {"dist"},
    "church": {"rev"},
    "room": {"rev"},
    "clear": {"eq"},
    "thin": {"eq"},
    "dark": {"eq", "rev"},
    "space": {"rev"},
    "air": {"eq", "rev"},
    "soft": {"comp", "rev"},
})