"""
scripts/generate_demo_sessions.py — Generate fresh, high-quality demo sessions by
actually running the InstructFX2FX pipeline (Layer 1 plan + Layer 3a CLAP gradient
descent), then saving them in the apps artifact format that build_demo_site.py reads.

  - FX restricted to EQ + reverb only (no Pedalboard / Bayesian path).
  - A strong, current LLM (claude-opus-4.1) for both effect selection and the
    parameter initialization the optimizer starts from.
  - Multi-turn (>= 3 turns) creative spatial / timbral transformations.

Usage:
    .venv/bin/python scripts/generate_demo_sessions.py cathedral
    .venv/bin/python scripts/generate_demo_sessions.py all
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "apps" / "web_api"))

from app.service import WebDemoService  # noqa: E402

LLM_MODEL = "openai/gpt-5.5"
SETTINGS = {
    "llm_model": LLM_MODEL,
    "n_iterations": 120,
    "learning_rate": 0.02,
    "snapshot_interval": 15,
}
AVAILABLE_FX = ["eq", "rev"]

EXAMPLES = {
    "cathedral": {
        "name": "[demo] cathedral",
        "audio": ROOT / "dry_audio" / "piano" / "piano.wav",
        "turns": [
            "make it sound like it is being played in a vast stone cathedral",
            "make it even bigger and more spacious, with a longer reverberant tail",
            "warm the tone up and soften the bright edges",
        ],
    },
    "underwater": {
        "name": "[demo] underwater",
        "audio": ROOT / "dry_audio" / "piano" / "piano.wav",
        "turns": [
            "make it sound like it is playing underwater, deep in a swimming pool",
            "make it more muffled and submerged",
            "open up a slow, cavernous echo around it",
        ],
    },
    "dream": {
        "name": "[demo] dream",
        "audio": ROOT / "dry_audio" / "piano" / "piano.wav",
        "turns": [
            "give it a dreamy, ethereal, floating atmosphere",
            "make it shimmer, brighter and airier",
            "pull it far into the distance, like a fading memory",
        ],
    },
}


def wait(service: WebDemoService, run_id: str, timeout: float = 1800) -> str:
    start = time.time()
    last = -1
    while time.time() - start < timeout:
        run = service.get_run(run_id)
        if run.status in ("completed", "failed"):
            return run.status
        if run.current_iteration != last:
            last = run.current_iteration
            print(f"      iter {run.current_iteration}/{run.total_iterations}  ({run.progress:.0f}%)", flush=True)
        time.sleep(3)
    return "timeout"


def run_example(service: WebDemoService, key: str) -> str:
    spec = EXAMPLES[key]
    print(f"\n=== {key} :: {spec['name']} ===", flush=True)
    rec = service.create_session(available_fx=list(AVAILABLE_FX), name=spec["name"])
    service.attach_audio(rec.session_id, spec["audio"], spec["audio"].name)
    for i, instr in enumerate(spec["turns"], 1):
        print(f"  turn {i}: {instr!r}", flush=True)
        run = service.submit_run(rec.session_id, instr, dict(SETTINGS))
        status = wait(service, run.run_id)
        result = service.get_run(run.run_id).result or {}
        fx = result.get("fx_chain")
        route = result.get("metadata", {}).get("route_case")
        traj = len(result.get("trajectory", []))
        if status != "completed":
            print(f"    !! turn {i} {status}: {service.get_run(run.run_id).error}", flush=True)
            break
        print(f"    done: route={route} fx={fx} checkpoints={traj}", flush=True)
    print(f"  SESSION_ID={rec.session_id}", flush=True)
    return rec.session_id


def main():
    keys = sys.argv[1:] or ["cathedral"]
    if keys == ["all"]:
        keys = list(EXAMPLES)
    service = WebDemoService()
    ids = {}
    for k in keys:
        ids[k] = run_example(service, k)
    print("\n=== generated session ids ===")
    for k, sid in ids.items():
        print(f"  {k}: {sid}")


if __name__ == "__main__":
    main()
