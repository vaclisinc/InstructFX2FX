"""
scripts/build_demo_site.py — Build the static demo site from generated multi-turn
sessions (scripts/generate_demo_sessions.py), all on the EQ + reverb gradient-descent
path.

Every clip (source, per-turn result, every trajectory checkpoint) is RE-RENDERED
from its stored parameters on the full dry audio, using the FX chain directly with
[0,1]-normalized params (no extra sigmoid). This guarantees:
  - turn N's trajectory "start" is byte-identical to turn N-1's result (same params),
  - all clips are full length and consistently rendered,
  - none of the optimizer's internal 5-second / double-sigmoid quirks leak in.

Usage:  .venv/bin/python scripts/build_demo_site.py
Requires: ffmpeg on PATH.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(ROOT / ".env")  # so REVERB_NUM_SAMPLES in .env applies to the build too

from effects.fx import FXChainFactory  # noqa: E402
from utilities.fx_processing import fx_initial_params_to_tensor  # noqa: E402

ARTIFACTS = ROOT / "apps" / "web_api" / "artifacts"
SESS_DIR = ARTIFACTS / "sessions"
OUT_DIR = ROOT / "presentation" / "demo"
AUDIO_OUT = OUT_DIR / "audio"
SR = 44100
PB_FX = {"comp", "dist", "delay", "pitchshift", "bitcrush"}
ITER_RE = re.compile(r"iter_(\d+)")
# canonical order: (canonical key, dict key, registry key). Chain + tensor share it.
CANON = [("eq", "EQ", "eq"), ("rev", "Reverb", "reverb")]

META = {
    "cathedral": ("From the chapel to the cosmos",
                  "A piano placed in a stone cathedral, opened into a vast space, then warmed. "
                  "The classic reverb arc, plus a closing EQ move."),
    "underwater": ("Underwater",
                   "A piano submerged and pulled deeper, then wrapped in a slow cavernous echo. "
                   "A creative spatial transformation built only from reverb and EQ."),
    "dream": ("A fading dream",
              "An ethereal, shimmering atmosphere that brightens, then drifts into the distance."),
}
ORDER = ["cathedral", "underwater", "dream"]

_CHAIN_CACHE: dict = {}


def render(params: dict, fx_chain: list, dry: torch.Tensor) -> torch.Tensor:
    """Render canonical-key params on dry audio. [0,1] params, no extra sigmoid."""
    present = [(c, dk, rk) for c, dk, rk in CANON
               if c in fx_chain and isinstance(params.get(c), dict)]
    if not present:
        return dry
    key = tuple(rk for _, _, rk in present)
    chain = _CHAIN_CACHE.setdefault(key, FXChainFactory.create_fx_chain_from_effects(list(key)))
    cfg = {dk: params[c] for c, dk, _ in present}
    t = fx_initial_params_to_tensor(cfg, device="cpu").clamp(0.0, 1.0)  # FX expects [0,1]
    # The DASP reverb is stochastic (random diffusion/modulation noise), so the
    # same params render differently each call. Seed before every render so
    # identical params give identical audio: turn N's "start" then matches turn
    # N-1's result exactly, and the trajectory morphs smoothly.
    torch.manual_seed(0)
    np.random.seed(0)
    with torch.no_grad():
        return chain(dry, t).cpu()


def write_mp3(audio: torch.Tensor, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    a = audio.squeeze(0).detach().cpu().numpy().T  # (samples, ch)
    peak = float(np.max(np.abs(a))) or 1.0
    a = (a / peak) * 0.97
    tmp = dst.with_suffix(".tmp.wav")
    sf.write(tmp, a, SR, subtype="FLOAT")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp),
                    "-ar", str(SR), "-codec:a", "libmp3lame", "-b:a", "144k", str(dst)], check=True)
    tmp.unlink()


def traj_key(item):
    label = item.get("label", "")
    if label == "start":
        return (0, 0)
    if label == "end":
        return (2, 0)
    m = ITER_RE.search(label)
    return (1, int(m.group(1)) if m else 0)


def routing_for(cur, prev):
    prev_s = set(prev)
    new = [f for f in cur if f not in prev_s]
    if not prev_s:
        return "Initialize-only", new, []
    if new:
        return "Mixed reuse-and-initialize", new, [f for f in cur if f in prev_s]
    return "Reuse-and-optimize", [], list(cur)


def backend_for(fx_chain):
    return "Bayesian optimization" if any(f in PB_FX for f in fx_chain) else "Gradient descent"


def discover():
    found = {}
    for sdir in SESS_DIR.iterdir():
        sj = sdir / "session.json"
        if not sj.exists():
            continue
        m = re.match(r"\[demo\]\s+(\w+)", json.loads(sj.read_text()).get("name", ""))
        if m and m.group(1) in META:
            found[m.group(1)] = sdir
    return [(k, found[k]) for k in ORDER if k in found]


def load_dry(sdir: Path) -> torch.Tensor:
    wav = next((sdir / "audio").glob("input.*"))
    a, sr = sf.read(wav, always_2d=True)
    t = torch.from_numpy(a.T).unsqueeze(0).float()
    return t


def build():
    if AUDIO_OUT.exists():
        shutil.rmtree(AUDIO_OUT)
    AUDIO_OUT.mkdir(parents=True, exist_ok=True)

    examples = []
    for key, sdir in discover():
        dry = load_dry(sdir)
        title, blurb = META[key]
        runs = []
        for rdir in (sdir / "runs").iterdir():
            if (rdir / "run.json").exists() and (rdir / "result.json").exists():
                rj = json.loads((rdir / "run.json").read_text())
                res = json.loads((rdir / "result.json").read_text())
                if rj.get("status") == "completed":
                    runs.append((rj["created_at"], rj, res))
        runs.sort(key=lambda x: x[0])

        write_mp3(dry, AUDIO_OUT / key / "source.mp3")
        turns, prev_fx = [], []
        for i, (_, rj, res) in enumerate(runs, 1):
            fx_chain = res["fx_chain"]
            write_mp3(render(res["params"], fx_chain, dry), AUDIO_OUT / key / f"t{i}_result.mp3")

            traj = []
            for cp in sorted(res.get("trajectory", []), key=traj_key):
                label = cp.get("label", "iter")
                write_mp3(render(cp.get("params", {}), fx_chain, dry), AUDIO_OUT / key / f"t{i}_{label}.mp3")
                traj.append({"label": label, "iteration": cp.get("iteration"),
                             "audio": f"audio/{key}/t{i}_{label}.mp3"})

            mode, new, reused = routing_for(fx_chain, prev_fx)
            prev_fx = fx_chain
            turns.append({
                "n": i, "instruction": rj["instruction"],
                "fx_chain": fx_chain, "fx_new": new, "fx_reused": reused,
                "routing": mode, "backend": backend_for(fx_chain),
                "result": f"audio/{key}/t{i}_result.mp3", "trajectory": traj,
            })
            print(f"  {key:10s} t{i}: {mode:26s} fx={fx_chain} traj={len(traj)}")

        examples.append({"id": key, "kind": "reverb", "title": title, "blurb": blurb,
                         "llm": "gpt-5.5", "n_turns": len(turns),
                         "source": f"audio/{key}/source.mp3", "turns": turns})

    (OUT_DIR / "data.json").write_text(json.dumps({"examples": examples}, indent=2))
    n = sum(len(t["trajectory"]) + 1 for e in examples for t in e["turns"]) + len(examples)
    print(f"\n✓ {len(examples)} examples (~{n} clips) -> {OUT_DIR/'data.json'}")


if __name__ == "__main__":
    build()
