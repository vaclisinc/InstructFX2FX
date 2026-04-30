"""End-to-end integration test for the 3-layer pipeline.

Usage:
    cd text2preset
    .venv/bin/python tests/test_pipeline.py

Requires:
    - OPENROUTER_API_KEY in .env
    - dry_audio/piano/piano.wav
    - CLAP model weights (auto-downloaded on first run)
"""

import sys
import os
import traceback

import soundfile as sf
import torch
import torchaudio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dotenv import load_dotenv

load_dotenv()

from llms.llmclient import LLMClient
from embeddings.clap import CLAPWrapper
from pipeline.orchestrator import Orchestrator
from session.session import Session

AUDIO_PATH = os.path.join(os.path.dirname(__file__), "..", "dry_audio", "piano", "piano.wav")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
N_ITERATIONS = 10
ALL_FX = ["eq", "comp", "rev", "dist", "delay", "pitchshift", "bitcrush"]


def load_audio(path: str) -> torch.Tensor:
    import numpy as np
    data, sr = sf.read(path, always_2d=True)  # [T, C]
    waveform = torch.from_numpy(data.T).float()  # [C, T]
    if sr != 44100:
        waveform = torchaudio.functional.resample(waveform, sr, 44100)
    return waveform.unsqueeze(0)  # [1, C, T]


def save_audio(tensor: torch.Tensor, path: str):
    audio_np = tensor.squeeze().cpu().numpy()
    sf.write(path, audio_np.T if audio_np.ndim == 2 else audio_np, 44100)
    print(f"  Saved: {path}")


def validate_output(result: dict, turn: str, session: Session) -> list[str]:
    errors = []
    if "fx_chain" not in result:
        errors.append(f"[{turn}] missing 'fx_chain' key")
    elif not isinstance(result["fx_chain"], list):
        errors.append(f"[{turn}] 'fx_chain' is not a list")

    if "params" not in result:
        errors.append(f"[{turn}] missing 'params' key")
    elif not isinstance(result["params"], dict):
        errors.append(f"[{turn}] 'params' is not a dict")

    return errors


def run_case(name: str, turns: list[tuple[str, list[str]]], orch: Orchestrator, audio: torch.Tensor) -> list[str]:
    """Run a multi-turn test case.

    Each turn is (instruction, expected_fx) where expected_fx is a minimum
    set — the FX Selector may pick additional effects beyond these.
    """
    print(f"\n{'='*60}")
    print(f"Case {name}")
    print(f"{'='*60}")
    errors = []
    session = Session(available_fx=ALL_FX)
    case_dir = os.path.join(OUTPUT_DIR, name)
    os.makedirs(case_dir, exist_ok=True)

    for i, (instruction, expected_fx) in enumerate(turns, 1):
        turn_label = f"{name}/turn{i}"
        print(f"\n--- Turn {i}: \"{instruction}\" ---")
        try:
            result = orch.run(instruction, session, audio)
        except Exception as e:
            errors.append(f"[{turn_label}] CRASHED: {e}")
            traceback.print_exc()
            break

        errs = validate_output(result, turn_label, session)
        errors.extend(errs)
        if errs:
            continue

        print(f"  FX chain: {result['fx_chain']}")
        print(f"  Params keys: {list(result['params'].keys())}")

        for fx_name in expected_fx:
            if fx_name not in result["fx_chain"]:
                errors.append(f"[{turn_label}] expected '{fx_name}' in fx_chain, got {result['fx_chain']}")

    # Check session accumulation
    print(f"\n  Session accumulated params: {list(session.current_params.keys())}")
    return errors


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading audio...")
    audio = load_audio(AUDIO_PATH)
    print(f"  Audio shape: {audio.shape}")

    print("Initializing LLM client...")
    llm = LLMClient()

    print("Initializing CLAP model...")
    clap = CLAPWrapper(device="cpu")

    print("Creating Orchestrator...")
    orch = Orchestrator(llm, clap, device="cpu", n_iterations=N_ITERATIONS)

    all_errors = []

    # Case A — EQ only (DASP path)
    # Turn 1: "bright" should select at least eq
    # Turn 2: "warmer" with eq already in session → Case 1 (reuse + re-optimize)
    errs = run_case("A", [
        ("bright", ["eq"]),
        ("warmer", ["eq"]),
    ], orch, audio)
    all_errors.extend(errs)

    # Case B — Mixed DASP + Pedalboard
    # Turn 1: "make it sound like a church" should select at least rev
    # Turn 2: "add some grit" should add dist (Case 3: new + existing)
    # Turn 3: "too harsh, soften it" should reuse existing FX (Case 1)
    errs = run_case("B", [
        ("make it sound like a church", ["rev"]),
        ("add some grit", ["dist"]),
        ("too harsh, soften it", []),
    ], orch, audio)
    all_errors.extend(errs)

    # Case C — Pedalboard only
    # "distorted and crushed" should select at least dist
    errs = run_case("C", [
        ("distorted and crushed", ["dist"]),
    ], orch, audio)
    all_errors.extend(errs)

    print(f"\n{'='*60}")
    if all_errors:
        print(f"FAILED — {len(all_errors)} error(s):")
        for e in all_errors:
            print(f"  {e}")
        sys.exit(1)
    else:
        print("ALL CASES PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
