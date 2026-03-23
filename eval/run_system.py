"""
eval/run_system.py — Phase 3: System runner for the InstructFX2FX eval pipeline.

Drives run_experiments() for every (word_A → word_B) pair × instrument,
organises outputs on disk, and exposes helpers to collect result paths.
"""

import glob
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

# Make src/ importable from any working directory
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from experimentation.experiment import (
    InstructionSet1,
    Method,
    run_experiments,
)

from eval.config import (
    DEVICE,
    FIXED_INIT,
    FX_EFFECTS,
    N_GRAD_ITER,
    NR_RUNS_PER_FILE,
    SAMPLE_RATE,
    SAVE_INTERVAL,
    SYSTEM_RESULTS_DIR,
)


def run_system_for_pair(
    word_A: str,
    word_B: str,
    instrument: str,
    dry_paths: List[str],
    methods: List[Method],
    llm_client,
    clap,
    output_dir: str = SYSTEM_RESULTS_DIR,
    fx_chain=None,
    effects: List[str] = None,
    n_iter: int = N_GRAD_ITER,
    save_interval: int = SAVE_INTERVAL,
    nr_runs: int = NR_RUNS_PER_FILE,
    device: str = DEVICE,
    fixed_init: bool = FIXED_INIT,
) -> str:
    """Run one (word_A → word_B) pair on all dry clips for one instrument.

    Calls run_experiments() internally; saves all audio to disk.
    Pass multiple methods (e.g. [Method.InstructFX2FX, Method.LLM_LLM]) to
    run them together in the same experiment directory, matching analysis.ipynb.

    Returns:
        pair_run_dir: str  — the experiment_{timestamp}/ directory created by
                             run_experiments(), i.e. the root you should pass to
                             collect_final_audio_paths / collect_trajectory_audio_paths.
    """
    if effects is None:
        effects = FX_EFFECTS

    if fx_chain is None:
        from effects.fx import FXChainFactory
        fx_chain = FXChainFactory.create_fx_chain_from_effects(effects, sample_rate=SAMPLE_RATE, device=device)

    pair_instrument_dir = os.path.join(output_dir, f"{word_A}_to_{word_B}", instrument)

    # Follow analysis.ipynb pattern:
    #   init:   InstructionSet1(target=word_A, context=...)
    #           → "This is violin music. Make this sound more bright."
    #   refine: InstructionSet1(anchor=not word_B, target=word_B, context=...)
    #           → "This is violin music, but the sound is not warm. Make this sound more warm."
    instructionset_initialization = InstructionSet1(
        target=word_A, context=f"{instrument} music"
    )
    instructionset_refinement = InstructionSet1(
        anchor=f"not {word_B}", target=word_B, context=f"{instrument} music"
    )

    results, experiment_dir = run_experiments(
        methods=methods,
        instructionset_initialization=instructionset_initialization,
        instructionset_refinement=instructionset_refinement,
        raw_audio_paths=dry_paths,
        llm_client=llm_client,
        embedding=clap,
        sample_rate=SAMPLE_RATE,
        iterations=n_iter,
        results_dir=pair_instrument_dir,
        device=device,
        nr_of_experiments_per_file=nr_runs,
        fx_chain=fx_chain,
        snapshot_interval=save_interval,
        effects=effects,
        fixed_init=fixed_init,
    )

    if not results:
        raise RuntimeError(
            f"run_experiments() returned no results for {word_A}_to_{word_B}/{instrument}"
        )

    return experiment_dir


def collect_final_audio_paths(pair_run_dir: str, method: Method) -> List[str]:
    """Find all *_final_output.wav files under pair_run_dir for the given method.

    Directory structure produced by run_experiments():
        pair_run_dir/{audio_stem}/{method.value}/run_*/{stem}_final_output.wav

    Returns paths sorted lexicographically (stable across runs).
    """
    print('[DEBUG] Looking for final audio paths in:', pair_run_dir, 'for method:', method.value)
    pattern = os.path.join(
        pair_run_dir, "**", method.value, "run_*", "*_final_output.wav"
    )
    print('[DEBUG] Final audio search pattern:', pattern)
    return sorted(glob.glob(pattern, recursive=True))


def collect_trajectory_audio_paths(pair_run_dir: str) -> Dict[str, List[str]]:
    """Find all trajectory snapshot WAVs from optimization_steps/ directories.

    Only InstructFX2FX (gradient descent) produces these.

    Directory structure:
        pair_run_dir/{stem}/{method}/run_*/intermediate/optimization_steps/*.wav

    Returns:
        trajectory: Dict[str, List[str]]
            key   = snapshot label ("start", "iter_0050", …, "end")
            value = sorted list of WAV paths (one per dry clip / run)
            dict is ordered: start → iter_N (ascending) → end
    """
    pattern = os.path.join(
        pair_run_dir, "**", "optimization_steps", "*.wav"
    )
    all_wavs = sorted(glob.glob(pattern, recursive=True))

    raw: Dict[str, List[str]] = defaultdict(list)
    for wav_path in all_wavs:
        stem = Path(wav_path).stem  # e.g. "start", "iter_0050", "end"
        raw[stem].append(wav_path)

    def _sort_key(step: str):
        if step == "start":
            return (0, 0)
        if step == "end":
            return (2, 0)
        if step.startswith("iter_"):
            m = re.search(r"\d+", step)
            return (1, int(m.group())) if m else (1, 0)
        return (3, 0)

    return dict(sorted(raw.items(), key=lambda kv: _sort_key(kv[0])))
