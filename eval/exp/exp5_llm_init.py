# test how CLAP deals best with composite instructions


from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import tqdm

from src.configurations.config import OptimizationMethod
from src.effects.fx import FXChainFactory
from src.prompts.instruction import InstructionSet1
from src.runners.runners import run_LLM, run_CLAP
from src.training.parameterengine import ParameterEngine

_ROOT = Path(__file__).resolve().parent.parent.parent   # project root
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

import eval.config as cfg
from eval.helpers import _dry_paths

from src.metrics.dsp_feature_metrics import (
    TIMBRE_FEATURE_NAMES,
    extract_timbre_features,
)

import torch
import soundfile as sf
import librosa
from src.configurations.config import (
    Config,
    LossFunction,
    ParameterInitializationMethod,
)
from src.training.parameterengine import ParameterEngine
from src.effects.fx import FXChainFactory
from src.prompts.prompt import Prompt, PromptFactory
from src.utilities.audio_processing import _ensure_bct
from src.metrics.run_metrics import run_all_metrics
from utilities.audio_processing import _process_and_save_audio_and_params


# ── Output directories (under eval/) ─────────────────────────────────────────

_EXP5_SYSTEM_RESULTS_DIR = os.path.join(cfg.SYSTEM_RESULTS_DIR, "exp5")
_EXP5_RESULTS_DIR = os.path.join(cfg.RESULTS_DIR, "exp5")

_EXP5_LLMCLAP_DIR = os.path.join(_EXP5_SYSTEM_RESULTS_DIR, "llm_clap")
_EXP5_LLM_DIR = os.path.join(_EXP5_SYSTEM_RESULTS_DIR, "llm")


def run_exp5(
    llm_client,
    words: List[str],
    clap,
    device: str = cfg.DEVICE,
    n_clap_calls: int = 100,
    effects: List[str] = None,
) -> Dict:

    unique_words = sorted(words)
    parameter_engine = ParameterEngine()

    fx_chain = FXChainFactory.create_fx_chain_from_effects(
        effects, sample_rate=cfg.SAMPLE_RATE, device=device,
    )

    for word in unique_words[:1]:
        for instrument in cfg.INSTRUMENTS:
            dry = _dry_paths(instrument)
            if not dry:
                continue

            name = instrument.replace(" ", "_")
            llm_out_dir  = os.path.join(_EXP5_LLM_DIR,    word, instrument)
            clap_out_dir = os.path.join(_EXP5_LLMCLAP_DIR, word, instrument)

            llm_done  = os.path.isdir(llm_out_dir)  and any(f.endswith(".wav") for f in os.listdir(llm_out_dir))
            clap_done = os.path.isdir(clap_out_dir) and any(f.endswith(".wav") for f in os.listdir(clap_out_dir))

            if llm_done and clap_done:
                print(f"[exp5] SKIP {word}/{instrument}: both methods exist")
                continue

            os.makedirs(llm_out_dir,  exist_ok=True)
            os.makedirs(clap_out_dir, exist_ok=True)
            print(f"[exp5] RUN  {word}/{instrument}")

            for dry_idx, dry_path in enumerate(dry):
                audio_np, sr = sf.read(dry_path)
                if audio_np.ndim > 1:
                    audio_np = audio_np.mean(axis=1)
                if sr != cfg.SAMPLE_RATE:
                    audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=cfg.SAMPLE_RATE)
                audio_t = torch.from_numpy(audio_np).float().to(device)
                audio_t = _ensure_bct(audio_t)

                instructionset = InstructionSet1(
                    target=word,
                    context=f"This is a {instrument} playing."
                )

                # ── LLM ──────────────────────────────────────────────────────
                if not llm_done or not clap_done:
                    llm_params_tensor, llm_params_dict, _ = run_LLM(
                        audio=audio_t,
                        fx_chain=fx_chain,
                        instructionset=instructionset,
                        llm_client=llm_client,
                        experiment_dir=os.path.join(llm_out_dir),
                        sample_rate=sr,
                        device=device,
                        effects=effects,
                    )
                    _process_and_save_audio_and_params(
                        audio_tensor=audio_t, params_tensor=llm_params_tensor, params_dict=llm_params_dict, fx_chain=fx_chain, sample_rate=cfg.SAMPLE_RATE, output_dir=Path(llm_out_dir), name=f"{name}_{dry_idx}"
                    )

                    clap_params_tensor, clap_params_dict, _ = run_CLAP(
                        audio=audio_t,
                        fx_chain=fx_chain,
                        instructionset=instructionset,
                        embedding=clap,
                        experiment_dir=os.path.join(clap_out_dir),
                        sample_rate=sr,
                        device=device,
                        effects=effects,
                        iterations=n_clap_calls,
                        optimization_method=OptimizationMethod.GRADIENT_DESCENT,
                        loss=LossFunction.GUIDED_SEMANTIC_LOSS,
                        initial_params_tensor=llm_params_tensor,
                        initial_params_dict=llm_params_dict,
                    )
                    _process_and_save_audio_and_params(
                        audio_tensor=audio_t, params_tensor=clap_params_tensor, params_dict=clap_params_dict, fx_chain=fx_chain, sample_rate=cfg.SAMPLE_RATE, output_dir=Path(clap_out_dir), name=f"{name}_{dry_idx}"
                    )

            print(f"[exp5] {word}/{instrument}: {len(dry)} files saved")
