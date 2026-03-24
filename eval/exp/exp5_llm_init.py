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
from src.training.parameterengine import ParameterEngine

_ROOT = Path(__file__).resolve().parent.parent.parent   # project root
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

import eval.config as cfg
from eval.data.socialfx_loader import load_all_params_for_word
from eval.exp.exp1_sequential import _dry_paths
from eval.gt_bank import build_word_gt_bank

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


# ── Labels ─────────────────────────────────────────────────────────────

SOURCE_GT = "SocialFX GT"
SOURCE_LLM = "LLM Init"
SOURCE_CLAP = "CLAP"

# ── Output directories (under eval/) ─────────────────────────────────────────

_EXP5_SYSTEM_RESULTS_DIR = os.path.join(cfg.SYSTEM_RESULTS_DIR, "exp5")
_EXP5_RESULTS_DIR = os.path.join(cfg.RESULTS_DIR, "exp5")

_EXP5_CLAP_DIR = os.path.join(_EXP5_SYSTEM_RESULTS_DIR, "llm_clap")
_EXP5_LLM_DIR = os.path.join(_EXP5_SYSTEM_RESULTS_DIR, "llm")


def run_exp5(
    llm_client,
    words: List[str],
    clap,
    fx_type: str = "eq",
    device: str = cfg.DEVICE,
    n_clap_calls: int = 100,
    effects: List[str] = None,
) -> Dict:

    unique_words = sorted(words)
    parameter_engine = ParameterEngine()

    fx_chain = FXChainFactory.create_fx_chain_from_effects(
        effects, sample_rate=cfg.SAMPLE_RATE, device=device,
    )

    for word in unique_words:
        for instrument in cfg.INSTRUMENTS:
            dry = _dry_paths(instrument)
            if not dry:
                continue
            out_dir = os.path.join(_EXP5_LLM_DIR, word, instrument)
            # Idempotent: skip if WAVs already exist
            if os.path.isdir(out_dir) and any(f.endswith(".wav") for f in os.listdir(out_dir)):
                print(f"[exp5] SKIP LLM {word}/{instrument}: exists")
                continue
            os.makedirs(out_dir, exist_ok=True)
            print(f"[exp5] RUN  LLM {word}/{instrument}")

            for dry_idx, dry_path in enumerate(dry):
                audio_np, sr = sf.read(dry_path)
                if audio_np.ndim > 1:
                    audio_np = audio_np.mean(axis=1)
                if sr != cfg.SAMPLE_RATE:
                    audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=cfg.SAMPLE_RATE)
                audio_t = torch.from_numpy(audio_np).float().to(device)
                audio_t = _ensure_bct(audio_t)

                # Single LLM call: init only, no loss function → returns immediately

                llm_config = Config(
                    prompt=PromptFactory.LLM_PARAMETER_INITIALIZATION_PROMPT_DASP(
                        fx_chain=fx_chain,
                        instruction=f"Make this {instrument} sound {word}.",
                        effects=effects,
                    ),
                    initialization_method=ParameterInitializationMethod.LLM,
                    loss_function=None,
                    llmclient=llm_client,
                    fx_chain=fx_chain,
                    embedding=clap,
                    device=device,
                )
                params_tensor, params_dict, _, _, _ = parameter_engine.get_params(audio_t, llm_config)

                name = instrument.replace(" ", "_")

                # Render audio through FX chain and save
                output_audio = fx_chain(audio_t, params_tensor) # No SIGMOID
                output_np = output_audio.squeeze().detach().cpu().numpy()
                wav_path = os.path.join(out_dir, f"{name}_llm.wav")
                sf.write(wav_path, output_np, cfg.SAMPLE_RATE)
                with open(os.path.join(out_dir, f"{name}_llm_params.json"), "w") as f:
                    json.dump(params_dict, f, indent=2)

            print(f"[exp5] LLM {word}/{instrument}: {len(dry)} files saved")


    for word in unique_words:
        for instrument in cfg.INSTRUMENTS:
            dry = _dry_paths(instrument)
            if not dry:
                continue
            out_dir = os.path.join(_EXP5_LLM_DIR, word, instrument)
            # Idempotent: skip if WAVs already exist
            if os.path.isdir(out_dir) and any(f.endswith(".wav") for f in os.listdir(out_dir)):
                print(f"[exp5] SKIP LLM {word}/{instrument}: exists")
                continue
            os.makedirs(out_dir, exist_ok=True)
            print(f"[exp5] RUN  LLM {word}/{instrument}")

            for dry_idx, dry_path in enumerate(dry):
                audio_np, sr = sf.read(dry_path)
                if audio_np.ndim > 1:
                    audio_np = audio_np.mean(axis=1)
                if sr != cfg.SAMPLE_RATE:
                    audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=cfg.SAMPLE_RATE)
                audio_t = torch.from_numpy(audio_np).float().to(device)
                audio_t = _ensure_bct(audio_t)

                # Single LLM call: init only, no loss function → returns immediately


                # instructionset =
                llm_config = Config(
                    prompt=PromptFactory.LLM_PARAMETER_INITIALIZATION_PROMPT_DASP(
                        fx_chain=fx_chain,
                        instruction=f"Make this {instrument} sound {word}.",
                        effects=effects,
                    ),
                    initialization_method=ParameterInitializationMethod.LLM,
                    loss_function=LossFunction.GUIDED_SEMANTIC_LOSS,
                    text_target=f"This sound is {word}",
                    llmclient=llm_client,
                    fx_chain=fx_chain,
                    embedding=clap,
                    device=device,
                )
                params_tensor, params_dict, _, _, _ = parameter_engine.get_params(audio_t, llm_config)

                name = instrument.replace(" ", "_")

                # Render audio through FX chain and save
                output_audio = fx_chain(audio_t, params_tensor) # No SIGMOID
                output_np = output_audio.squeeze().detach().cpu().numpy()
                wav_path = os.path.join(out_dir, f"{name}_llm.wav")
                sf.write(wav_path, output_np, cfg.SAMPLE_RATE)
                with open(os.path.join(out_dir, f"{name}_llm_params.json"), "w") as f:
                    json.dump(params_dict, f, indent=2)

            print(f"[exp5] LLM {word}/{instrument}: {len(dry)} files saved")

        # compare results
