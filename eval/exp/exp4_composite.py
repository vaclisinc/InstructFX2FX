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

_ROOT = Path(__file__).resolve().parent.parent.parent   # project root
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

import eval.config as cfg

import torch
import soundfile as sf
import librosa
import json
from src.configurations.config import Config, LossFunction, ParameterInitializationMethod, OptimizationMethod
from src.training.parameterengine import ParameterEngine
from src.effects.fx import FXChainFactory
from src.prompts.prompt import Prompt, PromptFactory
from src.metrics.run_metrics import run_metrics_against_gt
from eval.data.socialfx_loader import load_all_params_for_word
from eval.helpers import _dry_paths
import eval.config as cfg
from src.utilities.text_processing import to_serializable


# ── Output directories (under eval/) ─────────────────────────────────────────

_EXP4_SYSTEM_RESULTS_DIR = os.path.join(cfg.SYSTEM_RESULTS_DIR, "exp4")
_EXP4_RESULTS_DIR = os.path.join(cfg.RESULTS_DIR, "exp4")

_EXP4_CLAP_DIR = os.path.join(_EXP4_SYSTEM_RESULTS_DIR, "clap")


def run_exp4(
    llm_client,
    clap,
    fx_type: str = "eq",
    device: str = cfg.DEVICE,
    n_clap_calls: int = 100,
    effects: List[str] = None,
    words: Optional[List[str]] = None,
) -> Dict:
    """
    Experiment 4: Composite EQ prompts, CLAP vs LLM, multiple loss strategies.
    """

    combinations = [
        # ("Warm", "Old"),
        # ("Tin", "Soft"),
        # ("Loud", "Sad"),
        # ("Tin", "Old"),
        # ("Warm", "Sad"),
        # ("Tin", "Low"),
        # ("Soft", "Old"),
        ("loud", "warm"),
    ]
    methods = ["clap_concat", "clap_avg", "clap_sequential", "llm"]
    results = {}
    parameter_engine = ParameterEngine()
    fx_chain = FXChainFactory.create_fx_chain_from_effects(cfg.FX_EFFECTS, sample_rate=cfg.SAMPLE_RATE, device=device)
    instruments = cfg.INSTRUMENTS

    for wordA, wordB in combinations:
        pair_key = f"{wordA.lower()}_{wordB.lower()}"
        results[pair_key] = {}
        for instrument in instruments:
            dry_paths = _dry_paths(instrument)
            if not dry_paths:
                continue
            for dry_idx, dry_path in enumerate(dry_paths):
                audio_np, sr = sf.read(dry_path)
                if audio_np.ndim > 1:
                    audio_np = audio_np.mean(axis=1)
                if sr != cfg.SAMPLE_RATE:
                    audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=cfg.SAMPLE_RATE)
                audio_t = torch.from_numpy(audio_np).float().to(device)
                if audio_t.ndim == 1:
                    audio_t = audio_t.unsqueeze(0).unsqueeze(0)  # [1, 1, T]
                elif audio_t.ndim == 2:
                    audio_t = audio_t.unsqueeze(0)  # [1, C, T]

                # --- CLAP: (a) Concatenation ---
                concat_prompt = f"This sound is {wordA.lower()} and {wordB.lower()}."
                clap_config_concat = Config(
                    initialization_method=ParameterInitializationMethod.RANDOM,
                    loss_function=LossFunction.SEMANTIC_SIMILARITY_LOSS,
                    llmclient=llm_client,
                    fx_chain=fx_chain,
                    embedding=clap,
                    device=device,
                    num_iterations=n_clap_calls,
                    text_target=concat_prompt,
                    optimization_method=OptimizationMethod.GRADIENT_DESCENT,
                )
                _, _, result_audio_concat, _, _ = parameter_engine.get_params(audio_t, clap_config_concat)
                if result_audio_concat is not None:
                    result_np_concat = result_audio_concat.squeeze().detach().cpu().numpy()
                    results[pair_key].setdefault("clap_concat", []).append((instrument, result_np_concat))

                # --- CLAP: (b) Averaging ---
                embA = clap.get_text_embedding(f"This sound is {wordA.lower()}.")
                embB = clap.get_text_embedding(f"This sound is {wordB.lower()}.")
                avg_emb = (embA + embB) / 2
                clap_config_avg = Config(
                    initialization_method=ParameterInitializationMethod.RANDOM,
                    loss_function=LossFunction.SEMANTIC_SIMILARITY_LOSS,
                    llmclient=llm_client,
                    fx_chain=fx_chain,
                    embedding=clap,
                    device=device,
                    num_iterations=n_clap_calls,
                    text_target=None,  # Not used when passing target_embedding
                    optimization_method=OptimizationMethod.GRADIENT_DESCENT,
                )
                _, _, result_audio_avg, _, _ = parameter_engine.get_params(
                    audio_t, clap_config_avg, target_embedding=avg_emb
                )
                if result_audio_avg is not None:
                    result_np_avg = result_audio_avg.squeeze().detach().cpu().numpy()
                    results[pair_key].setdefault("clap_avg", []).append((instrument, result_np_avg))

                # --- CLAP: (c) Sequential (Directional) ---
                clap_config_seq = Config(
                    initialization_method=ParameterInitializationMethod.LLM,
                    prompt=PromptFactory.LLM_PARAMETER_INITIALIZATION_PROMPT_DASP(
                        fx_chain=fx_chain, instruction=f"Make this sound {wordA.lower()}.", effects=effects
                    ),
                    loss_function=LossFunction.GUIDED_SEMANTIC_LOSS,
                    llmclient=llm_client,
                    fx_chain=fx_chain,
                    embedding=clap,
                    device=device,
                    num_iterations=n_clap_calls,
                    text_target=f"This sound is {wordB.lower()}.",
                    optimization_method=OptimizationMethod.GRADIENT_DESCENT,
                )
                _, _, result_audio_seq, _, _ = parameter_engine.get_params(audio_t, clap_config_seq)
                if result_audio_seq is not None:
                    result_np_seq = result_audio_seq.squeeze().detach().cpu().numpy()
                    results[pair_key].setdefault("clap_sequential", []).append((instrument, result_np_seq))

                # --- LLM ---

                llm_config = Config(
                    prompt=PromptFactory.LLM_PARAMETER_INITIALIZATION_PROMPT_DASP(
                        fx_chain=fx_chain, instruction=f"Make this sound {wordA.lower()} and {wordB.lower()}.", effects=effects
                    ),
                    initialization_method=ParameterInitializationMethod.LLM,
                    loss_function=None,
                    llmclient=llm_client,
                    fx_chain=fx_chain,
                    embedding=clap,
                    device=device,
                )
                params_tensor, params_dict, _, _, _ = parameter_engine.get_params(audio_t, llm_config)
                output_audio = fx_chain(audio_t, params_tensor)
                output_np = output_audio.squeeze().detach().cpu().numpy()
                results[pair_key].setdefault("llm", []).append((instrument, output_np))

        # --- Metrics: compare each method's outputs to both GTs ---
        # Load GT params for wordA and wordB
        gt_params_A = load_all_params_for_word(wordA.lower())
        gt_params_B = load_all_params_for_word(wordB.lower())
        # Render GT audio for both words if not already cached
        from eval.gt_bank import build_word_gt_bank_per_instance
        for instrument in instruments:
            dry_paths = _dry_paths(instrument)
            build_word_gt_bank_per_instance(wordA.lower(), instrument, dry_paths, gt_params_A, cfg.GT_BANK_DIR_ONESHOT, sr=cfg.SAMPLE_RATE)
            build_word_gt_bank_per_instance(wordB.lower(), instrument, dry_paths, gt_params_B, cfg.GT_BANK_DIR_ONESHOT, sr=cfg.SAMPLE_RATE)
        gt_folder = cfg.GT_BANK_DIR_ONESHOT
        # For each method, save synthesized audio to temp folder structure for metrics
        scores = {}
        for method in methods:
            # Save all outputs for this method, preserving instrument subfolders
            outputs = results[pair_key].get(method, [])
            for idx, (instrument, audio_np) in enumerate(outputs):
                method_dir = os.path.join(_EXP4_SYSTEM_RESULTS_DIR, method, pair_key, instrument)
                os.makedirs(method_dir, exist_ok=True)
                out_path = os.path.join(method_dir, f"{instrument}_{idx}.wav")
                sf.write(out_path, audio_np, cfg.SAMPLE_RATE)
                print(f'[exp4] saved {method} output for {pair_key}/{instrument} to {out_path}')
            # Compute metrics against both GTs
            test_folders = {f"{method}": os.path.join(_EXP4_SYSTEM_RESULTS_DIR, method)}
            # Metrics vs GT(wordA)
            metrics_A = run_metrics_against_gt(gt_folder, test_folders, sr=cfg.SAMPLE_RATE, word_position_in_test_folder_name=0, search_patterns=None, save_report=False, verbose=True)
            # Metrics vs GT(wordB)
            metrics_B = run_metrics_against_gt(gt_folder, test_folders, sr=cfg.SAMPLE_RATE, word_position_in_test_folder_name=1, search_patterns=None, save_report=False, verbose=True)

            scores.setdefault(pair_key, {}).update({f"{method}_metrics_wordA": metrics_A})
            scores.setdefault(pair_key, {}).update({f"{method}_metrics_wordB": metrics_B})


    # Convert all numpy arrays in results to lists for JSON serialization


    serializable_results = to_serializable(scores)
    out_json = os.path.join(_EXP4_RESULTS_DIR, "exp4_composite_results.json")
    with open(out_json, "w") as f:
        json.dump(serializable_results, f, indent=2)
    return scores