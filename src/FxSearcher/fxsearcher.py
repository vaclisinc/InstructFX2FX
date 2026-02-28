#!/usr/bin/env python3

import argparse, os, json, csv
import re

import librosa
os.environ["TOKENIZERS_PARALLELISM"] = 'false'
import multiprocessing as mp
import time
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, *args, **kwargs: x
from dataclasses import dataclass
from typing import List, Dict, Tuple

from skopt import gp_minimize
from skopt.space import Real, Categorical
from skopt.utils import use_named_args

import numpy as np
import soundfile as sf
import optuna

from pedalboard import (
    Pedalboard,
    HighpassFilter,
    LowpassFilter,
    LowShelfFilter,
    HighShelfFilter,
    PeakFilter,
    Reverb,
    Delay,
    Distortion,
    PitchShift,
    Bitcrush,
)

import matplotlib.pyplot as plt

import torch, torchaudio

from utilities.audio_processing import resample_audio
from utilities.fx_processing import fx_initial_params_to_tensor, ALL_PARAM_RANGES

# -------------------------------
# Utility
# -------------------------------
def load_audio_mono(path: str, target_sr: int = 48000, max_duration: float = 10.0):
    audio, sr = librosa.load(path, sr=None, mono=True)

    if sr != target_sr:
        audio = librosa.resample(y=audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    max_samples = int(max_duration * sr)
    if audio.shape[-1] > max_samples:
        audio = audio[:max_samples]

    audio = audio[np.newaxis, :]

    return audio.astype(np.float32), sr

def save_audio(path: str, audio: np.ndarray, sr: int):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    audio = np.squeeze(audio)

    if audio.ndim == 2 and audio.shape[0] < audio.shape[1]:
        audio = audio.T

    print(f"Saving audio to {path} with shape {audio.shape} and sample rate {sr}")

    sf.write(path, audio, sr, subtype="PCM_16")

class EarlyStopper:
    """
    Custom early stopping callback.
    Stops the optimization if the best score hasn't improved by at least `delta`
    in the last `n_best` iterations.
    """
    def __init__(self, delta=0.0, n_best=10):
        self.delta = delta
        self.n_best = n_best
        self.best_func = np.inf
        self.counter = 0

    def __call__(self, res):
        # res.fun is the best score found so far
        if res.fun < self.best_func - self.delta:
            self.best_func = res.fun
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.n_best:
            print(f"\nStopping early because score hasn't improved in {self.n_best} iterations.")
            # Returning True stops the optimization
            return True

def safe_folder_name(text):
    return re.sub(r'[^a-zA-Z0-9_\-]', '_', text.strip().replace(' ', '_'))

# -------------------------------
# CLAP scoring
# -------------------------------


def clap_score_batch(audio_list: list[np.ndarray], sr: int, prompt: str, clap_model) -> list[float]:
    """Score a batch of audio clips against a prompt using laion_clap.

    Args:
        audio_list: List of numpy audio arrays (mono or stereo)
        sr: Sample rate (should be 48000 for laion_clap)
        prompt: Text prompt
        clap_model: CLAPWrapper instance with get_audio_embedding and get_text_embedding methods
    """
    # Convert numpy arrays to torch tensors and stack
    audio_tensors = []
    for a in audio_list:
        a = np.asarray(a, dtype=np.float32)
        if a.ndim == 1:
            a = a[np.newaxis, :]
        elif a.ndim == 2 and a.shape[0] > 2:
            a = a.T
        audio_tensors.append(torch.from_numpy(a))

    # Batch audio: stack along batch dimension
    if audio_tensors:
        audio_batch = torch.stack(audio_tensors, dim=0)  # [B, C, T]
        audio_embeddings = clap_model.get_audio_embedding(audio_batch, enable_grad=False)
    else:
        return []

    # Get text embedding
    text_embedding = clap_model.get_text_embedding(prompt)  # [1, D]

    # Compute cosine similarity scores
    scores = (audio_embeddings * text_embedding).sum(dim=-1).cpu().numpy().tolist()
    return [scores] if not isinstance(scores, list) else scores

def clap_score_batch_guide(audio_list: list[np.ndarray], sr: int, prompts: str or list[str], clap_model) -> list[float]: # type: ignore
    """Score audio clips using positive and negative prompts (guided scoring).

    Args:
        audio_list: List of numpy audio arrays
        sr: Sample rate (should be 48000)
        prompts: Either a single prompt string or list of [positive_prompt, negative_prompt]
        clap_model: CLAPWrapper instance
    """
    print(f"      [clap_score_batch_guide] Input audio_list length: {len(audio_list)}")
    print(f"      [clap_score_batch_guide] Audio shapes: {[a.shape if isinstance(a, np.ndarray) else type(a) for a in audio_list]}")

    # Convert and prepare audio
    audio_tensors = []
    for a in audio_list:
        a = np.asarray(a, dtype=np.float32)
        if a.ndim == 1:
            a = a[np.newaxis, :]
        elif a.ndim == 2 and a.shape[0] > 2:
            a = a.T
        audio_tensors.append(torch.from_numpy(a))

    if not audio_tensors:
        return []

    audio_batch = torch.stack(audio_tensors, dim=0)  # [B, C, T]
    print(f"      [clap_score_batch_guide] Audio batch shape: {audio_batch.shape}")

    # Handle prompts: can be str, list of 1 (positive), or list of 2 (positive, negative)
    if isinstance(prompts, str):
        positive_prompt = prompts
        negative_prompt = "harsh, distorted, muddy, unclear, oversaturated, unpleasant sound"
    elif isinstance(prompts, list):
        positive_prompt = prompts[0]
        negative_prompt = prompts[1] if len(prompts) > 1 else "harsh, distorted, muddy, unclear, oversaturated, unpleasant sound"

    print(f"      [clap_score_batch_guide] Positive prompt: {positive_prompt}")
    print(f"      [clap_score_batch_guide] Negative prompt: {negative_prompt}")

    # Get embeddings
    print(f"      [clap_score_batch_guide] Computing audio embeddings...")
    audio_embeddings = clap_model.get_audio_embedding(audio_batch, enable_grad=False)
    print(f"      [clap_score_batch_guide] Audio embeddings shape: {audio_embeddings.shape}")


    print(f"      [clap_score_batch_guide] Computing text embeddings...")
    positive_embedding = clap_model.get_text_embedding(positive_prompt)  # [1, D]
    negative_embedding = clap_model.get_text_embedding(negative_prompt)  # [1, D]
    print(f"      [clap_score_batch_guide] Text embeddings shapes: pos={positive_embedding.shape}, neg={negative_embedding.shape}")

    # Compute guided scores: positive_score - negative_score
    print(f"      [clap_score_batch_guide] Computing guided scores...")
    positive_scores = (audio_embeddings * positive_embedding).sum(dim=-1)  # [B]
    negative_scores = (audio_embeddings * negative_embedding).sum(dim=-1)  # [B]
    guided_scores = (positive_scores - negative_scores).cpu().numpy().tolist()
    print(f"      [clap_score_batch_guide] Final guided scores: {guided_scores}")
    return [guided_scores] if not isinstance(guided_scores, list) else guided_scores

def clap_score(audio_mono: np.ndarray, sr: int, prompt: str, clap_model) -> float:
    """Score a single audio clip against a prompt using laion_clap."""
    print(f"      [clap_score] Input audio shape: {audio_mono.shape}, sr: {sr}")
    result = clap_score_batch([audio_mono], sr, prompt, clap_model)[0]
    print(f"      [clap_score] Result: {result}")
    return result

# -------------------------------
# Plugin renderers
# -------------------------------
def build_eq_chain(mode: str, low_cut=80.0, high_cut=14000.0, q=1.0, gains: Dict[str, float] = None, peak1_freq=200.0, peak2_freq=1000.0, peak3_freq=5000.0):
    default_gains = {'low_shelf': 3.0, 'high_shelf': -2.0, 'peak1': -1.5, 'peak2': 2.0, 'peak3': 1.0}
    if gains is None:
        gains = default_gains.copy()
    else:
        for k, v in default_gains.items():
            if k not in gains:
                gains[k] = v

    chain = []
    # 1. decide low_mode and high_mode by splitting the string 'pass-shelf' at '-'
    try:
        low_mode, high_mode = mode.split('-')
    except ValueError:
        # Set default values in case of unexpected errors
        low_mode, high_mode = "pass", "pass"

    # 2. append highpassfilter according to low_mode
    if low_mode == "pass":
        chain.append(HighpassFilter(cutoff_frequency_hz=low_cut))
    elif low_mode == "shelf":
        chain.append(LowShelfFilter(cutoff_frequency_hz=low_cut, gain_db=gains['low_shelf'], q=q))

    # 3. append lowpassfilter according to high_mode
    if high_mode == "pass":
        chain.append(LowpassFilter(cutoff_frequency_hz=high_cut))
    elif high_mode == "shelf":
        chain.append(HighShelfFilter(cutoff_frequency_hz=high_cut, gain_db=gains['high_shelf'], q=q))

    # 4. The three peak filters are kept as is
    chain.extend([
        PeakFilter(cutoff_frequency_hz=peak1_freq, gain_db=gains['peak1'], q=q),
        PeakFilter(cutoff_frequency_hz=peak2_freq, gain_db=gains['peak2'], q=q),
        PeakFilter(cutoff_frequency_hz=peak3_freq, gain_db=gains['peak3'], q=q)
    ])
    return chain

# variant builder given params
def render(audio, sr, config: Dict):
    order = [
            "EQ",
            "Distortion",
            "Bitcrush",
            "PitchShift",
            "Delay",
            "Reverb"
        ]
    # sort config keys by processing order
    sorted_types = sorted(config.keys(), key=lambda x: order.index(x) if x in order else 99)
    board = []
    for fx_type in sorted_types:
        fx = config[fx_type]
        if fx_type == "Distortion":
            board.append(Distortion(drive_db=fx["drive_db"]))
        elif fx_type == "EQ":
            print('appluing EQ with config:', fx)
            board.extend(build_eq_chain(
            fx["mode"], fx["low_cut"], fx["high_cut"], fx.get("q", 1.0), fx.get("gains"),
            fx.get("peak1_freq", 200.0), fx.get("peak2_freq", 1000.0), fx.get("peak3_freq", 5000.0)

        ))
        elif fx_type == "Reverb":
            board.append(Reverb(room_size=fx["room_size"], damping=fx.get("damping", 0.4), wet_level=fx.get("wet_level", 0.12)))
        elif fx_type == "Delay":
            board.append(Delay(delay_seconds=fx["delay"], feedback=0.25))
        elif fx_type == "PitchShift":
            board.append(PitchShift(semitones=fx["semitones"]))
        elif fx_type == "Bitcrush":
            board.append(Bitcrush(bit_depth=fx["bit_depth"]))
    pb = Pedalboard(board)
    # Pedalboard expects numpy arrays; convert if needed
    if isinstance(audio, torch.Tensor):
        audio_np = audio.squeeze(0).cpu().numpy()
    else:
        audio_np = audio
    return pb(audio_np, sr)


def define_search_space(config, PARAM_RANGES):
    search_space = []
    param_names = []

    for fx_type, fx_params in config.items():
        if fx_type != "EQ":
            search_space.append(Real(0.0, 1.0, name=f"{fx_type}__activation"))
            param_names.append(f"{fx_type}__activation")

        for p_name, p_info in PARAM_RANGES[fx_type].items():
            full_param_name = f"{fx_type}__{p_name}"
            param_names.append(full_param_name)
            if p_info.get('type') == 'categorical':
                search_space.append(Categorical(p_info['choices'], name=full_param_name))
            elif p_info.get('scale') == 'log':
                search_space.append(Real(p_info['lo'], p_info['hi'], prior='log-uniform', name=full_param_name))
            else:
                search_space.append(Real(p_info['lo'], p_info['hi'], name=full_param_name))

    return search_space, param_names

def refine_candidate_bayesian(args_dict, plot=False):
    """Refines a candidate using Bayesian Optimization."""
    print("\n🎯 === Starting Bayesian Optimization ===")

    initial_config = args_dict['initial_config']
    audio = args_dict['audio']
    sr = args_dict['sr']
    PARAM_RANGES = args_dict['PARAM_RANGES']
    prompt = args_dict['prompt']
    outdir = args_dict['outdir']
    top_n = args_dict['top_n']
    n_calls = args_dict['n_calls']
    use_guide = args_dict['use_guide']
    clap_model = args_dict['clap_model']

    print(f"  ├─ Audio shape: {audio.shape if isinstance(audio, tuple) else audio.shape}, sr: {sr}")
    print(f"  ├─ Prompt: {prompt}")
    print(f"  ├─ N calls: {n_calls}, Top N: {top_n}, Use guide: {use_guide}")
    print(f"  ├─ Effects in initial_config: {list(initial_config.keys())}")
    print(f"  └─ Effects in PARAM_RANGES: {list(PARAM_RANGES.keys())}")

    print(f"\n📦 CLAP model already loaded (laion_clap)")

    # 1. Define the search space
    print(f"\n🔧 Defining search space...")
    search_space, param_names = define_search_space(initial_config, PARAM_RANGES)
    print(f"  ├─ Total parameters: {len(param_names)}")
    print(f"  └─ Sample params: {param_names[:5]}...")

    # 2. Define the objective function
    #    This function processes the audio with parameters provided by skopt and returns the score.
    @use_named_args(search_space)
    def vanilla_objective_function(**params):
        temp_config = {k: {**v} for k, v in initial_config.items()}
        active_config = {}

        for fx_type, fx_params in temp_config.items():
            # always activate EQ
            is_active = (fx_type == "EQ") or (params.get(f"{fx_type}__activation", 0.0) > 0.5)

            if is_active:
                # Update parameters of activated effects
                for p_name, p_value in params.items():
                    p_fx_type, param_key = p_name.split('__')
                    if p_fx_type == fx_type and param_key != 'activation':
                        keys = param_key.split('.')
                        if len(keys) == 2:
                            if keys[0] not in fx_params: fx_params[keys[0]] = {}
                            fx_params[keys[0]][keys[1]] = p_value
                        else:
                            fx_params[param_key] = p_value
                active_config[fx_type] = fx_params

        print(f"    🔄 Rendering with active effects: {list(active_config.keys())}")
        print(f'      ├─ original audio shape: {audio.shape if isinstance(audio, tuple) else audio.shape}')
        processed_audio = render(audio, sr, active_config)
        print(f"      ├─ Rendered audio shape: {processed_audio.shape}")

        print(f"    🔄 Resampling to 48kHz...")
        resampled_processed_audio = resample_audio(processed_audio, sr)
        print(f"      └─ Resampled shape: {resampled_processed_audio.shape}")

        print(f"    🔄 Computing CLAP score...")
        score = clap_score(resampled_processed_audio, 48000, prompt, clap_model)
        print(f"      └─ Score: {score:.4f}")

        # Store audio embedding for later visualization
        a = np.asarray(resampled_processed_audio, dtype=np.float32)
        if a.ndim == 1:
            a = a[np.newaxis, :]
        audio_emb = clap_model.get_audio_embedding(torch.from_numpy(a).unsqueeze(0), enable_grad=False)
        embeddings_history.append({
            'iteration': len(embeddings_history),
            'audio_embedding': audio_emb.detach().cpu(),
            'score': score,
        })

        return -1.0 * score

    scores_history = {}
    embeddings_history = []  # stores {'iteration', 'audio_embedding', 'score'} per call

    @use_named_args(search_space)
    def objective_function_guide(**params):
        temp_config = {k: {**v} for k, v in initial_config.items()}
        active_config = {}

        for fx_type, fx_params in temp_config.items():
            # always activate EQ
            is_active = (fx_type == "EQ") or (params.get(f"{fx_type}__activation", 0.0) > 0.5)

            if is_active:
                # Update parameters of activated effects
                for p_name, p_value in params.items():
                    p_fx_type, param_key = p_name.split('__')
                    if p_fx_type == fx_type and param_key != 'activation':
                        keys = param_key.split('.')
                        if len(keys) == 2:
                            if keys[0] not in fx_params: fx_params[keys[0]] = {}
                            fx_params[keys[0]][keys[1]] = p_value
                        else:
                            fx_params[param_key] = p_value
                active_config[fx_type] = fx_params

        print(f"    🔄 Rendering with active effects: {list(active_config.keys())}")
        processed_audio = render(audio, sr, active_config)
        print(f"      ├─ Rendered audio shape: {processed_audio.shape}")

        print(f"    🔄 Resampling to 48kHz...")
        resampled_processed_audio = resample_audio(processed_audio, sr)
        print(f"      └─ Resampled shape: {resampled_processed_audio.shape}")

        # Use both positive and guide prompts to calculate scores
        positive_prompt = prompt # e.g., "A clear vocal with a subtle club room ambience"
        guide_prompt = "A harsh, distorted, muddy, unclear, oversaturated, unpleasant sound"

        # Calculate scores for both prompts in a single batch
        prompts = [positive_prompt, guide_prompt]
        audio_batch = [resampled_processed_audio, resampled_processed_audio]

        print(f"    🔄 Computing guided CLAP scores (positive + guide)...")
        scores = clap_score_batch_guide(audio_batch, 48000, prompts, clap_model)

        positive_score = scores[0]
        guide_score = scores[1]
        final_score = positive_score - guide_score

        print(f"      ├─ Positive score: {positive_score:.4f}")
        print(f"      ├─ Guide score: {guide_score:.4f}")
        print(f"      └─ Final score: {final_score:.4f}")

        params_tuple = tuple(sorted(params.items()))
        scores_history[params_tuple] = positive_score

        # Store audio embedding for later visualization
        a = np.asarray(resampled_processed_audio, dtype=np.float32)
        if a.ndim == 1:
            a = a[np.newaxis, :]
        audio_emb = clap_model.get_audio_embedding(torch.from_numpy(a).unsqueeze(0), enable_grad=False)
        embeddings_history.append({
            'iteration': len(embeddings_history),
            'audio_embedding': audio_emb.detach().cpu(),
            'score': float(positive_score),
            'guided_score': float(final_score),
        })

        # skopt minimizes the objective function, so we return -1 times the final score
        return -1.0 * final_score

    pbar = tqdm(total=n_calls, desc="Bayesian Optimization Progress", unit="iteration")
    def pbar_callback(res):
        pbar.update(1)
    early_stopper = EarlyStopper(delta=0.001, n_best=30)

    # 3. Run bayesian optimization
    objective_function = objective_function_guide if use_guide else vanilla_objective_function
    print(f"\n⚙️  Running GP minimize with {n_calls} calls...")
    print(f"  ├─ Objective: {'guided CLAP' if use_guide else 'vanilla CLAP'}")
    print(f"  └─ Search space dimensionality: {len(search_space)}")

    result = gp_minimize(objective_function,
                         search_space,
                         n_calls=n_calls,
                         acq_func="LCB",
                         kappa=5,
                         n_initial_points=20,
                         random_state=42,
                         callback=[pbar_callback, early_stopper]
                        )
    pbar.close()

    print(f"\n✓ GP minimize completed!")
    print(f"  ├─ Best score: {result.fun:.4f}")
    print(f"  ├─ Iterations completed: {len(result.func_vals)}")
    print(f"  └─ Extracting top {top_n} results...")

    def plot_optimization_trajectories(result, param_names, search_space, outdir):
        # result.x_iters is a list of parameter lists per iteration
        x_iters = result.x_iters
        if not x_iters:
            return

        n_iters = len(x_iters)
        n_params = len(param_names)

        # Transpose to shape (n_params, n_iters)
        param_iters = [[] for _ in range(n_params)]
        for it in x_iters:
            for i, val in enumerate(it):
                param_iters[i].append(val)

        for i, name in enumerate(param_names):
            values = param_iters[i]

            plt.figure(figsize=(10, 3))
            # Handle categorical parameters
            space_obj = search_space[i]
            if isinstance(space_obj, Categorical):
                categories = list(space_obj.categories)
                # map category values to indices for plotting
                numeric = [categories.index(v) if v in categories else None for v in values]
                plt.plot(numeric, marker='o')
                plt.yticks(range(len(categories)), categories)
                plt.ylabel('Category')
            else:
                # Numeric (Real)
                numeric = [float(v) if v is not None else np.nan for v in values]
                plt.plot(numeric, marker='o')
                plt.ylabel('Value')

            plt.title(f"Parameter: {name}")
            plt.xlabel('Iteration')
            plt.grid(alpha=0.3)
            fname = safe_folder_name(name)
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, f"param_trajectory_{fname}.png"))
            plt.close()

    try:
        if plot:
            plot_optimization_trajectories(result, param_names, search_space, outdir)
            print(f"Optimization trajectories plotted and saved to {outdir}")
    except Exception as e:
        print(f"Warning: failed to plot optimization trajectories: {e}")

    # 4. Extract Top-N results from the search history
    print(f"\n📊 Extracting top {top_n} results from {len(result.func_vals)} total evaluations...")
    all_results = sorted(zip(result.func_vals, result.x_iters), key=lambda x: x[0])
    top_n_results = all_results[:top_n]
    final_presets = []
    presets_to_sort = []

    for rank, (score, params_list) in enumerate(top_n_results):
        print(f"  ├─ Processing rank {rank + 1}: score={score:.4f}")
        best_params = dict(zip(param_names, params_list))
        final_config = {}
        params_tuple = tuple(sorted(best_params.items()))
        benchmark_clap_score = scores_history.get(params_tuple, 0.0) if use_guide else -1.0 * score

        for fx_type, fx_params in initial_config.items():
            is_active = (fx_type == "EQ") or (best_params.get(f"{fx_type}__activation", 0.0) > 0.5)

            if is_active:
                new_fx = {**fx_params}
                for p_name, p_value in best_params.items():
                    p_fx_type, p_key = p_name.split('__')
                    if p_fx_type == fx_type and p_key != 'activation':
                        keys = p_key.split('.')
                        if len(keys) == 2:
                            if keys[0] not in new_fx: new_fx[keys[0]] = {}
                            new_fx[keys[0]][keys[1]] = p_value
                        else:
                            new_fx[p_key] = p_value
                final_config[fx_type] = new_fx

        presets_to_sort.append({
            "composite_score": -1.0 * score,
            "benchmark_clap_score": benchmark_clap_score,
            "plugins": final_config
        })

        # 5. Re-sort presets by benchmark CLAP score and save audio files
        final_presets = sorted(presets_to_sort, key=lambda x: x['benchmark_clap_score'], reverse=True)

        print(f"  └─ Rendering final audio for rank {rank + 1}...")
        for rank, preset in enumerate(final_presets):
            # Render audio only once before saving
            final_audio = render(audio, sr, preset['plugins'])

            # Determine file name based on sorted rank
            if rank == 0:
                filename = "best.wav"
            else:
                filename = f"rank_{rank + 1}.wav"

            save_audio(os.path.join(outdir, filename), final_audio, sr)

            # Add final rank to the dictionary
            preset['rank'] = rank + 1

    print(f"\n✓ === Bayesian Optimization Complete ===")
    print(f"  ├─ Output directory: {outdir}")
    print(f"  ├─ Presets saved: {len(final_presets)}")
    print(f"  └─ Embeddings stored: {len(embeddings_history)}")

    return final_presets, embeddings_history

ALL_PARAM_RANGES_FXSearcher = {
    "Distortion": {"drive_db": {"lo": 0, "hi": 15, "res": 0.1, "scale": "linear"}},
    "EQ": {
        "mode": {"choices":
                 ["pass-pass", "pass-shelf", "shelf-pass", "shelf-shelf"], "type": "categorical"},
        "low_cut": {"lo": 50, "hi": 500, "res": 10, "scale": "log"},
        "high_cut": {"lo": 8000, "hi": 16000, "res": 100, "scale": "log"},
        "q": {"lo": 0.1, "hi": 10.0, "res": 0.1, "scale": "linear"},
        "gains.low_shelf": {"lo": -20.0, "hi": 20.0, "res": 0.2, "scale": "linear"},
        "gains.high_shelf": {"lo": -20.0, "hi": 20.0, "res": 0.2, "scale": "linear"},
        "gains.peak1": {"lo": -20.0, "hi": 20.0, "res": 0.2, "scale": "linear"},
        "gains.peak2": {"lo": -20.0, "hi": 20.0, "res": 0.2, "scale": "linear"},
        "gains.peak3": {"lo": -20.0, "hi": 20.0, "res": 0.2, "scale": "linear"},
        "peak1_freq": {"lo": 100.0, "hi": 500.0, "res": 10.0, "scale": "log"},
        "peak2_freq": {"lo": 500.0, "hi": 4000.0, "res": 100.0, "scale": "log"},
        "peak3_freq": {"lo": 4000.0, "hi": 12000.0, "res": 1000.0, "scale": "log"}
    },
    "Reverb": {
        "room_size": {"lo": 0.0, "hi": 1.0, "res": 0.05, "scale": "linear"},
        "damping": {"lo": 0.0, "hi": 1.0, "res": 0.05, "scale": "linear"},
        "wet_level": {"lo": 0.00, "hi": 1.0, "res": 0.01, "scale": "linear"},
    },
    "Delay": {"delay": {"lo": 0.0, "hi": 0.05, "res": 0.01, "scale": "linear"}},
    "PitchShift": {"semitones": {"lo": -12, "hi": 12, "res": 1, "scale": "linear"}},
    "Bitcrush": {"bit_depth": {"lo": 0, "hi": 16, "res": 1, "scale": "linear"}},
}



# ==============================================================================
# Main Logic
# ==============================================================================
def fxsearcher(audio: str = None,
               prompt: str = None,
               outdir: str = None,
               clap_model = None,
               top_n: int = 5,
               n_calls: int = 100,
               use_guide: bool = False,
               fxs: list = None,
               initial_params: dict = None,
               plot: bool = False,
               all_param_ranges: dict = None):
    """Run the FX searcher.

    Parameters can be provided directly (programmatic use) or via CLI (when
    `audio`, `prompt` or `outdir` are not provided).
    """
    from datetime import datetime
    print(' Starting FX Searcher with the following parameters:')

    print('initial params', initial_params)

    if all_param_ranges is not None:
        ALL_PARAM_RANGES = all_param_ranges
    else:
        ALL_PARAM_RANGES = ALL_PARAM_RANGES_FXSearcher
        print('Using default parameter ranges. To provide custom ranges, pass a dictionary to `all_param_ranges` argument.')

    # If essential arguments are missing, fall back to CLI parsing for backwards compatibility
    if audio is None or prompt is None or outdir is None:
        ap = argparse.ArgumentParser()
        ap.add_argument("--audio", required=True)
        ap.add_argument("--prompt", required=True)
        ap.add_argument("--outdir", required=True)
        ap.add_argument("--model", default="laion/clap-htsat-unfused")
        ap.add_argument("--top_n", type=int, default=5, help="Number of top candidates to refine in parallel.")
        ap.add_argument("--n_calls", type=int, default=100, help="Number of calls for Bayesian optimization.")
        ap.add_argument("--use_guide", action="store_true", help="Whether to use guiding prompt.")
        ap.add_argument("--plot", action="store_true", help="Whether to plot the results.")
        args = ap.parse_args()

        audio_input = args.audio
        prompt = args.prompt
        outdir_base = args.outdir
        model = args.model
        top_n = args.top_n
        n_calls = args.n_calls
        use_guide = args.use_guide
    else:
        # When provided programmatically, treat `outdir` as the base directory
        audio_input = audio
        outdir_base = outdir

    prompt_folder = safe_folder_name(prompt)
    run_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(outdir_base, f"{prompt_folder}_{run_time}")
    os.makedirs(outdir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load CLAP model if not passed (CLI case)
    if clap_model is None:
        print(f"📦 Loading laion_clap model...")
        from embeddings.clap import CLAPWrapper
        clap_model = CLAPWrapper(device=device)
        print(f"  └─ ✓ laion_clap model loaded")

    # Load audio: `audio_input` is expected to be a file path when called from CLI
    if isinstance(audio_input, str):
        audio, sr = load_audio_mono(audio_input)
    else:
        # allow passing (audio_array, sr) tuple programmatically
        if isinstance(audio_input, (list, tuple)) and len(audio_input) == 2:
            audio, sr = audio_input
        else:
            raise ValueError("`audio` must be a file path or a (audio_array, sr) tuple when calling fxsearcher programmatically")

    # save original audio
    save_audio(os.path.join(outdir, "original.wav"), audio, sr)

    start_time = time.time()

    if initial_params is None:
        initial_params = {
            "EQ": {"mode": "shelf", "low_cut": 120.0, "high_cut": 12000.0, "q": 1.0, "gains": {},
                    "peak1_freq": 200.0, "peak2_freq": 1000.0, "peak3_freq": 5000.0},
            "Distortion": {"drive_db": 1.0},
            "Reverb": {"room_size": 0.3, "damping": 0.5, "wet_level": 0.1},
            "Delay": {"delay": 0.1},
            "PitchShift": {"semitones": 0},
            "Bitcrush": {"bit_depth": 0},
        }
    if fxs is not None:
        initial_params = {k: v for k, v in initial_params.items() if k in fxs}
        PARAM_RANGES = {fx: ALL_PARAM_RANGES[fx] for fx in ALL_PARAM_RANGES if fx in fxs}

    else:
        PARAM_RANGES = ALL_PARAM_RANGES

    print(f"Initial parameters: {json.dumps(initial_params, indent=2)}")

    print(f"\n--- Starting Bayesian Optimization for {', '.join(PARAM_RANGES.keys())} ---")

    args_dict = {
        'initial_config': initial_params,
        'audio': audio, 'sr': sr, 'PARAM_RANGES': PARAM_RANGES,
        'prompt': prompt, 'outdir': outdir,
        'top_n': top_n, 'n_calls': n_calls, 'use_guide': use_guide,
        'clap_model': clap_model
    }

    final_result, embeddings_history = refine_candidate_bayesian(args_dict, plot=plot)

    elapsed = time.time() - start_time
    print(f"\nRefinement finished. Total search time: {elapsed:.2f} seconds")

    final_output = {
        "prompt": prompt,
        "search_time_seconds": elapsed,
        "results": final_result
    }
    with open(os.path.join(outdir, "best_presets.json"), "w") as f:
        json.dump(final_output, f, indent=2)

    print(f"Top {len(final_result)} configs saved to best_presets.json at {outdir}.")

    print("\nFX Searcher completed successfully.")

    print('Final result:', final_result)

    # FXSearcher returns Pedalboard params (incompatible with DASP chain)
    # The audio has already been rendered and saved internally
    # Return embeddings_history as second value for downstream visualization
    audio, sr = torchaudio.load(os.path.join(outdir, "best.wav"))
    return audio, embeddings_history, None


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    fxsearcher()