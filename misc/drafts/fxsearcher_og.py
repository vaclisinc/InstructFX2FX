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

import torch
from transformers import ClapProcessor, ClapModel

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
    sf.write(path, audio.T, sr, subtype="PCM_16")

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
def get_clap_model(model_name: str, device: str):
    model = ClapModel.from_pretrained(model_name).to(device)
    processor = ClapProcessor.from_pretrained(model_name)
    return model, processor

def clap_score_batch(audio_list: list[np.ndarray], sr: int, prompt: str, model, processor, device: str) -> list[float]:
    wavs = [a.squeeze().astype(np.float32) for a in audio_list]
    # NOTE: The processor is smart enough to handle a single prompt for a batch of audio.
    # We pass the single prompt string directly.
    inputs = processor(text=prompt, audio=wavs, sampling_rate=sr, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        # text_features = model.get_text_features(inputs['input_ids'], attention_mask=inputs['attention_mask'])
        # audio_features = model.get_audio_features(input_features=inputs['input_features'])

        # text_features = torch.nn.functional.normalize(text_features, p=2, dim=-1)
        # audio_features = torch.nn.functional.normalize(audio_features, p=2, dim=-1)

        outputs = model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        input_features=inputs["input_features"],
        return_dict=True,
    )
        text_features = outputs.text_embeds
        audio_features = outputs.audio_embeds

        if hasattr(text_features, "pooler_output"):
            text_features = text_features.pooler_output
        if hasattr(audio_features, "pooler_output"):
            audio_features = audio_features.pooler_output

        text_features = torch.nn.functional.normalize(text_features, dim=-1)
        audio_features = torch.nn.functional.normalize(audio_features, dim=-1)


        scores_tensor = audio_features @ text_features.T

        scores = scores_tensor.squeeze().cpu().numpy().tolist()

    return [scores] if not isinstance(scores, list) else scores

def clap_score_batch_guide(audio_list: list[np.ndarray], sr: int, prompts: str or list[str], model, processor, device: str) -> list[float]: # type: ignore
    """
    Calculates CLAP scores. Handles both a single prompt for all audio clips
    and a list of prompts corresponding to each audio clip.
    """
    wavs = [a.squeeze().astype(np.float32) for a in audio_list]

    if isinstance(prompts, str):
        prompts = [prompts] * len(wavs)

    inputs = processor(text=prompts, audio=wavs, sampling_rate=sr, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        input_features=inputs["input_features"],
        return_dict=True,
    )
        text_features = outputs.text_embeds
        audio_features = outputs.audio_embeds

        if hasattr(text_features, "pooler_output"):
            text_features = text_features.pooler_output
        if hasattr(audio_features, "pooler_output"):
            audio_features = audio_features.pooler_output

        text_features = torch.nn.functional.normalize(text_features, dim=-1)
        audio_features = torch.nn.functional.normalize(audio_features, dim=-1)

        scores_tensor = (audio_features * text_features).sum(dim=1)

        scores = scores_tensor.cpu().numpy().tolist()

    return [scores] if not isinstance(scores, list) else scores

def clap_score(audio_mono: np.ndarray, sr: int, prompt: str, model, processor, device: str) -> float:
    return clap_score_batch([audio_mono], sr, prompt, model, processor, device)[0]

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
        # sort config
    config_sorted = sorted(config, key=lambda x: order.index(x["type"]) if x["type"] in order else 99)
    board = []
    for fx in config_sorted:
        fx_type = fx["type"]
        if fx_type == "Distortion":
            board.append(Distortion(drive_db=fx["drive_db"]))
        elif fx_type == "EQ":
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
    return pb(audio, sr)


def refine_candidate_bayesian(args_dict, plot=False):
    """Refines a candidate using Bayesian Optimization."""
    initial_config = args_dict['initial_config']
    audio = args_dict['audio']
    sr = args_dict['sr']
    PARAM_RANGES = args_dict['PARAM_RANGES']
    model_name = args_dict['model_name']
    prompt = args_dict['prompt']
    outdir = args_dict['outdir']
    top_n = args_dict['top_n']
    n_calls = args_dict['n_calls']
    use_guide = args_dict['use_guide']

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, processor = get_clap_model(model_name, device)

    # 1. Define the search space
    search_space = []
    param_names = []

    for fx in initial_config:
        fx_type = fx["type"]
        if fx_type not in PARAM_RANGES: continue

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
    # 2. Define the objective function
    #    This function processes the audio with parameters provided by skopt and returns the score.
    @use_named_args(search_space)
    def vanilla_objective_function(**params):
        temp_config = [p.copy() for p in initial_config]
        active_config = []

        for fx in temp_config:
            fx_type = fx['type']

            # always activate EQ
            is_active = (fx_type == "EQ") or (params.get(f"{fx_type}__activation", 0.0) > 0.5)

            if is_active:
                # Update parameters of activated effects
                for p_name, p_value in params.items():
                    p_fx_type, param_key = p_name.split('__')
                    if p_fx_type == fx_type and param_key != 'activation':
                        keys = param_key.split('.')
                        if len(keys) == 2:
                            if keys[0] not in fx: fx[keys[0]] = {}
                            fx[keys[0]][keys[1]] = p_value
                        else:
                            fx[param_key] = p_value
                active_config.append(fx)

        processed_audio = render(audio, sr, active_config)
        score = clap_score(processed_audio, sr, prompt, model, processor, device)
        return -1.0 * score

    scores_history = {}

    @use_named_args(search_space)
    def objective_function_guide(**params):
        temp_config = [p.copy() for p in initial_config]
        active_config = []

        for fx in temp_config:
            fx_type = fx['type']

            # always activate EQ
            is_active = (fx_type == "EQ") or (params.get(f"{fx_type}__activation", 0.0) > 0.5)

            if is_active:
                # Update parameters of activated effects
                for p_name, p_value in params.items():
                    p_fx_type, param_key = p_name.split('__')
                    if p_fx_type == fx_type and param_key != 'activation':
                        keys = param_key.split('.')
                        if len(keys) == 2:
                            if keys[0] not in fx: fx[keys[0]] = {}
                            fx[keys[0]][keys[1]] = p_value
                        else:
                            fx[param_key] = p_value
                active_config.append(fx)

        processed_audio = render(audio, sr, active_config)

        # Use both positive and guide prompts to calculate scores
        positive_prompt = prompt # e.g., "A clear vocal with a subtle club room ambience"
        guide_prompt = "A harsh, distorted, muddy, unclear, oversaturated, unpleasant sound"

        # Calculate scores for both prompts in a single batch
        prompts = [positive_prompt, guide_prompt]
        audio_batch = [processed_audio, processed_audio]

        scores = clap_score_batch_guide(audio_batch, sr, prompts, model, processor, device)

        positive_score = scores[0]
        guide_score = scores[1]

        final_score = positive_score - guide_score

        params_tuple = tuple(sorted(params.items()))
        scores_history[params_tuple] = positive_score

        # skopt minimizes the objective function, so we return -1 times the final score
        return -1.0 * final_score

    pbar = tqdm(total=n_calls, desc="Bayesian Optimization Progress", unit="iteration")
    def pbar_callback(res):
        pbar.update(1)
    early_stopper = EarlyStopper(delta=0.001, n_best=30)
    # 3. Run bayesian optimization
    objective_function = objective_function_guide if use_guide else vanilla_objective_function
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
    all_results = sorted(zip(result.func_vals, result.x_iters), key=lambda x: x[0])
    top_n_results = all_results[:top_n]
    final_presets = []
    presets_to_sort = []

    for rank, (score, params_list) in enumerate(top_n_results):
        best_params = dict(zip(param_names, params_list))
        final_config = []
        params_tuple = tuple(sorted(best_params.items()))
        benchmark_clap_score = scores_history.get(params_tuple, 0.0) if use_guide else -1.0 * score

        for fx in initial_config:
            fx_type = fx['type']

            is_active = (fx_type == "EQ") or (best_params.get(f"{fx_type}__activation", 0.0) > 0.5)

            if is_active:
                new_fx = fx.copy()
                for p_name, p_value in best_params.items():
                    p_fx_type, p_key = p_name.split('__')
                    if p_fx_type == fx_type and p_key != 'activation':
                        keys = p_key.split('.')
                        if len(keys) == 2:
                            if keys[0] not in new_fx: new_fx[keys[0]] = {}
                            new_fx[keys[0]][keys[1]] = p_value
                        else:
                            new_fx[p_key] = p_value
                final_config.append(new_fx)

        presets_to_sort.append({
            "composite_score": -1.0 * score,
            "benchmark_clap_score": benchmark_clap_score,
            "plugins": final_config
        })

        # 5. Re-sort presets by benchmark CLAP score and save audio files
        final_presets = sorted(presets_to_sort, key=lambda x: x['benchmark_clap_score'], reverse=True)

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
    return final_presets

ALL_PARAM_RANGES_PB = {
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
def fxsearcher_og(audio: str = None,
               prompt: str = None,
               outdir: str = None,
               model: str = "laion/clap-htsat-unfused",
               top_n: int = 5,
               n_calls: int = 100,
               use_guide: bool = False, fxs: list = None,
               initial_params: dict = None,
               plot: bool = False,
               all_param_ranges: dict = None):
    """Run the FX searcher.

    Parameters can be provided directly (programmatic use) or via CLI (when
    `audio`, `prompt` or `outdir` are not provided).
    """
    from datetime import datetime

    if all_param_ranges is not None:
        ALL_PARAM_RANGES = all_param_ranges
    else:
        ALL_PARAM_RANGES = ALL_PARAM_RANGES_PB

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
        initial_params = [
            {"type": "EQ", "mode": "shelf", "low_cut": 120.0, "high_cut": 12000.0, "q": 1.0, "gains": {},
             "peak1_freq": 200.0, "peak2_freq": 1000.0, "peak3_freq": 5000.0},
            {"type": "Distortion", "drive_db": 1.0},
            {"type": "Reverb", "room_size": 0.3, "damping": 0.5, "wet_level": 0.1},
            {"type": "Delay", "delay": 0.1},
        {"type": "PitchShift", "semitones": 0},
        {"type": "Bitcrush", "bit_depth": 0},
    ]
    if fxs is not None:
        initial_params = [fx for fx in initial_params if fx['type'] in fxs]
        PARAM_RANGES = {fx: ALL_PARAM_RANGES[fx] for fx in ALL_PARAM_RANGES if fx in fxs}

        print("Param Ranges", PARAM_RANGES)
    else:
        PARAM_RANGES = ALL_PARAM_RANGES

    print(f"\n--- Starting Bayesian Optimization for {', '.join(PARAM_RANGES.keys())} ---")

    args_dict = {
        'initial_config': initial_params,
        'audio': audio, 'sr': sr, 'PARAM_RANGES': PARAM_RANGES,
        'model_name': model, 'prompt': prompt, 'outdir': outdir,
        'top_n': top_n, 'n_calls': n_calls, 'use_guide': use_guide
    }

    final_result = refine_candidate_bayesian(args_dict, plot=plot)

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

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    fxsearcher()