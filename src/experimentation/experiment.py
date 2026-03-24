"""
Experimentation framework for running and comparing different audio processing methods.

Supports:
- InstructFX2FX: LLM-based parameter initialization and gradient-based refinement
- LLM+LLM: Two-stage LLM approach for parameter generation
- TextFX+Text2FX: Baseline text-to-audio-effects methods
"""

from dataclasses import dataclass
import os
import json
from enum import Enum
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path

import torch
import numpy as np
import soundfile as sf
import librosa

from configurations.config import (
    Config,
    ParameterInitializationMethod,
    OptimizationMethod,
    LossFunction,
)
from prompts.prompt import Prompt, PromptFactory
from src.utilities.fx_processing import fx_tensor_to_params_dict
from training.parameterengine import ParameterEngine
from effects.fx import FXChainFactory
from llms.llmclient import LLMClient
from embeddings.clap import CLAPWrapper


class Method(Enum):
    InstructFX2FX = "InstructFX2FX"
    LLM_LLM = "LLM+LLM"
    TextFX_Text2FX = "TextFX_Text2FX"


@dataclass
class InstructionSet(ABC):
    """Base class for instruction sets."""
    task: str = ""
    instruction: str = ""
    text_anchor: str = ""
    text_target: str = ""


class InstructionSet1(InstructionSet):
    """Standard instruction set for audio transformation tasks."""
    task: str = ""
    instruction: str = ""
    text_anchor: str = ""
    text_target: str = ""

    def __init__(self, anchor: str = "", target: str = "", context: str = ""):
        self.task = f"Make this sound more {target}."
        self.text_anchor = f"This sound is {anchor}"
        self.text_target = f"This sound is {target}"
        if anchor:
            self.instruction = f"This is {context}, but the sound is {anchor}. {self.task}"
        else:
            self.instruction = f"This is {context}. {self.task}"

    def to_dict(self) -> Dict[str, str]:
        return {
            "task": self.task,
            "instruction": self.instruction,
            "text_anchor": self.text_anchor,
            "text_target": self.text_target,
        }


@dataclass
class ExperimentResult:
    """Container for experiment results."""
    method: str
    audio_path: str
    parameters: Dict[str, Any]
    output_audio_path: str
    stages_dir: str = ""
    metrics: Optional[Dict[str, float]] = None
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


def _save_audio_and_params(
    audio_tensor: torch.Tensor,
    params_tensor: torch.Tensor,
    params_dict: Dict[str, Any],
    fx_chain,
    sample_rate: int,
    output_dir: Path,
    stage_name: str,
    details: Dict[str, Any] = None,
) -> Tuple[str, str]:
    """
    Process audio through FX chain and save both audio and parameters.

    Returns:
        Tuple of (audio_file_path, params_file_path)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process audio
    if params_tensor is not None:
        output_audio = fx_chain(audio_tensor, params_tensor)
        output_audio_np = output_audio.squeeze().detach().cpu().numpy()
    else:
        output_audio_np = audio_tensor.squeeze().detach().cpu().numpy()

    # Save audio
    audio_path = output_dir / f"{stage_name}.wav"
    sf.write(audio_path, output_audio_np, sample_rate)

    # Save parameters
    params_path = output_dir / f"{stage_name}_params.json"
    with open(params_path, "w") as f:
        json.dump(
            {"params": params_dict or {},
             "details": details or {}}, f, indent=2
        )

    return str(audio_path), str(params_path)


def _ensure_bct(audio_tensor: torch.Tensor) -> torch.Tensor:
    """Ensure audio is shaped as [batch, channels, time] for dasp_pytorch."""
    if audio_tensor.dim() == 1:
        return audio_tensor.unsqueeze(0).unsqueeze(0)
    if audio_tensor.dim() == 2:
        return audio_tensor.unsqueeze(0)
    if audio_tensor.dim() == 3:
        return audio_tensor
    raise ValueError(f"Expected audio tensor with 1-3 dims, got shape {tuple(audio_tensor.shape)}")


def run_LLM_LLM(
    audio: torch.Tensor,
    fx_chain,
    instructionset_initialization: InstructionSet1,
    instructionset_refinement: InstructionSet1,
    llm_client: LLMClient,
    embedding: CLAPWrapper,
    sample_rate: int = 44100,
    device: str = "cuda",
    iterations: int = 100,
    experiment_dir: Optional[Path] = None,
    filename_prefix: str = "audio",
    effects: List[str] = None,
    initial_params_tensor: Optional[torch.Tensor] = None,
    initial_params_dict: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, Dict[str, Any], Dict[str, Any]]:
    """
    Run LLM+LLM method: two-stage LLM approach for parameter generation.

    Args:
        audio: Input audio tensor [batch, channels, time]
        fx_chain: FX chain to process audio
        instructionset_initialization: Initial instruction set
        instructionset_refinement: Refinement instruction set
        llm_client: LLM client for parameter generation
        embedding: CLAP embedding model
        sample_rate: Sample rate for audio
        device: Device to run computations on
        iterations: Number of refinement iterations
        experiment_dir: Directory to save intermediate results
        filename_prefix: Prefix for saved files

    Returns:
        Tuple of (final_params_tensor, final_params_dict, stage_info)
    """
    parameter_engine = ParameterEngine()
    stage_info = {}

    # Create stages directory
    if experiment_dir:
        stages_dir = Path(experiment_dir) / "intermediate"
        stages_dir.mkdir(parents=True, exist_ok=True)

        # Save original audio
        audio_path, params_path = _save_audio_and_params(
            audio, None, {}, fx_chain, sample_rate, stages_dir, "00_original"
        )
        stage_info["original"] = {"audio": audio_path, "params": params_path}
        print(f"✓ Saved original audio to: {audio_path}")

    # Stage 1: Initialize parameters with LLM (skip if pre-generated params provided)
    init_details = {}
    if initial_params_tensor is None:
        config_initialization = Config(
            prompt=PromptFactory.LLM_PARAMETER_INITIALIZATION_PROMPT_DASP(
                fx_chain=fx_chain, instruction=instructionset_initialization.instruction,
                effects=effects,
            ),
            initialization_method=ParameterInitializationMethod.LLM,
            llmclient=llm_client,
            fx_chain=fx_chain,
            embedding=embedding,
            device=device,
        )
        initial_params_tensor, initial_params_dict, _ = parameter_engine.get_params(
            audio, config_initialization
        )
        init_details = {"sys_prompt": config_initialization.prompt.sys_prompt}
    else:
        init_details = {"source": "pre-generated fixed init"}

    # Store initial params so they can be reused by InstructFX2FX
    stage_info["initial_params_tensor"] = initial_params_tensor
    stage_info["initial_params_dict"] = initial_params_dict

    if experiment_dir:
        audio_path, params_path = _save_audio_and_params(
            audio, initial_params_tensor, initial_params_dict, fx_chain,
            sample_rate, stages_dir, "01_initialized", init_details
        )
        stage_info["initialization"] = {"audio": audio_path, "params": params_path}
        print(f"✓ Saved initialization stage to: {audio_path}")

    # Stage 2: Refine parameters with second LLM call
    config_refinement = Config(
        prompt=PromptFactory.LLM_PARAMETER_REFINEMENT_PROMPT_DASP(
            fx_chain,
            f"The current parameters are: {initial_params_dict}\n{instructionset_refinement.instruction}",
            current_parameters_dict=initial_params_dict,
            effects=effects,
        ),
        initialization_method=ParameterInitializationMethod.LLM,
        llmclient=llm_client,
        fx_chain=fx_chain,
        embedding=embedding,
        device=device,
        num_iterations=iterations,
    )

    refined_params_tensor, refined_params_dict, _ = parameter_engine.get_params(
        audio, config_refinement
    )

    if experiment_dir:
        audio_path, params_path = _save_audio_and_params(
            audio, refined_params_tensor, refined_params_dict, fx_chain,
            sample_rate, stages_dir, "02_refined", {"sys_prompt": config_refinement.prompt.sys_prompt}
        )
        stage_info["refinement"] = {"audio": audio_path, "params": params_path}
        print(f"✓ Saved refinement stage to: {audio_path}")

    return refined_params_tensor, refined_params_dict, stage_info


def run_InstructFX2FX(
    audio: torch.Tensor,
    fx_chain,
    instructionset_initialization: InstructionSet1,
    instructionset_refinement: InstructionSet1,
    llm_client: LLMClient,
    embedding: CLAPWrapper,
    sample_rate: int = 44100,
    device: str = "cuda",
    iterations: int = 1000,
    learning_rate: float = 0.01,
    optimization_method: OptimizationMethod = OptimizationMethod.GRADIENT_DESCENT,
    experiment_dir: Optional[Path] = None,
    filename_prefix: str = "audio",
    initial_params_tensor: Optional[torch.Tensor] = None,
    initial_params_dict: Optional[Dict[str, Any]] = None,
    snapshot_interval: Optional[int] = None,
    effects: List[str] = None,
) -> Tuple[torch.Tensor, Dict[str, Any], Dict[str, Any]]:
    """
    Run InstructFX2FX method: LLM initialization + gradient-based refinement.

    Args:
        audio: Input audio tensor [batch, channels, time]
        fx_chain: FX chain to process audio
        instructionset_initialization: Initial instruction set
        instructionset_refinement: Refinement instruction set
        llm_client: LLM client for parameter generation
        embedding: CLAP embedding model
        sample_rate: Sample rate for audio
        device: Device to run computations on
        iterations: Number of optimization iterations
        learning_rate: Learning rate for gradient descent
        optimization_method: Optimization method to use
        experiment_dir: Directory to save intermediate results
        filename_prefix: Prefix for saved files

    Returns:
        Tuple of (final_params_tensor, final_params_dict, stage_info)
    """
    parameter_engine = ParameterEngine()
    stage_info = {}

    # Create stages directory
    if experiment_dir:
        stages_dir = Path(experiment_dir) / "intermediate"
        stages_dir.mkdir(parents=True, exist_ok=True)

        # Save original audio
        audio_path, params_path = _save_audio_and_params(
            audio, None, {}, fx_chain, sample_rate, stages_dir, "00_original"
        )
        stage_info["original"] = {"audio": audio_path, "params": params_path}
        print(f"✓ Saved original audio to: {audio_path}")

    # Stage 1: LLM initialization
    # If initial params are not pre-provided (i.e. LLM_LLM did not run first),
    # call the LLM explicitly here so we can save the initialized state to disk.
    if initial_params_tensor is None:
        config_init = Config(
            prompt=PromptFactory.LLM_PARAMETER_INITIALIZATION_PROMPT_DASP(
                fx_chain, instructionset_initialization.instruction, effects=effects
            ),
            initialization_method=ParameterInitializationMethod.LLM,
            llmclient=llm_client,
            fx_chain=fx_chain,
            embedding=embedding,
            device=device,
        )
        initial_params_tensor, initial_params_dict, _ = parameter_engine.get_params(
            audio, config_init
        )

    if experiment_dir:
        source = "reused from LLM_LLM" if stage_info.get("initialization") is None else "LLM init"
        audio_path, params_path = _save_audio_and_params(
            audio, initial_params_tensor, initial_params_dict or {}, fx_chain,
            sample_rate, stages_dir, "01_initialized", {"source": source}
        )
        stage_info["initialization"] = {"audio": audio_path, "params": params_path}
        print(f"✓ Saved initialization stage to: {audio_path}")

    # Stage 2: Gradient descent refinement (initial_params always available now)
    _snapshot_interval = snapshot_interval if snapshot_interval is not None else max(1, iterations // 10)
    config_refinement = Config(
        prompt=PromptFactory.LLM_PARAMETER_INITIALIZATION_PROMPT_DASP(
            fx_chain, instructionset_initialization.instruction, effects=effects
        ),
        initialization_method=ParameterInitializationMethod.INPUT,
        loss_function=LossFunction.DIRECTIONAL_LOSS,
        optimization_method=optimization_method,
        text_anchor=instructionset_refinement.text_anchor,
        text_target=instructionset_refinement.text_target,
        llmclient=llm_client,
        fx_chain=fx_chain,
        embedding=embedding,
        device=device,
        num_iterations=iterations,
        learning_rate=learning_rate,
        save_checkpoints=True,
        snapshot_interval=_snapshot_interval,
    )

    refined_params_tensor, history , audios = parameter_engine.get_params(
        audio,
        config_refinement,
        initial_params_dict=initial_params_dict,
        initial_params_tensor=initial_params_tensor,
    )
    refined_params_dict = fx_tensor_to_params_dict(refined_params_tensor) # FIXME is this still in the right order when we don't have all effects in the tensor?

    # Save refinement stage
    if experiment_dir:
        audio_path, params_path = _save_audio_and_params(
            audio, refined_params_tensor, history, fx_chain,
            sample_rate, stages_dir, "02_refined"
        )
        stage_info["refinement"] = {"audio": audio_path, "params": params_path}
        print(f"✓ Saved refinement stage to: {audio_path}")

        # Save intermediate snapshots from history.
        # Supports either:
        # - dict[str, tuple(audio_tensor, params_tensor)]
        # - list[tuple(audio_tensor, params_tensor)] / list[dict]
        if history:
            intermediate_dir = stages_dir / "optimization_steps"
            intermediate_dir.mkdir(parents=True, exist_ok=True)

            if isinstance(history, dict):
                history_items = list(history.items())
            else:
                history_items = [(f"iter_{idx:04d}", item) for idx, item in enumerate(history)]

            saved_count = 0
            for item_key, item_value in history_items:
                stage_name = str(item_key)
                snap_params_tensor = None

                if isinstance(item_value, tuple) and len(item_value) >= 2:
                    snap_params_tensor = item_value[1]
                elif isinstance(item_value, dict):
                    snap_params_tensor = item_value.get("params")

                if snap_params_tensor is None:
                    continue

                if not isinstance(snap_params_tensor, torch.Tensor):
                    try:
                        snap_params_tensor = torch.tensor(snap_params_tensor, dtype=audio.dtype, device=audio.device)
                    except Exception:
                        continue

                if snap_params_tensor.dim() == 1:
                    snap_params_tensor = snap_params_tensor.unsqueeze(0)
                snap_params_tensor = snap_params_tensor.to(device=audio.device, dtype=audio.dtype)

                try:
                    # Apply param tensor values directly to the original audio.
                    effected = fx_chain(audio, snap_params_tensor)
                except Exception:
                    continue

                effected_np = effected.squeeze().detach().cpu().numpy()
                snap_path = intermediate_dir / f"{stage_name}.wav"
                sf.write(snap_path, effected_np, sample_rate)

                snap_params_path = intermediate_dir / f"{stage_name}_params.json"
                with open(snap_params_path, "w") as f:
                    json.dump(
                        {
                            "params_tensor": snap_params_tensor.detach().cpu().tolist(),
                        },
                        f,
                        indent=2,
                    )
                saved_count += 1

            print(f"✓ Saved {saved_count} intermediate snapshots")

    return refined_params_tensor, refined_params_dict, stage_info


def run_experiments(
    methods: list[Method],
    instructionset_initialization: InstructionSet1,
    instructionset_refinement: InstructionSet1,
    raw_audio_paths: List[str],
    llm_client: LLMClient,
    embedding: CLAPWrapper,
    sample_rate: int = 44100,
    iterations: int = 1000,
    results_dir: str = "results",
    device: str = "cpu",
    nr_of_experiments_per_file: int = 1,
    fx_chain=None,
    effects: List[str] = None,
    snapshot_interval: Optional[int] = None,
    fixed_init: bool = True,
) -> Tuple[List[ExperimentResult], str]:
    """
    Run experiments across multiple audio files.

    Args:
        method: The method to use (InstructFX2FX, LLM+LLM, etc.).
        instructionset_initialization: Initial instruction set.
        instructionset_refinement: Refinement instruction set.
        raw_audio_paths: List of paths to raw audio files.
        llm_client: LLM client for parameter generation.
        embedding: CLAP embedding model.
        sample_rate: Audio sample rate.
        iterations: Number of optimization iterations.
        results_dir: Directory to save results.
        device: Device to run computations on.

    Returns:
        (results, experiment_dir): list of ExperimentResult objects and the
        path to the experiment_{timestamp}/ directory where all outputs were saved.
    """
    results_dir_path = Path(results_dir)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_dir = results_dir_path / f"experiment_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available() and device == "cpu":
        print("⚠️  CUDA is available but 'cpu' was specified. Consider using device='cuda'.")
    elif torch.cuda.is_available() and device == "cuda":
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(f"✓ Using CPU")

    # Create FX chain (use provided chain, or build from effects list)
    if fx_chain is None:
        _effects = effects if effects is not None else ["eq", "compressor", "reverb"]
        fx_chain = FXChainFactory.create_fx_chain_from_effects(_effects, sample_rate=sample_rate, device=device)

    results = []

    for audio_iteration, audio_path in enumerate(raw_audio_paths):


        print(f"\n{'='*60}")
        print(f"Processing audio {audio_iteration+1}/{len(raw_audio_paths)}: {Path(audio_path).name}")
        print(f"{'='*60}")

        # Load audio
        audio, sr = sf.read(audio_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)  # Convert to mono
        if sr != sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)

        # Convert to tensor
        audio_tensor = torch.from_numpy(audio).float().to(device)
        audio_tensor = _ensure_bct(audio_tensor)

        file_dir = experiment_dir / Path(audio_path).stem
        file_dir.mkdir(parents=True, exist_ok=True)

        # fixed_init=True: call the LLM once before the runs loop and reuse
        # the same init params for all runs and both methods.
        # fixed_init=False: each run generates its own init on the fly.
        fixed_init_params_tensor: Optional[torch.Tensor] = None
        fixed_init_params_dict: Optional[Dict[str, Any]] = None
        if fixed_init:
            print("[fixed_init] Generating single LLM init shared across all runs …")
            _pe = ParameterEngine()
            _cfg_init = Config(
                prompt=PromptFactory.LLM_PARAMETER_INITIALIZATION_PROMPT_DASP(
                    fx_chain=fx_chain,
                    instruction=instructionset_initialization.instruction,
                    effects=effects,
                ),
                initialization_method=ParameterInitializationMethod.LLM,
                llmclient=llm_client,
                fx_chain=fx_chain,
                embedding=embedding,
                device=device,
            )
            fixed_init_params_tensor, fixed_init_params_dict, _ = _pe.get_params(audio_tensor, _cfg_init)
            print("[fixed_init] Init generated — reusing for all runs.")

        for run in range(1, nr_of_experiments_per_file + 1):

            # All runs share the same init (fixed_init=True) or generate their
            # own on the fly via LLM_LLM (fixed_init=False).
            if fixed_init:
                llm_init_params_tensor = fixed_init_params_tensor
                llm_init_params_dict = fixed_init_params_dict
            else:
                llm_init_params_tensor = None
                llm_init_params_dict = None

            # Ensure LLM_LLM runs before InstructFX2FX so init params are available
            sorted_methods = sorted(methods, key=lambda m: 0 if m == Method.LLM_LLM else 1)

            # Run selected method
            for method in sorted_methods:
                run_dir = file_dir / method.value / f"run_{run}"
                run_dir.mkdir(parents=True, exist_ok=True)

                if method.value == "LLM+LLM":
                    params_tensor, params_dict, stage_info = run_LLM_LLM(
                        audio=audio_tensor,
                        fx_chain=fx_chain,
                        instructionset_initialization=instructionset_initialization,
                        instructionset_refinement=instructionset_refinement,
                        llm_client=llm_client,
                        embedding=embedding,
                        sample_rate=sample_rate,
                        device=device,
                        iterations=iterations,
                        experiment_dir=run_dir,
                        filename_prefix=Path(audio_path).stem,
                        effects=effects,
                        initial_params_tensor=llm_init_params_tensor,
                        initial_params_dict=llm_init_params_dict,
                    )
                    # Capture LLM init params for InstructFX2FX to reuse
                    # (in fixed_init mode these are already set; in stochastic
                    #  mode we grab them from stage_info as before)
                    if not fixed_init:
                        llm_init_params_tensor = stage_info.get("initial_params_tensor")
                        llm_init_params_dict = stage_info.get("initial_params_dict")

                elif method.value == "InstructFX2FX":
                    params_tensor, params_dict, stage_info = run_InstructFX2FX(
                        audio=audio_tensor,
                        fx_chain=fx_chain,
                        instructionset_initialization=instructionset_initialization,
                        instructionset_refinement=instructionset_refinement,
                        llm_client=llm_client,
                        embedding=embedding,
                        sample_rate=sample_rate,
                        device=device,
                        iterations=iterations,
                        experiment_dir=run_dir,
                        filename_prefix=Path(audio_path).stem,
                        initial_params_tensor=llm_init_params_tensor,
                        initial_params_dict=llm_init_params_dict,
                        snapshot_interval=snapshot_interval,
                        effects=effects,
                    )
                else:
                    raise ValueError(f"Unknown method: {method}")

                # Process audio with final parameters
                output_audio = fx_chain(audio_tensor, params_tensor)
                output_audio_np = output_audio.squeeze().detach().cpu().numpy()

                # Save final output audio
                audio_filename = Path(audio_path).stem
                output_audio_path = run_dir / f"{audio_filename}_final_output.wav"
                sf.write(output_audio_path, output_audio_np, sample_rate)

                # Create result object
                result = ExperimentResult(
                    method=method.value,
                    audio_path=audio_path,
                    parameters=params_dict or {},
                    output_audio_path=str(output_audio_path),
                    stages_dir=str(run_dir / "intermediate"),
                    timestamp=timestamp,
                )
                results.append(result)

                # Save individual result
                project_root = Path.cwd()
                result_json_path = run_dir / f"{audio_filename}_run{run}_summary.json"
                with open(result_json_path, "w") as f:
                    json.dump(
                        {
                            "method": result.method,
                            "input_audio_path": os.path.relpath(result.audio_path, project_root),
                            "final_output_audio_path": os.path.relpath(result.output_audio_path, project_root),
                            "stages_directory": os.path.relpath(result.stages_dir, project_root),
                            "final_parameters": result.parameters,
                            "timestamp": result.timestamp,
                        },
                        f,
                        indent=2,
                    )

                print(f"✓ Saved final output to: {output_audio_path}")
                print(f"✓ Saved result summary to: {result_json_path}")

                summary_path = file_dir / method.value / f"{Path(audio_path).stem}_experiment_summary.json"
                with open(summary_path, "w") as f:
                    json.dump(
                        {
                            "method": method.value,
                            "timestamp": timestamp,
                            "runs": nr_of_experiments_per_file,
                            "sample_rate": sample_rate,
                            "instruction_init": instructionset_initialization.to_dict(),
                            "instruction_refine": instructionset_refinement.to_dict(),
                            "original_audio_path": os.path.relpath(audio_path, project_root),
                            "runs": [
                                {
                                    "run": run,
                                    "run_directory": os.path.relpath(str(run_dir), project_root),
                                    "final_output_audio_path": os.path.relpath(r.output_audio_path, project_root),
                                }
                                for r in results
                            ],
                        },
                        f,
                        indent=2,
                    )

    # Save master summary
    summary_path = experiment_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "methods": [method.value for method in methods],
                "timestamp": timestamp,
                "num_audio_files": len(raw_audio_paths),
                "audio_files": [
                    {
                        "file_name": str(Path(p).stem),
                        "num_runs": nr_of_experiments_per_file,
                    }
                    for p in raw_audio_paths
                ],
                "embedding_model": embedding.__class__.__name__,
                "user": os.getenv("USER") or os.getenv("USERNAME") or "unknown",
            },
            f,
            indent=2,
        )

    print(f"\n{'='*60}")
    print(f"✓ Experiment complete!")
    print(f"  Results saved to: {experiment_dir}")
    print(f"  Summary: {summary_path}")
    print(f"{'='*60}\n")

    return results, str(experiment_dir)
