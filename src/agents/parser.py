import glob as _glob
import json
import os
import tempfile

import torch

from configurations.config import LossFunction
from effects.fx import FXChainFactory
from FxSearcher.fxsearcher import fxsearcher, render as pb_render
from prompts.prompt import PromptFactory
from session.session import Session
from training.trainer import move_in_CLAP
from utilities.fx_processing import fx_initial_params_to_tensor, fx_tensor_to_params_dict
from effects.fx import ALL_PARAM_RANGES_DASP, ALL_PARAM_RANGES_PB

# ---------------------------------------------------------------------------
# FX name mappings
# ---------------------------------------------------------------------------

# Canonical names that go through DASP + gradient descent
DASP_FX = {"eq", "rev"}

# canonical → DASP EFFECT_REGISTRY key (used to build FXChain)
_DASP_REGISTRY_KEY = {"eq": "eq", "rev": "reverb"}

# canonical → DASP params-dict key (used in param dicts / PromptFactory)
_DASP_DICT_KEY = {"eq": "EQ", "rev": "Reverb"}

# canonical → Pedalboard params-dict key (used in fxsearcher / PromptFactory)
_PB_DICT_KEY = {
    "comp":       "Compressor",
    "dist":       "Distortion",
    "delay":      "Delay",
    "pitchshift": "PitchShift",
    "bitcrush":   "Bitcrush",
    # "panner":     "Panner",
}

# canonical → PromptFactory effect name for Pedalboard prompts
_PB_PROMPT_KEY = {
    "comp":       "compressor",
    "dist":       "distortion",
    "delay":      "delay",
    "pitchshift": "pitchshift",
    "bitcrush":   "bitcrush",
}

# ASSUMPTION: Compressor goes through Pedalboard + BO, not DASP + gradient descent, since it has discrete attack/release times that are not well-suited to gradient-based optimization. If this changes, we may need to update the prompts and how we route FX in Parser.


class Parser:
    """Layer 2: routes to the correct initialization/optimization path.

    Cases:
        1. All FX in fx_chain already exist in session → re-optimize from existing params.
        2. None exist → LLM-initialize only (no optimization).
        3. Mixed → LLM-initialize the missing ones, merge with existing, then optimize all.

    Optimization (Sequential C):
        DASP effects (eq, rev)  → gradient descent via move_in_CLAP
        Pedalboard effects       → Bayesian BO via fxsearcher
        DASP runs first; its output audio is fed as input to the Pedalboard stage.
    """

    def __init__(self, llm_client, clap_model, device="cpu", n_iterations=100, lr=0.01,
                 loss_function=LossFunction.SEMANTIC_SIMILARITY_LOSS):
        self.llm = llm_client
        self.clap_model = clap_model
        self.device = device
        self.n_iterations = n_iterations
        self.lr = lr
        self.loss_function = loss_function

    def _resolve_run_settings(self, run_settings: dict | None) -> dict:
        run_settings = run_settings or {}
        snapshot_interval = run_settings.get("snapshot_interval")
        return {
            "n_iterations": int(run_settings.get("n_iterations", self.n_iterations)),
            "learning_rate": float(run_settings.get("learning_rate", self.lr)),
            "snapshot_interval": int(snapshot_interval) if snapshot_interval is not None else None,
            "loss_function": self.loss_function.value,
        }

    @staticmethod
    def _iteration_from_label(label: str) -> int | None:
        if label == "start":
            return 0
        if label == "end":
            return None
        if label.startswith("iter_"):
            try:
                return int(label.split("_", 1)[1])
            except ValueError:
                return None
        return None

    def _build_canonical_dasp_params(self, params_tensor, dasp_dict_keys: list[str], dasp_fxs: list[str]) -> dict:
        if params_tensor is None:
            return {}

        if not isinstance(params_tensor, torch.Tensor):
            params_tensor = torch.tensor(params_tensor, dtype=torch.float32)
        if params_tensor.dim() == 1:
            params_tensor = params_tensor.unsqueeze(0)

        params_dict = fx_tensor_to_params_dict(params_tensor, dasp_dict_keys)
        return {
            fx: params_dict[_DASP_DICT_KEY[fx]]
            for fx in dasp_fxs
            if _DASP_DICT_KEY[fx] in params_dict
        }

    def _build_dasp_trajectory(self, snapshots: dict, dasp_dict_keys: list[str], dasp_fxs: list[str]) -> list[dict]:
        trajectory = []
        for label, snapshot in snapshots.items():
            audio_tensor = None
            params_tensor = None
            needs_sigmoid = False

            if isinstance(snapshot, tuple) and len(snapshot) >= 2:
                audio_tensor, params_tensor = snapshot[0], snapshot[1]
            elif isinstance(snapshot, dict):
                audio_tensor = snapshot.get("audio")
                params_tensor = snapshot.get("params")
                needs_sigmoid = True

            if params_tensor is None:
                continue

            if not isinstance(params_tensor, torch.Tensor):
                params_tensor = torch.tensor(params_tensor, dtype=torch.float32)
            if params_tensor.dim() == 1:
                params_tensor = params_tensor.unsqueeze(0)

            normalized_tensor = torch.sigmoid(params_tensor) if needs_sigmoid else params_tensor
            canonical_params = self._build_canonical_dasp_params(
                normalized_tensor,
                dasp_dict_keys=dasp_dict_keys,
                dasp_fxs=dasp_fxs,
            )
            trajectory.append(
                {
                    "label": label,
                    "iteration": self._iteration_from_label(label),
                    "audio": audio_tensor.detach().cpu() if isinstance(audio_tensor, torch.Tensor) else audio_tensor,
                    "params": canonical_params,
                }
            )

        return trajectory

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def route(
        self,
        instruction: str,
        fx_chain: list,
        session: Session,
        audio,
        run_settings: dict | None = None,
        return_metadata: bool = False,
    ) -> dict:
        """Return optimized params dict keyed by canonical FX name.

        Args:
            instruction: user descriptor (e.g. "warm")
            fx_chain: ordered list of canonical FX names from FXSelectorAgent
            session: current Session (read current_params, do NOT update here)
            audio: torch tensor [1, T] at 44100 Hz

        Returns:
            (params_dict, rendered_audio) where params_dict is keyed by canonical FX name
        """
        settings = self._resolve_run_settings(run_settings)
        progress_callback = run_settings.get("progress_callback") if run_settings else None
        metadata = {
            "route_case": None,
            "settings_used": settings,
            "stages": {},
        }

        if not fx_chain:
            metadata["route_case"] = "empty_chain"
            if return_metadata:
                return {}, audio, metadata
            return {}, audio

        existing = [fx for fx in fx_chain if session.fx_exists(fx)]
        missing  = [fx for fx in fx_chain if not session.fx_exists(fx)]

        if not missing:
            # Case 1: all FX already have params → re-optimize from session init
            init_params = {fx: session.current_params[fx] for fx in fx_chain}
            metadata["route_case"] = "reuse_all"
            if progress_callback is not None:
                progress_callback({"type": "route", "route_case": "reuse_all", "fx_chain": list(fx_chain)})
            params, rendered_audio, optimize_meta = self._optimize(
                instruction,
                fx_chain,
                init_params,
                audio,
                settings,
                progress_callback,
            )
            metadata["stages"] = optimize_meta
            if return_metadata:
                return params, rendered_audio, metadata
            return params, rendered_audio
        elif not existing:
            # Case 2: no FX have params yet → LLM-initialize, then single render pass (no optimization loop)
            init_params = self._llm_init(instruction, fx_chain)
            rendered = self._render_once(init_params, audio)
            metadata["route_case"] = "initialize_all"
            if progress_callback is not None:
                progress_callback(
                    {
                        "type": "route",
                        "route_case": "initialize_all",
                        "fx_chain": list(fx_chain),
                    }
                )
            if return_metadata:
                return init_params, rendered, metadata
            return init_params, rendered
        else:
            # Case 3: some exist, some don't → LLM-init missing, merge, then optimize
            new_params  = self._llm_init(instruction, missing)
            init_params = {**{fx: session.current_params[fx] for fx in existing}, **new_params}
            metadata["route_case"] = "reuse_and_initialize"
            if progress_callback is not None:
                progress_callback({"type": "route", "route_case": "reuse_and_initialize", "fx_chain": list(fx_chain)})
            params, rendered_audio, optimize_meta = self._optimize(
                instruction,
                fx_chain,
                init_params,
                audio,
                settings,
                progress_callback,
            )
            metadata["stages"] = optimize_meta
            if return_metadata:
                return params, rendered_audio, metadata
            return params, rendered_audio

    # ------------------------------------------------------------------
    # LLM initialization
    # ------------------------------------------------------------------

    def _llm_init(self, instruction: str, fx_list: list) -> dict:
        dasp_fxs = [fx for fx in fx_list if fx in DASP_FX]
        pb_fxs   = [fx for fx in fx_list if fx not in DASP_FX]
        result   = {}

        if dasp_fxs:
            registry_keys = [_DASP_REGISTRY_KEY[fx] for fx in dasp_fxs]
            prompt = PromptFactory.LLM_PARAMETER_INITIALIZATION_PROMPT_DASP(
                fx_chain=registry_keys,
                instruction=instruction,
                effects=registry_keys,
            )
            params_dict = self.llm.generate_parameters(prompt)
            for fx in dasp_fxs:
                dict_key = _DASP_DICT_KEY[fx]
                if dict_key in params_dict:
                    result[fx] = params_dict[dict_key]

        if pb_fxs:
            prompt_names = [_PB_PROMPT_KEY[fx] for fx in pb_fxs]
            prompt = PromptFactory.LLM_PARAMETER_INITIALIZATION_PROMPT_PEDALBOARD(
                instruction=instruction,
                effects=prompt_names,
            )
            pb_dict = self.llm.generate_parameters(prompt)
            for fx in pb_fxs:
                dict_key = _PB_DICT_KEY.get(fx)
                if dict_key and dict_key in pb_dict:
                    result[fx] = pb_dict[dict_key]

        return result

    def _render_once(self, init_params: dict, audio) -> torch.Tensor:
        """Apply LLM-initialized params to audio in a single forward pass (no optimization loop).

        DASP FX (eq, rev) use the DASP FX chain; PB FX use pedalboard.
        Audio flows DASP → PB, matching the _optimize stage order.
        """
        dasp_fxs = [fx for fx in init_params if fx in DASP_FX]
        pb_fxs   = [fx for fx in init_params if fx not in DASP_FX and fx in _PB_DICT_KEY]
        current_audio = audio

        if dasp_fxs:
            registry_keys  = [_DASP_REGISTRY_KEY[fx] for fx in dasp_fxs]
            fx_chain_obj   = FXChainFactory.create_fx_chain_from_effects(registry_keys)
            dasp_init_dict = {_DASP_DICT_KEY[fx]: init_params[fx] for fx in dasp_fxs}
            init_tensor    = fx_initial_params_to_tensor(dasp_init_dict, device=self.device)
            with torch.no_grad():
                current_audio = fx_chain_obj(
                    current_audio.to(self.device), torch.sigmoid(init_tensor)
                ).cpu()

        if pb_fxs:
            render_config = {_PB_DICT_KEY[fx]: init_params[fx] for fx in pb_fxs}
            if isinstance(current_audio, torch.Tensor):
                audio_np = current_audio.squeeze(0).cpu().numpy()
            else:
                audio_np = current_audio
            rendered_np   = pb_render(audio_np, 44100, render_config)
            current_audio = torch.from_numpy(rendered_np).unsqueeze(0)

        return current_audio

    # ------------------------------------------------------------------
    # Optimization dispatch
    # ------------------------------------------------------------------

    def _optimize(self, instruction: str, fx_chain: list, init_params: dict, audio, settings: dict, progress_callback=None):
        dasp_fxs = [fx for fx in fx_chain if fx in DASP_FX]
        pb_fxs   = [fx for fx in fx_chain if fx not in DASP_FX]

        result = dict(init_params)
        current_audio = audio
        stage_metadata = {}

        # Stage 1: DASP gradient descent (eq, rev)
        if dasp_fxs:
            current_audio, dasp_meta = self._optimize_dasp(
                instruction,
                dasp_fxs,
                init_params,
                current_audio,
                result,
                settings,
                progress_callback,
            )
            stage_metadata["dasp"] = dasp_meta

        # Stage 2: Pedalboard BO on current_audio (DASP output if stage 1 ran)
        if pb_fxs:
            pb_audio, pb_meta = self._optimize_pb(
                instruction,
                pb_fxs,
                init_params,
                current_audio,
                result,
                settings,
                progress_callback,
            )
            stage_metadata["pedalboard"] = pb_meta
            if pb_audio is not None:
                current_audio = pb_audio

        return result, current_audio, stage_metadata

    def _optimize_dasp(self, instruction, dasp_fxs, init_params, audio, result, settings, progress_callback=None):
        registry_keys = [_DASP_REGISTRY_KEY[fx] for fx in dasp_fxs]
        fx_chain_obj  = FXChainFactory.create_fx_chain_from_effects(registry_keys)

        dasp_init_dict = {_DASP_DICT_KEY[fx]: init_params[fx] for fx in dasp_fxs if fx in init_params}
        init_tensor    = fx_initial_params_to_tensor(dasp_init_dict, device=self.device)

        if isinstance(audio, torch.Tensor):
            audio = audio.to(self.device)

        def _trainer_progress(event: dict):
            if progress_callback is None:
                return

            if event.get("type") == "checkpoint":
                checkpoint_params = self._build_canonical_dasp_params(
                    event.get("params"),
                    dasp_dict_keys=[_DASP_DICT_KEY[fx] for fx in dasp_fxs],
                    dasp_fxs=dasp_fxs,
                )
                progress_callback(
                    {
                        "type": "checkpoint",
                        "stage": "dasp",
                        "label": event.get("label"),
                        "iteration": event.get("iteration"),
                        "total_iterations": event.get("total_iterations"),
                        "audio": event.get("audio"),
                        "params": checkpoint_params,
                        "loss": event.get("loss"),
                    }
                )
            else:
                forwarded = dict(event)
                forwarded["stage"] = "dasp"
                progress_callback(forwarded)

        final_tensor, final_audio, loss_history, audios = move_in_CLAP(
            audio=audio,
            fx_chain=fx_chain_obj,
            initial_params=init_tensor,
            text_anchor="dry audio",
            target=instruction,
            clap_model=self.clap_model,
            loss_function=self.loss_function,
            n_iterations=settings["n_iterations"],
            lr=settings["learning_rate"],
            snapshot_interval=settings["snapshot_interval"],
            device=self.device,
            progress_callback=_trainer_progress,
        )

        dasp_dict_keys  = [_DASP_DICT_KEY[fx] for fx in dasp_fxs]
        optimized_dict  = fx_tensor_to_params_dict(final_tensor, dasp_dict_keys)
        for fx in dasp_fxs:
            result[fx] = optimized_dict[_DASP_DICT_KEY[fx]]

        # Return DASP-processed audio for the PB stage
        stage_metadata = {
            "fx": list(dasp_fxs),
            "optimizer": "gradient_descent",
            "loss_function": self.loss_function.value,
            "loss_history": loss_history,
            "trajectory_supported": True,
            "trajectory": self._build_dasp_trajectory(audios, dasp_dict_keys, dasp_fxs),
        }
        return final_audio, stage_metadata

    def _optimize_pb(self, instruction, pb_fxs, init_params, audio, result, settings, progress_callback=None):
        pb_init = {
            _PB_DICT_KEY[fx]: init_params[fx]
            for fx in pb_fxs
            if fx in init_params and fx in _PB_DICT_KEY
        }
        pb_param_ranges = {
            _PB_DICT_KEY[fx]: ALL_PARAM_RANGES_PB[_PB_DICT_KEY[fx]]
            for fx in pb_fxs
            if fx in _PB_DICT_KEY and _PB_DICT_KEY[fx] in ALL_PARAM_RANGES_PB
        }

        if not pb_param_ranges:
            return None, {
                "fx": list(pb_fxs),
                "optimizer": "bayesian_optimization",
                "trajectory_supported": False,
                "trajectory": [],
            }

        # Convert audio tensor to numpy for fxsearcher
        if isinstance(audio, torch.Tensor):
            audio_np = audio.squeeze(0).cpu().numpy()
        else:
            audio_np = audio

        with tempfile.TemporaryDirectory() as tmpdir:
            pb_audio, _, _ = fxsearcher(
                audio=(audio_np, 44100),
                prompt=instruction,
                outdir=tmpdir,
                clap_model=self.clap_model,
                top_n=1,
                n_calls=settings["n_iterations"],
                use_guide=False,
                initial_params=pb_init,
                all_param_ranges=pb_param_ranges,
            )

            # Read best params back from JSON written by fxsearcher
            json_files = _glob.glob(os.path.join(tmpdir, "**", "best_presets.json"), recursive=True)
            if not json_files:
                return None
            with open(json_files[0]) as f:
                best_output = json.load(f)

            results_list = best_output.get("results", [])
            if not results_list:
                return None, {
                    "fx": list(pb_fxs),
                    "optimizer": "bayesian_optimization",
                    "trajectory_supported": False,
                    "trajectory": [],
                }

            best_config = results_list[0].get("plugins", {})
            for fx in pb_fxs:
                dict_key = _PB_DICT_KEY.get(fx)
                if dict_key and dict_key in best_config:
                    result[fx] = best_config[dict_key]

            return pb_audio, {
                "fx": list(pb_fxs),
                "optimizer": "bayesian_optimization",
                "trajectory_supported": False,
                "trajectory": [],
                "best_config": best_config,
            }
