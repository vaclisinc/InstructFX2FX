import glob as _glob
import json
import os
import tempfile

import torch

from effects.fx import FXChainFactory
from FxSearcher.fxsearcher import fxsearcher
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

# ASSUMPTION: Compressor goes through Pedalboard + BO, not DASP + gradient descent, since it has discrete attack/release times that are not well-suited to gradient-based optimization. If this changes, we may need to update the prompts and how we route FX in Parser.


class Parser:
    """Layer 2: routes to the correct initialization/optimization path.

    Cases:
        1. All FX in fx_chain already exist in session → optimize from existing params.
        2. None exist → LLM-initialize all, then optimize.
        3. Mixed → LLM-initialize the missing ones, merge, then optimize.

    Optimization (Sequential C):
        DASP effects (eq, rev)  → gradient descent via move_in_CLAP
        Pedalboard effects       → Bayesian BO via fxsearcher
        DASP runs first; its output audio is fed as input to the Pedalboard stage.
    """

    def __init__(self, llm_client, clap_model, device="cpu", n_iterations=100, lr=0.01):
        self.llm = llm_client
        self.clap_model = clap_model
        self.device = device
        self.n_iterations = n_iterations
        self.lr = lr

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def route(self, instruction: str, fx_chain: list, session: Session, audio) -> dict:
        """Return optimized params dict keyed by canonical FX name.

        Args:
            instruction: user descriptor (e.g. "warm")
            fx_chain: ordered list of canonical FX names from FXSelectorAgent
            session: current Session (read current_params, do NOT update here)
            audio: torch tensor [1, T] at 44100 Hz

        Returns:
            dict {canonical_fx_name: params_dict}
        """
        if not fx_chain:
            return {}

        existing = [fx for fx in fx_chain if session.fx_exists(fx)]
        missing  = [fx for fx in fx_chain if not session.fx_exists(fx)]

        if not missing:
            # Case 1: all FX already have params
            init_params = {fx: session.current_params[fx] for fx in fx_chain}
        elif not existing:
            # Case 2: no FX have params yet
            init_params = self._llm_init(instruction, fx_chain)
        else:
            # Case 3: some exist, some don't
            new_params  = self._llm_init(instruction, missing)
            init_params = {**{fx: session.current_params[fx] for fx in existing}, **new_params}

        return self._optimize(instruction, fx_chain, init_params, audio)

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
            prompt = PromptFactory.LLM_PARAMETER_INITIALIZATION_PROMPT_PEDALBOARD(
                instruction=instruction,
                effects=pb_fxs,
            )
            pb_dict = self.llm.generate_parameters(prompt)
            for fx in pb_fxs:
                dict_key = _PB_DICT_KEY.get(fx)
                if dict_key and dict_key in pb_dict:
                    result[fx] = pb_dict[dict_key]

        return result

    # ------------------------------------------------------------------
    # Optimization dispatch
    # ------------------------------------------------------------------

    def _optimize(self, instruction: str, fx_chain: list, init_params: dict, audio) -> dict:
        dasp_fxs = [fx for fx in fx_chain if fx in DASP_FX]
        pb_fxs   = [fx for fx in fx_chain if fx not in DASP_FX]

        result = dict(init_params)
        current_audio = audio

        # Stage 1: DASP gradient descent (eq, rev)
        if dasp_fxs:
            current_audio = self._optimize_dasp(instruction, dasp_fxs, init_params, current_audio, result)

        # Stage 2: Pedalboard BO on current_audio (DASP output if stage 1 ran)
        if pb_fxs:
            self._optimize_pb(instruction, pb_fxs, init_params, current_audio, result)

        return result

    def _optimize_dasp(self, instruction, dasp_fxs, init_params, audio, result):
        registry_keys = [_DASP_REGISTRY_KEY[fx] for fx in dasp_fxs]
        fx_chain_obj  = FXChainFactory.create_fx_chain_from_effects(registry_keys)

        dasp_init_dict = {_DASP_DICT_KEY[fx]: init_params[fx] for fx in dasp_fxs if fx in init_params}
        init_tensor    = fx_initial_params_to_tensor(dasp_init_dict, device=self.device)

        final_tensor, _, audios = move_in_CLAP(
            audio=audio,
            fx_chain=fx_chain_obj,
            initial_params=init_tensor,
            text_anchor="dry audio",
            target=instruction,
            clap_model=self.clap_model,
            n_iterations=self.n_iterations,
            lr=self.lr,
            device=self.device,
        )

        dasp_dict_keys  = [_DASP_DICT_KEY[fx] for fx in dasp_fxs]
        optimized_dict  = fx_tensor_to_params_dict(final_tensor, dasp_dict_keys)
        for fx in dasp_fxs:
            result[fx] = optimized_dict[_DASP_DICT_KEY[fx]]

        # Return DASP-processed audio for the PB stage
        return audios["end"][0]

    def _optimize_pb(self, instruction, pb_fxs, init_params, audio, result):
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
            return

        # Convert audio tensor to numpy for fxsearcher
        if isinstance(audio, torch.Tensor):
            audio_np = audio.squeeze(0).cpu().numpy()
        else:
            audio_np = audio

        with tempfile.TemporaryDirectory() as tmpdir:
            fxsearcher(
                audio=(audio_np, 44100),
                prompt=instruction,
                outdir=tmpdir,
                clap_model=self.clap_model,
                top_n=1,
                n_calls=self.n_iterations,
                use_guide=False,
                initial_params=pb_init,
                all_param_ranges=pb_param_ranges,
            )

            # Read best params back from JSON written by fxsearcher
            json_files = _glob.glob(os.path.join(tmpdir, "**", "best_presets.json"), recursive=True)
            if not json_files:
                return
            with open(json_files[0]) as f:
                best_output = json.load(f)

            results_list = best_output.get("results", [])
            if not results_list:
                return

            best_config = results_list[0].get("config", {})
            for fx in pb_fxs:
                dict_key = _PB_DICT_KEY.get(fx)
                if dict_key and dict_key in best_config:
                    result[fx] = best_config[dict_key]
