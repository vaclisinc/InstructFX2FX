from configurations.config import Config, LossFunction, OptimizationMethod
from training.loss import refine_with_directional_loss
from effects.fx import initialize_random_params, initialize_uniform_params,llm_params_dict_example, llm_params_tensor_example, llm_params_dict_example_pedalboard
from configurations.config import ParameterInitializationMethod
from utilities.fx_processing import fx_initial_params_to_tensor
from effects.fx import ALL_PARAM_RANGES
from FxSearcher.fxsearcher import fxsearcher, ALL_PARAM_RANGES_FXSearcher
from dataclasses import dataclass
import torch

@dataclass
class ParameterEngine:

    def _initialize_parameters(self, config) -> dict:
        if not config:
            raise ValueError("Config must be set to initialize parameters.")

        init_method_value = config.initialization_method.value if config.initialization_method else None

        if init_method_value == ParameterInitializationMethod.RANDOM.value:
            initial_params_dict = initialize_random_params()
            initial_params_tensor = fx_initial_params_to_tensor(initial_params_dict, device=config.device, param_ranges=ALL_PARAM_RANGES)

        elif init_method_value == ParameterInitializationMethod.LLM.value:
            print("🔄 Initializing parameters using LLM...")
            if config.llmclient is None:
                raise ValueError("LLM client must be provided in config for LLM-based initialization.")

            if config.effects_type == "Pedalboard":
                if llm_params_dict_example_pedalboard:
                    llm_params_dict = llm_params_dict_example_pedalboard
                else:
                    llm_params_dict = config.llmclient.generate_parameters(config.prompt)
                llm_params_tensor = torch.zeros((1, 49), device=config.device) + 0.5
                print(f"✓ LLM generated parameters for Bayesian Optimization: {llm_params_dict.keys()}")
            else:
                print("Generating new parameters using LLM client...")
                llm_params_dict = config.llmclient.generate_parameters(config.prompt)
                llm_params_tensor = fx_initial_params_to_tensor(llm_params_dict, device=config.device, param_ranges=ALL_PARAM_RANGES)
                print(f"✓ LLM generated {llm_params_tensor.shape[1]} parameters")
                print(llm_params_dict, llm_params_tensor)
            # else:
            #     print("✓ Using existing LLM-generated example parameters for Optimization")
            #     llm_params_tensor = fx_initial_params_to_tensor(llm_params_dict_example, device=config.device, param_ranges=ALL_PARAM_RANGES)
            #     llm_params_dict = llm_params_dict_example
            initial_params_dict = llm_params_dict
            initial_params_tensor = llm_params_tensor


        elif init_method_value == ParameterInitializationMethod.UNIFORM.value:
            initial_params_dict = initialize_uniform_params()
            initial_params_tensor = fx_initial_params_to_tensor(initial_params_dict, device=config.device, param_ranges=ALL_PARAM_RANGES)

        else:
            raise ValueError(f"Unknown initialization method: {config.initialization_method}")
        return initial_params_dict, initial_params_tensor


    def get_params(self, audio, config, initial_params_dict = None, initial_params_tensor = None):
        """
        Return normalized, sigmoid, optimized parameters for the given audio and config.
        """

        # Initialize parameters based on the specified method
        if config.initialization_method == ParameterInitializationMethod.INPUT:
            assert initial_params_dict is not None and initial_params_tensor is not None, "Initial parameters must be provided when using INPUT initialization method."
        else:
            initial_params_dict, initial_params_tensor = self._initialize_parameters(config=config)

        loss_fn_value = config.loss_function.value if config.loss_function else None

        if loss_fn_value is None:
            return initial_params_tensor, initial_params_dict, None

        if loss_fn_value == LossFunction.DIRECTIONAL_LOSS.value:
            return refine_with_directional_loss(
                audio=audio,
                fx_chain=config.fx_chain,
                initial_params=initial_params_tensor,
                text_anchor=config.text_anchor,
                text_target=config.text_target,
                clap_model=config.embedding,
                n_iterations=config.num_iterations,
                lr=config.learning_rate,
                device=config.device,
                snapshot_interval=config.snapshot_interval if config.save_checkpoints else None,
                optimization_method=config.optimization_method
            )

        elif loss_fn_value == LossFunction.SEMANTIC_SIMILARITY_LOSS.value:
            return fxsearcher(
                audio = (audio, 44100),
                prompt=config.prompt.instruction,
                outdir = "../results/fxsearcher",
                clap_model=config.embedding,
                top_n = 1,
                n_calls = config.num_iterations,
                use_guide=False,
                all_param_ranges=ALL_PARAM_RANGES_FXSearcher,
                initial_params=initial_params_dict
            )

        elif loss_fn_value == LossFunction.GUIDED_SEMANTIC_LOSS.value:
            return fxsearcher(
                audio = (audio, 44100),
                prompt=config.prompt.instruction,
                outdir = "../results/fxsearcher",
                clap_model=config.embedding,
                top_n = 1,
                n_calls = config.num_iterations,
                use_guide=True,
                all_param_ranges=ALL_PARAM_RANGES_FXSearcher,
                initial_params=initial_params_dict
            )

        else:
            raise ValueError(f"Unknown loss function: {config.loss_function}")

        # elif init_method_value == ParameterInitializationMethod.PRESET.value:
        #     initial_params_dict = {
        #         "EQ": {"mode": "shelf", "low_cut": 120.0, "high_cut": 12000.0, "q": 1.0, "gains": {},
        #             "peak1_freq": 200.0, "peak2_freq": 1000.0, "peak3_freq": 5000.0},
        #         "Distortion": {"drive_db": 1.0},
        #         "Reverb": {"room_size": 0.3, "damping": 0.5, "wet_level": 0.1},
        #         "Delay": {"delay": 0.1},
        #         "PitchShift": {"semitones": 0},
        #         "Bitcrush": {"bit_depth": 0},
        #     }
        #     initial_params_tensor = torch.zeros((1, 49), device=config.device) + 0.5
