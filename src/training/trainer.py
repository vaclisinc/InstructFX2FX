from configurations.config import Config, LossFunction, OptimizationMethod
from training.loss import refine_with_directional_loss
from effects.fx import initialize_random_params, initialize_uniform_params,llm_params_dict_example, llm_params_tensor_example, llm_params_dict_example_pedalboard
from configurations.config import ParameterInitializationMethod
from utilities.fx_processing import fx_initial_params_to_tensor
from effects.fx import ALL_PARAM_RANGES
from fxsearcher.fxsearcher import fxsearcher, ALL_PARAM_RANGES_FXSearcher
from dataclasses import dataclass
import torch

@dataclass
class ParameterEngine:
    config: Config = None

    def _initialize_parameters(self) -> dict:
        if not self.config:
            raise ValueError("Config must be set to initialize parameters.")

        # Compare by value (enum.value) to avoid autoreload comparison issues
        init_method_value = self.config.initialization_method.value if self.config.initialization_method else None

        if init_method_value == ParameterInitializationMethod.RANDOM.value:
            initial_params_dict = initialize_random_params()
            initial_params_tensor = fx_initial_params_to_tensor(initial_params_dict, device=self.config.device, param_ranges=ALL_PARAM_RANGES)

        elif init_method_value == ParameterInitializationMethod.LLM.value:
            if self.config.llmclient is None:
                raise ValueError("LLM client must be provided in config for LLM-based initialization.")

            if self.config.effects_type == "Pedalboard":
                if llm_params_dict_example_pedalboard:
                    llm_params_dict = llm_params_dict_example_pedalboard
                else:
                    llm_params_dict = self.config.llmclient.generate_parameters(self.config.prompt)
                llm_params_tensor = torch.zeros((1, 49), device=self.config.device) + 0.5
                print(f"✓ LLM generated parameters for Bayesian Optimization: {llm_params_dict.keys()}")
            elif not llm_params_tensor_example or not llm_params_dict_example:
                llm_params_dict = self.config.llmclient.generate_parameters(self.config.prompt)
                llm_params_tensor = fx_initial_params_to_tensor(llm_params_dict, device=self.config.device, param_ranges=ALL_PARAM_RANGES)
                print(f"✓ LLM generated {llm_params_tensor.shape[1]} parameters")
                print(f"Sample: {llm_params_tensor[0, :5].tolist()}...")
            else:
                llm_params_tensor = fx_initial_params_to_tensor(llm_params_dict_example, device=self.config.device, param_ranges=ALL_PARAM_RANGES)
                llm_params_dict = llm_params_dict_example
            initial_params_dict = llm_params_dict
            initial_params_tensor = llm_params_tensor

        elif init_method_value == ParameterInitializationMethod.PRESET.value:
            initial_params_dict = {
                "EQ": {"mode": "shelf", "low_cut": 120.0, "high_cut": 12000.0, "q": 1.0, "gains": {},
                    "peak1_freq": 200.0, "peak2_freq": 1000.0, "peak3_freq": 5000.0},
                "Distortion": {"drive_db": 1.0},
                "Reverb": {"room_size": 0.3, "damping": 0.5, "wet_level": 0.1},
                "Delay": {"delay": 0.1},
                "PitchShift": {"semitones": 0},
                "Bitcrush": {"bit_depth": 0},
            }
            initial_params_tensor = torch.zeros((1, 49), device=self.config.device) + 0.5

        elif init_method_value == ParameterInitializationMethod.UNIFORM.value:
            initial_params_dict = initialize_uniform_params()
            initial_params_tensor = fx_initial_params_to_tensor(initial_params_dict, device=self.config.device, param_ranges=ALL_PARAM_RANGES)

        else:
            raise ValueError(f"Unknown initialization method: {self.config.initialization_method}")


        return initial_params_dict, initial_params_tensor

    def _train(self, audio):
        pass


    def get_params(self, audio, config):
        """
        Return normalized, sigmoid, optimized parameters for the given audio and config.
        """

        self.config = config

        # Initialize parameters based on the specified method
        initial_params_dict, initial_params_tensor = self._initialize_parameters()

        loss_fn_value = self.config.loss_function.value if self.config.loss_function else None

        if loss_fn_value == LossFunction.DIRECTIONAL_LOSS.value:
            return refine_with_directional_loss(
                audio=audio,
                fx_chain=self.config.fx_chain,
                initial_params=initial_params_tensor,
                text_anchor=self.config.text_anchor,
                text_target=self.config.text_target,
                clap_model=self.config.embedding,
                n_iterations=self.config.num_iterations,
                lr=self.config.learning_rate,
                device=self.config.device,
                snapshot_interval=10 if self.config.save_checkpoints else None,
                optimization_method=self.config.optimization_method
            )

        elif loss_fn_value == LossFunction.SEMANTIC_SIMILARITY_LOSS.value:
            return fxsearcher(
                audio = (audio, 44100),
                prompt=self.config.prompt.instruction,
                outdir = "../results/fxsearcher",
                clap_model=self.config.embedding,
                top_n = 1,
                n_calls = self.config.num_iterations,
                use_guide=False,
                all_param_ranges=ALL_PARAM_RANGES_FXSearcher,
                initial_params=initial_params_dict
            )

        elif loss_fn_value == LossFunction.GUIDED_SEMANTIC_LOSS.value:
            return fxsearcher(
                audio = (audio, 44100),
                prompt=self.config.prompt.instruction,
                outdir = "../results/fxsearcher",
                clap_model=self.config.embedding,
                top_n = 1,
                n_calls = self.config.num_iterations,
                use_guide=True,
                all_param_ranges=ALL_PARAM_RANGES_FXSearcher,
                initial_params=initial_params_dict
            )

        else:
            print(f"⚠️ Unknown loss function: {self.config.loss_function}. Returning initial parameters without optimization.")
            return torch.sigmoid(initial_params_tensor), None, None