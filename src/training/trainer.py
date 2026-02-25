from configurations.config import Config, LossFunction, OptimizationMethod
from training.loss import refine_with_directional_loss


class Trainer:
    def __init__(self, config: Config):
        self.config = config

    def train(self, audio, embedding, initial_params, text_anchor, text_target):

        if self.config.loss_function == LossFunction.DIRECTIONAL_LOSS:
            return refine_with_directional_loss(
                audio=audio,
                fx_chain=self.config.fx_chain,
                initial_params=initial_params,
                text_anchor=text_anchor,
                text_target=text_target,
                clap_model=embedding,
                n_iterations=self.config.num_iterations,
                lr=self.config.learning_rate,
                device=self.config.device,
                snapshot_interval=10 if self.config.save_checkpoints else None
            )
