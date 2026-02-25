from dataclasses import dataclass
import torch

from enum import Enum
from metrics.metric import Metric
from embeddings.embeddingspace import EmbeddingSpace
from effects.fx import FXChain
from prompts.prompt import Prompt

from abc import ABC, abstractmethod

class OptimizationMethod(Enum):
    GRADIENT_DESCENT = 'gradient_descent'
    BAYESIAN_OPTIMIZATION = 'bayesian_optimization'

class LossFunction(Enum):
    DIRECTIONAL_LOSS = 'directional_loss'

class ParameterInitializationMethod(Enum):
    RANDOM = 'random'
    LLM = 'llm'
    PRESET = 'preset'

class ParameterInitialization(ABC):
    @abstractmethod
    def initialize(self, fxchain : FXChain) -> dict:
        pass

class RandomInitialization(ParameterInitialization):
    def initialize(self, fxchain : FXChain) -> dict:
        # Placeholder for random initialization logic
        return {}

class LLMInitialization(ParameterInitialization):
    def initialize(self, fxchain : FXChain) -> dict:
        # Placeholder for LLM-based initialization logic
        return {}

class PresetInitialization(ParameterInitialization):
    def initialize(self, fxchain : FXChain) -> dict:
        # Placeholder for preset-based initialization logic
        return {}

@dataclass
class Config:
    # ========== General Configurations ==========
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    # ========== LLM Prompt Configurations ==========
    prompt: Prompt = None
    # ========== Initialization Configurations ==========
    initialization_method: ParameterInitializationMethod = ParameterInitializationMethod.RANDOM  # or ParameterInitializationMethod.LLM
    # ========== Optimization Configurations ==========
    optimization_method: OptimizationMethod = OptimizationMethod.GRADIENT_DESCENT
    num_iterations: int = 100
    learning_rate: float = 0.01
    # ========== Embedding Configurations ==========
    embedding_space: EmbeddingSpace = EmbeddingSpace.CLAP  # Placeholder for potential future models
    # ========== Metric Configurations ==========
    metrics: list[Metric] = None  # List of metric instances to compute, e.g., [CLAPSimilarity()]
    # ========== Effects ==========
    fx_chain: FXChain = None  # List of effects
    # ========== Logging Configurations ==========
    log_dir: str = './logs'
    save_checkpoints: bool = True
    checkpoint_dir: str = './checkpoints'
    # ========== Loss Function Configurations ==========
    loss_function: LossFunction = LossFunction.DIRECTIONAL_LOSS