from dataclasses import dataclass
import torch

from enum import Enum
from metrics.metric import Metric
from effects.fx import FXChain
from prompts.prompt import Prompt
from llms.llmclient import LLMClient
from embeddings.clap import EmbeddingWrapper

from abc import ABC, abstractmethod

class OptimizationMethod(Enum):
    GRADIENT_DESCENT = 'gradient_descent'
    BAYESIAN_OPTIMIZATION = 'bayesian_optimization'

class LossFunction(Enum):
    DIRECTIONAL_LOSS = 'directional_loss'
    SEMANTIC_SIMILARITY_LOSS = 'semantic_similarity_loss'
    GUIDED_SEMANTIC_LOSS = 'guided_semantic_loss'

class ParameterInitializationMethod(Enum):
    RANDOM = 'random'
    LLM = 'llm'
    PRESET = 'preset'
    UNIFORM = 'uniform'

@dataclass
class Config:
    # ========== General Configurations ==========
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    # ========== LLM Prompt Configurations ==========
    prompt: Prompt = None
    text_anchor: str = None
    text_target: str = None
    # ========== Initialization Configurations ==========
    initialization_method: ParameterInitializationMethod = None  # or ParameterInitializationMethod.LLM
    # ========== Optimization Configurations ==========
    optimization_method: OptimizationMethod = None
    num_iterations: int = 100
    learning_rate: float = 0.01
    # ========== Embedding Configurations ==========
    embedding: EmbeddingWrapper = None  # Placeholder for potential future models
    # ========== Metric Configurations ==========
    metrics: list[Metric] = None  # List of metric instances to compute, e.g., [CLAPSimilarity()]
    # ========== Effects ==========
    fx_chain: FXChain = None  # List of effects
    # ========== Logging Configurations ==========
    log_dir: str = './logs'
    save_checkpoints: bool = True
    checkpoint_dir: str = './checkpoints'
    snapshot_interval: int = 10  # Save intermediate results every N iterations
    # ========== Loss Function Configurations ==========
    loss_function: LossFunction = None

    llmclient: LLMClient = None

    effects_type: str = "DASP"  # or "Pedalboard"
