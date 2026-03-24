from .metric import Metric
from typing import Any, Optional, Dict
from src.prompts.prompt import Prompt

class AIJudgeQwen(Metric):
    """Qwen2.5-omni-7B: Absolute 1-5 Alignment Rating"""
    def compute(self, target_audio: Any, prompt: Prompt):
        # Implementation: Send audio + prompt to Qwen-Omni
        # Return numeric score 1.0 - 5.0
        return 0.0

class AIJudgeGemini(Metric):
    """Gemini 2.5 Flash: Pairwise preference (Win Rate)"""
    def compute(self, audio_a: Any, audio_b: Any, prompt: Prompt):
        # Implementation: Ask Gemini which audio matches prompt better
        # Return "A", "B", or "Tie"
        return "A"