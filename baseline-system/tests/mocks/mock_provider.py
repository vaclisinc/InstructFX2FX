"""Mock LLM provider for testing without API calls.

This module provides MockLLMProvider that simulates different provider behaviors
for testing purposes without incurring API costs or requiring network connectivity.
"""

import json
import time
from typing import Dict, Any
from models.llm_judge.base import LLMProvider
from models.llm_judge.types import LLMRequest, LLMResponse


class MockLLMProvider(LLMProvider):
    """Mock LLM provider for testing without API calls.

    Supports different response modes to simulate various API behaviors:
    - 'valid': Returns properly formatted JSON response
    - 'invalid_json': Returns malformed JSON to test error handling
    - 'error': Raises exception to test error recovery
    - 'timeout': Simulates timeout to test retry logic

    Attributes:
        response_mode: The behavior mode for this mock
        call_count: Number of times generate() was called
    """

    def __init__(self, response_mode: str = "valid", config: Dict[str, Any] = None):
        """Initialize mock provider.

        Args:
            response_mode: One of 'valid', 'invalid_json', 'error', 'timeout'
            config: Optional configuration dictionary
        """
        if config is None:
            config = {
                "api_key": "test-key",
                "model": "mock-model"
            }

        super().__init__(config)
        self.response_mode = response_mode
        self.call_count = 0

    def validate_config(self) -> bool:
        """Validate mock configuration.

        Returns:
            Always True for mock provider
        """
        return True

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate mock response based on response_mode.

        Args:
            request: LLM request parameters

        Returns:
            Mock LLM response

        Raises:
            Exception: If response_mode is 'error'
            TimeoutError: If response_mode is 'timeout' (after delay)
        """
        self.call_count += 1

        if self.response_mode == "error":
            raise Exception("Mock API error")

        if self.response_mode == "timeout":
            await asyncio.sleep(10)
            raise TimeoutError("Mock timeout error")

        if self.response_mode == "invalid_json":
            return LLMResponse(
                content="This is not valid JSON {malformed",
                model=request.model or "mock-model",
                tokens_used=100,
                finish_reason="stop",
                provider="mock"
            )

        # Return valid mock response with proper JSON structure
        if "score" in request.prompt.lower() or "eval" in request.prompt.lower():
            # Scoring response
            content = json.dumps({
                "overall_score": 75.0,
                "confidence": 0.8,
                "dimensions": [
                    {
                        "name": "semantic_match",
                        "score": 80.0,
                        "reasoning": "Mock reasoning for semantic match"
                    },
                    {
                        "name": "technical_quality",
                        "score": 70.0,
                        "reasoning": "Mock reasoning for technical quality"
                    },
                    {
                        "name": "specificity",
                        "score": 75.0,
                        "reasoning": "Mock reasoning for specificity"
                    }
                ],
                "feedback": "Mock feedback for improvement",
                "suggestions": [
                    "Mock suggestion 1",
                    "Mock suggestion 2"
                ]
            })
        else:
            # Parameter generation response
            content = json.dumps({
                "reverb": {
                    "delay_time": 0.03,
                    "decay": 0.5,
                    "stereo_spread": 0.0,
                    "cutoff_freq": 10000,
                    "wet_gain": 0.0,
                    "wet_dry": 0.5
                },
                "eq": {
                    "low_gain": 0.0,
                    "mid_gain": 0.0,
                    "high_gain": 0.0,
                    "low_freq": 100,
                    "mid_freq": 1000,
                    "high_freq": 8000
                }
            })

        return LLMResponse(
            content=content,
            model=request.model or "mock-model",
            tokens_used=150,
            finish_reason="stop",
            provider="mock"
        )


# Import asyncio at module level for timeout simulation
import asyncio


__all__ = ["MockLLMProvider"]
