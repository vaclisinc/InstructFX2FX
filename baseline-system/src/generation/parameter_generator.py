"""Parameter generation module for translating descriptions to effect parameters.

This module provides the ParameterGenerator class that uses LLMs to translate
high-level textual descriptions into structured JSON audio effect parameters.
"""

import json
import logging
from typing import List, Optional, Dict, Any
from pathlib import Path

from pydantic import ValidationError as PydanticValidationError

from models.llm_judge import LLMProvider, LLMRequest, LLMResponse
from src.models.parameters import (
    EQParameters,
    ReverbParameters,
    CompressorParameters,
    EffectChain,
    EffectParameter
)
from src.prompts.loader import load_prompt_template, PromptTemplate
from src.prompts.templates import format_prompt, format_correction_prompt

from .exceptions import (
    ParameterGenerationError,
    JSONParseError,
    ValidationError,
    LLMProviderError,
    PromptTemplateError
)


logger = logging.getLogger(__name__)


class ParameterGenerator:
    """Generates audio effect parameters from textual descriptions using LLMs.

    This class orchestrates the parameter generation pipeline:
    1. Formats prompts using templates and few-shot examples
    2. Calls LLM provider to generate parameters
    3. Parses and validates JSON output against Pydantic schemas
    4. Handles errors with retry and correction logic

    Attributes:
        llm_provider: LLM provider instance for generation
        prompt_version: Prompt template version to use
        template: Loaded prompt template
        max_correction_attempts: Maximum correction retry attempts
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        prompt_version: str = "v1",
        prompts_dir: Optional[Path] = None,
        max_correction_attempts: int = 3
    ):
        """Initialize parameter generator.

        Args:
            llm_provider: LLM provider instance
            prompt_version: Prompt template version (default: "v1")
            prompts_dir: Optional custom prompts directory
            max_correction_attempts: Max retry attempts for corrections

        Raises:
            PromptTemplateError: If template loading fails
        """
        self.llm_provider = llm_provider
        self.prompt_version = prompt_version
        self.max_correction_attempts = max_correction_attempts

        # Load prompt template
        try:
            self.template = load_prompt_template(
                version=prompt_version,
                prompts_dir=prompts_dir
            )
            logger.info(f"Loaded prompt template version {prompt_version}")
        except Exception as e:
            raise PromptTemplateError(
                f"Failed to load prompt template version {prompt_version}",
                template_version=prompt_version,
                template_error=e
            )

    async def generate_parameters(
        self,
        description: str,
        effects: Optional[List[str]] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        include_examples: bool = True,
        num_examples: Optional[int] = None
    ) -> EffectChain:
        """Generate effect parameters from textual description.

        This is the main entry point for parameter generation. It handles
        the complete pipeline from prompt formatting to validated output.

        Args:
            description: High-level description of desired audio characteristics
            effects: List of effect types to generate (default: ["eq", "reverb", "compressor"])
            temperature: LLM sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate
            include_examples: Whether to include few-shot examples in prompt
            num_examples: Number of examples to include (None = all)

        Returns:
            EffectChain with validated effect parameters

        Raises:
            ParameterGenerationError: If generation fails after all retries
            LLMProviderError: If LLM provider fails
            ValidationError: If output cannot be validated after corrections

        Example:
            >>> generator = ParameterGenerator(llm_provider)
            >>> chain = await generator.generate_parameters(
            ...     description="warm and intimate vocal sound",
            ...     effects=["eq", "reverb"]
            ... )
            >>> print(chain.to_dict())
        """
        if effects is None:
            effects = ["eq", "reverb", "compressor"]

        # Validate effects list
        valid_effects = {"eq", "reverb", "compressor"}
        for effect in effects:
            if effect not in valid_effects:
                raise ValueError(
                    f"Invalid effect type '{effect}'. "
                    f"Must be one of: {', '.join(valid_effects)}"
                )

        logger.info(
            f"Generating parameters for description: '{description}' "
            f"with effects: {effects}"
        )

        # Format prompt
        try:
            prompt = format_prompt(
                description=description,
                effects=effects,
                template_version=self.prompt_version,
                include_examples=include_examples,
                num_examples=num_examples
            )
        except Exception as e:
            raise PromptTemplateError(
                "Failed to format prompt",
                template_version=self.prompt_version,
                template_error=e
            )

        # Generate with LLM
        try:
            response = await self._generate_with_provider(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
        except Exception as e:
            raise LLMProviderError(
                "LLM provider failed to generate response",
                provider_error=e,
                provider_name=self.llm_provider.get_provider_name(),
                request_info={
                    "description": description[:100],
                    "effects": effects,
                    "temperature": temperature
                }
            )

        # Parse and validate output
        try:
            effect_chain = self.parse_and_validate(
                json_str=response.content,
                expected_effects=effects,
                description=description
            )
            logger.info(
                f"Successfully generated {len(effect_chain.effects)} effects: "
                f"{effect_chain.order}"
            )
            return effect_chain

        except (JSONParseError, ValidationError) as e:
            # Attempt correction
            logger.warning(f"Initial generation failed: {e}. Attempting correction...")
            return await self._attempt_correction(
                original_description=description,
                invalid_output=response.content,
                error=e,
                effects=effects,
                temperature=temperature,
                max_tokens=max_tokens
            )

    async def _generate_with_provider(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        system_prompt: Optional[str] = None
    ) -> LLMResponse:
        """Generate response using LLM provider with retry logic.

        Args:
            prompt: Formatted prompt text
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            system_prompt: Optional system prompt override

        Returns:
            LLMResponse from provider

        Raises:
            LLMProviderError: If generation fails
        """
        # Use template system prompt if not provided
        if system_prompt is None:
            system_prompt = self.template.system_prompt

        request = LLMRequest(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )

        # Provider already has retry logic built-in
        response = await self.llm_provider.generate_with_retry(request)
        return response

    def parse_and_validate(
        self,
        json_str: str,
        expected_effects: Optional[List[str]] = None,
        description: Optional[str] = None
    ) -> EffectChain:
        """Parse JSON string and validate against effect parameter schemas.

        This method:
        1. Extracts JSON from LLM output (handles markdown code blocks)
        2. Parses JSON to dictionary
        3. Validates against Pydantic models
        4. Constructs EffectChain with proper effect order

        Args:
            json_str: JSON string from LLM (may include markdown)
            expected_effects: Expected effect types for validation
            description: Original description (used if missing in output)

        Returns:
            Validated EffectChain instance

        Raises:
            JSONParseError: If JSON parsing fails
            ValidationError: If schema validation fails
        """
        # Extract JSON from response (handle markdown code blocks)
        json_content = self._extract_json(json_str)

        # Parse JSON
        try:
            data = json.loads(json_content)
        except json.JSONDecodeError as e:
            raise JSONParseError(
                "Failed to parse JSON from LLM output",
                raw_output=json_content,
                parse_error=e
            )

        # Validate structure
        if not isinstance(data, dict):
            raise JSONParseError(
                f"Expected JSON object, got {type(data).__name__}",
                raw_output=json_content
            )

        # Ensure description field exists
        if "description" not in data:
            if description:
                data["description"] = description
            else:
                raise ValidationError(
                    "Missing required field 'description' in output",
                    invalid_data=data
                )

        # Ensure effects field exists
        if "effects" not in data or not isinstance(data["effects"], list):
            raise ValidationError(
                "Missing or invalid 'effects' field in output",
                invalid_data=data
            )

        # Parse each effect and build effect chain
        effects: List[EffectParameter] = []
        order: List[str] = []

        for i, effect_data in enumerate(data["effects"]):
            if not isinstance(effect_data, dict):
                raise ValidationError(
                    f"Effect at index {i} is not a valid object",
                    invalid_data=data
                )

            # Get effect type
            effect_type = effect_data.get("type")
            if not effect_type:
                raise ValidationError(
                    f"Effect at index {i} missing 'type' field",
                    invalid_data=effect_data
                )

            # Parse effect parameters
            try:
                effect = self._parse_effect(effect_type, effect_data)
                effects.append(effect)
                order.append(effect_type)
            except PydanticValidationError as e:
                raise ValidationError(
                    f"Failed to validate {effect_type} parameters",
                    validation_errors=e.errors(),
                    invalid_data=effect_data
                )

        # Validate expected effects if provided
        if expected_effects:
            missing = set(expected_effects) - set(order)
            if missing:
                logger.warning(
                    f"Generated effects missing expected types: {missing}"
                )

        # Create effect chain
        try:
            effect_chain = EffectChain(
                description=data["description"],
                effects=effects,
                order=order
            )
            return effect_chain
        except Exception as e:
            raise ValidationError(
                "Failed to create effect chain",
                validation_errors=[{"msg": str(e)}],
                invalid_data=data
            )

    def _parse_effect(self, effect_type: str, effect_data: dict) -> EffectParameter:
        """Parse individual effect from data dictionary.

        Args:
            effect_type: Type of effect ("eq", "reverb", "compressor")
            effect_data: Effect data dictionary

        Returns:
            Validated effect parameter instance

        Raises:
            ValidationError: If effect type is invalid or validation fails
        """
        # Get parameters from either top-level or nested 'parameters' field
        if "parameters" in effect_data:
            params = effect_data["parameters"]
        else:
            # Use all fields except 'type'
            params = {k: v for k, v in effect_data.items() if k != "type"}

        # Parse based on effect type
        if effect_type == "eq":
            return EQParameters(**params)
        elif effect_type == "reverb":
            return ReverbParameters(**params)
        elif effect_type == "compressor":
            return CompressorParameters(**params)
        else:
            raise ValidationError(
                f"Unknown effect type: '{effect_type}'",
                invalid_data=effect_data
            )

    def _extract_json(self, text: str) -> str:
        """Extract JSON content from LLM output.

        Handles cases where LLM wraps JSON in markdown code blocks or
        includes additional explanatory text.

        Args:
            text: Raw LLM output text

        Returns:
            Extracted JSON string
        """
        # Remove markdown code blocks if present
        if "```json" in text:
            # Extract content between ```json and ```
            start = text.find("```json") + 7
            end = text.find("```", start)
            if end != -1:
                return text[start:end].strip()

        if "```" in text:
            # Try generic code block
            start = text.find("```") + 3
            end = text.find("```", start)
            if end != -1:
                content = text[start:end].strip()
                # Remove language identifier if present
                if content.startswith("json\n"):
                    content = content[5:]
                return content

        # Try to find JSON object boundaries
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            return text[start:end].strip()

        # Return as-is if no extraction needed
        return text.strip()

    async def _attempt_correction(
        self,
        original_description: str,
        invalid_output: str,
        error: Exception,
        effects: List[str],
        temperature: float,
        max_tokens: int
    ) -> EffectChain:
        """Attempt to correct invalid LLM output.

        Sends a correction prompt to the LLM with error details and
        attempts to get valid output.

        Args:
            original_description: Original user description
            invalid_output: The invalid output from LLM
            error: The error that occurred
            effects: List of effect types
            temperature: Sampling temperature
            max_tokens: Maximum tokens

        Returns:
            Corrected and validated EffectChain

        Raises:
            ValidationError: If correction fails after max attempts
        """
        for attempt in range(1, self.max_correction_attempts + 1):
            logger.info(f"Correction attempt {attempt}/{self.max_correction_attempts}")

            # Format correction prompt
            correction_prompt = format_correction_prompt(
                original_description=original_description,
                invalid_output=invalid_output[:500],  # Truncate long outputs
                error_message=str(error),
                effects=effects,
                template_version=self.prompt_version
            )

            # Generate with LLM
            try:
                response = await self._generate_with_provider(
                    prompt=correction_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens
                )

                # Try to parse and validate
                effect_chain = self.parse_and_validate(
                    json_str=response.content,
                    expected_effects=effects,
                    description=original_description
                )

                logger.info(f"Correction successful on attempt {attempt}")
                return effect_chain

            except (JSONParseError, ValidationError) as e:
                logger.warning(f"Correction attempt {attempt} failed: {e}")
                invalid_output = response.content if 'response' in locals() else invalid_output
                error = e

                if attempt == self.max_correction_attempts:
                    raise ValidationError(
                        f"Failed to generate valid parameters after {self.max_correction_attempts} "
                        f"correction attempts",
                        validation_errors=[{"msg": str(error)}],
                        invalid_data={"last_output": invalid_output[:200]}
                    )

            except Exception as e:
                logger.error(f"Correction attempt {attempt} encountered error: {e}")
                raise LLMProviderError(
                    "LLM provider failed during correction attempt",
                    provider_error=e,
                    provider_name=self.llm_provider.get_provider_name()
                )


__all__ = [
    "ParameterGenerator",
]
