"""Prompt template loader with YAML support and caching.

This module provides functionality for loading and managing prompt templates
used in parameter generation. It supports:
- Loading prompt templates from YAML files
- Template versioning for A/B testing
- Few-shot example loading from JSON files
- Template caching for performance
"""

import json
from pathlib import Path
from typing import Optional, Dict, Any, List

import yaml
from pydantic import BaseModel, Field, ValidationError


class FewShotExample(BaseModel):
    """Few-shot example for prompt engineering.

    Attributes:
        file: Path to the JSON example file
        description: Description of what the example demonstrates
        content: Loaded JSON content (populated after loading)
    """
    file: str = Field(description="Path to example JSON file")
    description: str = Field(description="What this example demonstrates")
    content: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Loaded example content"
    )


class PromptTemplate(BaseModel):
    """Prompt template configuration loaded from YAML.

    Attributes:
        version: Template version identifier
        name: Template name
        description: Template description
        system_prompt: System prompt for the LLM
        user_prompt_template: User prompt template with placeholders
        few_shot_examples: List of few-shot examples
        notes: Additional notes about the template
    """
    version: str = Field(description="Template version (e.g., '1.0')")
    name: str = Field(description="Template name")
    description: str = Field(description="Template description")
    system_prompt: str = Field(description="System prompt for LLM")
    user_prompt_template: str = Field(
        description="User prompt template with {placeholders}"
    )
    few_shot_examples: List[FewShotExample] = Field(
        default_factory=list,
        description="List of few-shot examples"
    )
    notes: str = Field(
        default="",
        description="Additional notes about the template"
    )

    def format_user_prompt(self, **kwargs) -> str:
        """Format user prompt with provided variables.

        Args:
            **kwargs: Variables to substitute in the template

        Returns:
            Formatted user prompt string

        Raises:
            KeyError: If required placeholder is missing
        """
        try:
            return self.user_prompt_template.format(**kwargs)
        except KeyError as e:
            raise KeyError(
                f"Missing required placeholder in user prompt template: {e}"
            )

    def get_examples_as_text(self) -> str:
        """Get all few-shot examples formatted as text.

        Returns:
            Formatted string containing all examples
        """
        if not self.few_shot_examples:
            return ""

        examples_text = []
        for i, example in enumerate(self.few_shot_examples, 1):
            if example.content:
                examples_text.append(
                    f"Example {i}: {example.description}\n"
                    f"```json\n{json.dumps(example.content, indent=2)}\n```"
                )

        return "\n\n".join(examples_text)


class PromptLoader:
    """Loader for prompt templates with caching support."""

    def __init__(self, prompts_dir: Optional[Path] = None):
        """Initialize prompt loader.

        Args:
            prompts_dir: Directory containing prompt templates.
                        Defaults to 'configs/prompts/'
        """
        if prompts_dir is None:
            prompts_dir = Path("configs/prompts")

        self.prompts_dir = Path(prompts_dir)
        self._cache: Dict[str, PromptTemplate] = {}

    def load(
        self,
        version: str = "v1",
        template_name: str = "parameter_generation",
        force_reload: bool = False
    ) -> PromptTemplate:
        """Load prompt template by version and name.

        Args:
            version: Template version (e.g., 'v1', 'v2')
            template_name: Template name
            force_reload: Force reload even if cached

        Returns:
            Loaded PromptTemplate instance

        Raises:
            FileNotFoundError: If template file doesn't exist
            ValidationError: If template validation fails
            yaml.YAMLError: If YAML parsing fails
        """
        cache_key = f"{template_name}_{version}"

        # Return cached template if available
        if not force_reload and cache_key in self._cache:
            return self._cache[cache_key]

        # Construct template file path
        template_file = self.prompts_dir / f"{template_name}_{version}.yaml"

        if not template_file.exists():
            raise FileNotFoundError(
                f"Prompt template not found: {template_file}"
            )

        # Load YAML file
        with open(template_file, 'r') as f:
            data = yaml.safe_load(f)

        if data is None:
            data = {}

        # Create PromptTemplate instance
        try:
            template = PromptTemplate.model_validate(data)
        except ValidationError as e:
            raise ValidationError.from_exception_data(
                title=f"Prompt template validation failed for {template_file}",
                line_errors=e.errors()
            )

        # Load few-shot examples
        template = self._load_examples(template)

        # Cache the template
        self._cache[cache_key] = template

        return template

    def _load_examples(self, template: PromptTemplate) -> PromptTemplate:
        """Load few-shot example content from JSON files.

        Args:
            template: PromptTemplate with example file references

        Returns:
            PromptTemplate with loaded example content
        """
        for example in template.few_shot_examples:
            example_path = Path(example.file)

            # Make path absolute if relative
            if not example_path.is_absolute():
                example_path = self.prompts_dir.parent.parent / example_path

            if example_path.exists():
                try:
                    with open(example_path, 'r') as f:
                        example.content = json.load(f)
                except json.JSONDecodeError as e:
                    # Log warning but continue
                    print(
                        f"Warning: Failed to load example from {example_path}: {e}"
                    )
                    example.content = None
            else:
                print(f"Warning: Example file not found: {example_path}")
                example.content = None

        return template

    def list_versions(self, template_name: str = "parameter_generation") -> List[str]:
        """List available versions for a template.

        Args:
            template_name: Template name to search for

        Returns:
            List of available version strings
        """
        versions = []
        pattern = f"{template_name}_v*.yaml"

        for file_path in self.prompts_dir.glob(pattern):
            # Extract version from filename
            stem = file_path.stem  # e.g., "parameter_generation_v1"
            version = stem.replace(f"{template_name}_", "")
            versions.append(version)

        return sorted(versions)

    def clear_cache(self) -> None:
        """Clear the template cache."""
        self._cache.clear()

    def reload(self, version: str = "v1", template_name: str = "parameter_generation") -> PromptTemplate:
        """Force reload a template from disk.

        Args:
            version: Template version
            template_name: Template name

        Returns:
            Reloaded PromptTemplate instance
        """
        return self.load(version=version, template_name=template_name, force_reload=True)


# Global loader instance
_prompt_loader: Optional[PromptLoader] = None


def get_prompt_loader(prompts_dir: Optional[Path] = None) -> PromptLoader:
    """Get or create global prompt loader instance.

    Args:
        prompts_dir: Optional prompts directory path

    Returns:
        PromptLoader instance
    """
    global _prompt_loader
    if _prompt_loader is None:
        _prompt_loader = PromptLoader(prompts_dir)
    return _prompt_loader


def load_prompt_template(
    version: str = "v1",
    template_name: str = "parameter_generation",
    prompts_dir: Optional[Path] = None,
    force_reload: bool = False
) -> PromptTemplate:
    """Load prompt template using global loader.

    Args:
        version: Template version
        template_name: Template name
        prompts_dir: Optional prompts directory
        force_reload: Force reload from disk

    Returns:
        Loaded PromptTemplate instance
    """
    loader = get_prompt_loader(prompts_dir)
    return loader.load(version=version, template_name=template_name, force_reload=force_reload)
