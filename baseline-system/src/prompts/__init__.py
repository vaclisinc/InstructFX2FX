"""Prompt engineering module for parameter generation.

This module provides prompt templates, few-shot examples, and utilities
for generating audio effect parameters from natural language descriptions.
"""

from .loader import PromptLoader, load_prompt_template
from .templates import PromptTemplate, format_prompt

__all__ = [
    "PromptLoader",
    "load_prompt_template",
    "PromptTemplate",
    "format_prompt",
]
