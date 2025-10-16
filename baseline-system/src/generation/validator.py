"""Validation utilities for parameter generation.

This module provides pre-validation and post-validation utilities to ensure
generated parameters meet requirements before and after LLM processing.
"""

import logging
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum

from pydantic import ValidationError as PydanticValidationError

from src.models.parameters import (
    EQParameters,
    ReverbParameters,
    CompressorParameters,
    EffectChain,
    EffectParameter
)


logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation severity levels."""
    ERROR = "error"      # Must be fixed
    WARNING = "warning"  # Should be fixed but not critical
    INFO = "info"        # Informational only


@dataclass
class ValidationIssue:
    """Represents a validation issue.

    Attributes:
        level: Severity level
        field: Field name that has the issue
        message: Human-readable description
        current_value: Current value (if applicable)
        expected_value: Expected value or range (if applicable)
    """
    level: ValidationLevel
    field: str
    message: str
    current_value: Any = None
    expected_value: Any = None

    def __str__(self) -> str:
        """Format validation issue as string."""
        parts = [f"[{self.level.value.upper()}]", f"{self.field}:", self.message]
        if self.current_value is not None:
            parts.append(f"(current: {self.current_value})")
        if self.expected_value is not None:
            parts.append(f"(expected: {self.expected_value})")
        return " ".join(parts)


@dataclass
class ValidationResult:
    """Result of validation check.

    Attributes:
        is_valid: Whether validation passed (no errors)
        issues: List of validation issues found
    """
    is_valid: bool
    issues: List[ValidationIssue]

    def __bool__(self) -> bool:
        """Allow using result in boolean context."""
        return self.is_valid

    def has_errors(self) -> bool:
        """Check if result has any error-level issues."""
        return any(issue.level == ValidationLevel.ERROR for issue in self.issues)

    def has_warnings(self) -> bool:
        """Check if result has any warning-level issues."""
        return any(issue.level == ValidationLevel.WARNING for issue in self.issues)

    def get_errors(self) -> List[ValidationIssue]:
        """Get only error-level issues."""
        return [issue for issue in self.issues if issue.level == ValidationLevel.ERROR]

    def get_warnings(self) -> List[ValidationIssue]:
        """Get only warning-level issues."""
        return [issue for issue in self.issues if issue.level == ValidationLevel.WARNING]

    def format_report(self) -> str:
        """Format validation result as detailed report."""
        if self.is_valid:
            return "✓ Validation passed"

        lines = ["Validation failed:"]

        errors = self.get_errors()
        if errors:
            lines.append(f"\nErrors ({len(errors)}):")
            for issue in errors:
                lines.append(f"  {issue}")

        warnings = self.get_warnings()
        if warnings:
            lines.append(f"\nWarnings ({len(warnings)}):")
            for issue in warnings:
                lines.append(f"  {issue}")

        return "\n".join(lines)


# Parameter ranges for pre-validation
PARAMETER_RANGES = {
    "eq": {
        "frequency": (20.0, 20000.0),
        "gain": (-12.0, 12.0),
        "q": (0.1, 10.0),
        "bands_count": (3, 10)
    },
    "reverb": {
        "room_size": (0.0, 1.0),
        "damping": (0.0, 1.0),
        "wet_level": (0.0, 1.0),
        "dry_level": (0.0, 1.0),
        "width": (0.0, 1.0),
    },
    "compressor": {
        "threshold": (-60.0, 0.0),
        "ratio": (1.0, 20.0),
        "attack": (0.1, 100.0),
        "release": (10.0, 1000.0),
        "knee": (0.0, 12.0),
        "makeup_gain": (0.0, 24.0),
    }
}


def validate_effect_structure(effect_data: Dict[str, Any]) -> ValidationResult:
    """Pre-validate effect data structure before Pydantic parsing.

    Checks for:
    - Required fields present
    - Field types are reasonable
    - Values are within expected ranges

    Args:
        effect_data: Effect data dictionary

    Returns:
        ValidationResult with any issues found
    """
    issues = []

    # Check for effect type
    if "type" not in effect_data:
        issues.append(ValidationIssue(
            level=ValidationLevel.ERROR,
            field="type",
            message="Missing required 'type' field"
        ))
        return ValidationResult(is_valid=False, issues=issues)

    effect_type = effect_data["type"]
    if effect_type not in PARAMETER_RANGES:
        issues.append(ValidationIssue(
            level=ValidationLevel.ERROR,
            field="type",
            message=f"Invalid effect type: {effect_type}",
            current_value=effect_type,
            expected_value=list(PARAMETER_RANGES.keys())
        ))
        return ValidationResult(is_valid=False, issues=issues)

    ranges = PARAMETER_RANGES[effect_type]

    # Validate based on effect type
    if effect_type == "eq":
        issues.extend(_validate_eq_structure(effect_data, ranges))
    elif effect_type == "reverb":
        issues.extend(_validate_reverb_structure(effect_data, ranges))
    elif effect_type == "compressor":
        issues.extend(_validate_compressor_structure(effect_data, ranges))

    has_errors = any(issue.level == ValidationLevel.ERROR for issue in issues)
    return ValidationResult(is_valid=not has_errors, issues=issues)


def _validate_eq_structure(data: Dict[str, Any], ranges: Dict[str, tuple]) -> List[ValidationIssue]:
    """Validate EQ effect structure."""
    issues = []

    # Check for bands
    if "bands" not in data:
        issues.append(ValidationIssue(
            level=ValidationLevel.ERROR,
            field="bands",
            message="Missing required 'bands' field"
        ))
        return issues

    bands = data["bands"]
    if not isinstance(bands, list):
        issues.append(ValidationIssue(
            level=ValidationLevel.ERROR,
            field="bands",
            message="Bands must be a list",
            current_value=type(bands).__name__
        ))
        return issues

    # Check band count
    min_bands, max_bands = ranges["bands_count"]
    if len(bands) < min_bands:
        issues.append(ValidationIssue(
            level=ValidationLevel.ERROR,
            field="bands",
            message=f"Too few bands (minimum {min_bands})",
            current_value=len(bands),
            expected_value=f"{min_bands}-{max_bands}"
        ))
    elif len(bands) > max_bands:
        issues.append(ValidationIssue(
            level=ValidationLevel.ERROR,
            field="bands",
            message=f"Too many bands (maximum {max_bands})",
            current_value=len(bands),
            expected_value=f"{min_bands}-{max_bands}"
        ))

    # Validate each band
    for i, band in enumerate(bands):
        if not isinstance(band, dict):
            issues.append(ValidationIssue(
                level=ValidationLevel.ERROR,
                field=f"bands[{i}]",
                message="Band must be an object"
            ))
            continue

        # Check required fields
        for field in ["frequency", "gain", "q"]:
            if field not in band:
                issues.append(ValidationIssue(
                    level=ValidationLevel.ERROR,
                    field=f"bands[{i}].{field}",
                    message=f"Missing required field '{field}'"
                ))
                continue

            value = band[field]
            if not isinstance(value, (int, float)):
                issues.append(ValidationIssue(
                    level=ValidationLevel.ERROR,
                    field=f"bands[{i}].{field}",
                    message=f"Must be a number",
                    current_value=type(value).__name__
                ))
                continue

            # Check range
            min_val, max_val = ranges[field]
            if value < min_val or value > max_val:
                issues.append(ValidationIssue(
                    level=ValidationLevel.ERROR,
                    field=f"bands[{i}].{field}",
                    message=f"Value out of range",
                    current_value=value,
                    expected_value=f"{min_val}-{max_val}"
                ))

    return issues


def _validate_reverb_structure(data: Dict[str, Any], ranges: Dict[str, tuple]) -> List[ValidationIssue]:
    """Validate reverb effect structure."""
    issues = []

    required_fields = ["room_size", "damping", "wet_level", "dry_level", "width"]

    for field in required_fields:
        if field not in data:
            issues.append(ValidationIssue(
                level=ValidationLevel.ERROR,
                field=field,
                message=f"Missing required field '{field}'"
            ))
            continue

        value = data[field]
        if not isinstance(value, (int, float)):
            issues.append(ValidationIssue(
                level=ValidationLevel.ERROR,
                field=field,
                message="Must be a number",
                current_value=type(value).__name__
            ))
            continue

        # Check range
        min_val, max_val = ranges[field]
        if value < min_val or value > max_val:
            issues.append(ValidationIssue(
                level=ValidationLevel.ERROR,
                field=field,
                message="Value out of range",
                current_value=value,
                expected_value=f"{min_val}-{max_val}"
            ))

    # Check wet/dry balance
    if "wet_level" in data and "dry_level" in data:
        wet = data["wet_level"]
        dry = data["dry_level"]
        if isinstance(wet, (int, float)) and isinstance(dry, (int, float)):
            total = wet + dry
            if total > 2.0:
                issues.append(ValidationIssue(
                    level=ValidationLevel.WARNING,
                    field="wet_level+dry_level",
                    message="Combined wet/dry levels unusually high",
                    current_value=total,
                    expected_value="≤2.0"
                ))

    return issues


def _validate_compressor_structure(data: Dict[str, Any], ranges: Dict[str, tuple]) -> List[ValidationIssue]:
    """Validate compressor effect structure."""
    issues = []

    required_fields = ["threshold", "ratio", "attack", "release", "knee", "makeup_gain"]

    for field in required_fields:
        if field not in data:
            issues.append(ValidationIssue(
                level=ValidationLevel.ERROR,
                field=field,
                message=f"Missing required field '{field}'"
            ))
            continue

        value = data[field]
        if not isinstance(value, (int, float)):
            issues.append(ValidationIssue(
                level=ValidationLevel.ERROR,
                field=field,
                message="Must be a number",
                current_value=type(value).__name__
            ))
            continue

        # Check range
        min_val, max_val = ranges[field]
        if value < min_val or value > max_val:
            issues.append(ValidationIssue(
                level=ValidationLevel.ERROR,
                field=field,
                message="Value out of range",
                current_value=value,
                expected_value=f"{min_val}-{max_val}"
            ))

    # Check attack/release relationship
    if "attack" in data and "release" in data:
        attack = data["attack"]
        release = data["release"]
        if isinstance(attack, (int, float)) and isinstance(release, (int, float)):
            if attack >= release:
                issues.append(ValidationIssue(
                    level=ValidationLevel.WARNING,
                    field="attack/release",
                    message="Attack should be shorter than release for natural compression",
                    current_value=f"attack={attack}, release={release}"
                ))

    return issues


def validate_effect_chain_structure(chain_data: Dict[str, Any]) -> ValidationResult:
    """Pre-validate effect chain structure.

    Args:
        chain_data: Effect chain data dictionary

    Returns:
        ValidationResult with any issues found
    """
    issues = []

    # Check for description
    if "description" not in chain_data:
        issues.append(ValidationIssue(
            level=ValidationLevel.WARNING,
            field="description",
            message="Missing description field"
        ))

    # Check for effects
    if "effects" not in chain_data:
        issues.append(ValidationIssue(
            level=ValidationLevel.ERROR,
            field="effects",
            message="Missing required 'effects' field"
        ))
        return ValidationResult(is_valid=False, issues=issues)

    effects = chain_data["effects"]
    if not isinstance(effects, list):
        issues.append(ValidationIssue(
            level=ValidationLevel.ERROR,
            field="effects",
            message="Effects must be a list",
            current_value=type(effects).__name__
        ))
        return ValidationResult(is_valid=False, issues=issues)

    if len(effects) == 0:
        issues.append(ValidationIssue(
            level=ValidationLevel.ERROR,
            field="effects",
            message="Effects list cannot be empty"
        ))
        return ValidationResult(is_valid=False, issues=issues)

    # Validate each effect
    for i, effect_data in enumerate(effects):
        if not isinstance(effect_data, dict):
            issues.append(ValidationIssue(
                level=ValidationLevel.ERROR,
                field=f"effects[{i}]",
                message="Effect must be an object"
            ))
            continue

        effect_result = validate_effect_structure(effect_data)
        # Prepend effect index to field names
        for issue in effect_result.issues:
            issue.field = f"effects[{i}].{issue.field}"
            issues.append(issue)

    has_errors = any(issue.level == ValidationLevel.ERROR for issue in issues)
    return ValidationResult(is_valid=not has_errors, issues=issues)


def validate_effect_parameter(effect: EffectParameter) -> ValidationResult:
    """Post-validate an effect parameter using Pydantic model validation.

    This validates that a Pydantic model instance is valid. Useful for
    checking parameters after normalization.

    Args:
        effect: Effect parameter instance

    Returns:
        ValidationResult
    """
    issues = []

    try:
        # Try to re-validate by dumping and re-parsing
        data = effect.model_dump()
        effect_type = data.get("effect_type")

        if effect_type == "eq":
            EQParameters(**data)
        elif effect_type == "reverb":
            ReverbParameters(**data)
        elif effect_type == "compressor":
            CompressorParameters(**data)
        else:
            issues.append(ValidationIssue(
                level=ValidationLevel.ERROR,
                field="effect_type",
                message=f"Unknown effect type: {effect_type}"
            ))
            return ValidationResult(is_valid=False, issues=issues)

        return ValidationResult(is_valid=True, issues=[])

    except PydanticValidationError as e:
        for error in e.errors():
            loc = ".".join(str(x) for x in error["loc"])
            issues.append(ValidationIssue(
                level=ValidationLevel.ERROR,
                field=loc,
                message=error["msg"],
                current_value=error.get("input")
            ))
        return ValidationResult(is_valid=False, issues=issues)
    except Exception as e:
        issues.append(ValidationIssue(
            level=ValidationLevel.ERROR,
            field="validation",
            message=f"Validation error: {str(e)}"
        ))
        return ValidationResult(is_valid=False, issues=issues)


def validate_effect_chain(chain: EffectChain) -> ValidationResult:
    """Post-validate an effect chain.

    Args:
        chain: EffectChain instance

    Returns:
        ValidationResult
    """
    issues = []

    # Validate the chain itself
    try:
        data = chain.model_dump()
        EffectChain(**data)
    except PydanticValidationError as e:
        for error in e.errors():
            loc = ".".join(str(x) for x in error["loc"])
            issues.append(ValidationIssue(
                level=ValidationLevel.ERROR,
                field=loc,
                message=error["msg"],
                current_value=error.get("input")
            ))
        return ValidationResult(is_valid=False, issues=issues)

    # Validate each effect
    for i, effect in enumerate(chain.effects):
        result = validate_effect_parameter(effect)
        for issue in result.issues:
            issue.field = f"effects[{i}].{issue.field}"
            issues.append(issue)

    has_errors = any(issue.level == ValidationLevel.ERROR for issue in issues)
    return ValidationResult(is_valid=not has_errors, issues=issues)


__all__ = [
    "ValidationLevel",
    "ValidationIssue",
    "ValidationResult",
    "PARAMETER_RANGES",
    "validate_effect_structure",
    "validate_effect_chain_structure",
    "validate_effect_parameter",
    "validate_effect_chain",
]
