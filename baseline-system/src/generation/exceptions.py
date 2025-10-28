"""Custom exceptions for parameter generation module.

This module defines specialized exceptions for parameter generation errors,
providing clear error handling and debugging information.
"""


class ParameterGenerationError(Exception):
    """Base exception for parameter generation errors.

    All parameter generation exceptions inherit from this base class,
    allowing catch-all error handling when needed.

    Attributes:
        message: Human-readable error message
        details: Optional additional error details
    """

    def __init__(self, message: str, details: str = None):
        """Initialize parameter generation error.

        Args:
            message: Error message
            details: Optional additional details
        """
        self.message = message
        self.details = details
        super().__init__(self.message)

    def __str__(self) -> str:
        """Format error message with details."""
        if self.details:
            return f"{self.message}\nDetails: {self.details}"
        return self.message


class JSONParseError(ParameterGenerationError):
    """Exception raised when LLM output cannot be parsed as JSON.

    This occurs when the LLM generates malformed JSON or includes
    non-JSON text in the response.

    Attributes:
        raw_output: The raw LLM output that failed to parse
        parse_error: The original JSON parsing exception
    """

    def __init__(self, message: str, raw_output: str = None, parse_error: Exception = None):
        """Initialize JSON parse error.

        Args:
            message: Error message
            raw_output: The raw LLM output
            parse_error: Original parsing exception
        """
        self.raw_output = raw_output
        self.parse_error = parse_error

        details = []
        if raw_output:
            # Truncate long outputs for readability
            truncated = raw_output[:200] + "..." if len(raw_output) > 200 else raw_output
            details.append(f"Raw output: {truncated}")
        if parse_error:
            details.append(f"Parse error: {str(parse_error)}")

        super().__init__(message, "\n".join(details) if details else None)


class ValidationError(ParameterGenerationError):
    """Exception raised when generated parameters fail schema validation.

    This occurs when the LLM generates JSON that doesn't match the expected
    schema or has parameter values outside valid ranges.

    Attributes:
        validation_errors: List of Pydantic validation errors
        invalid_data: The data that failed validation
    """

    def __init__(self, message: str, validation_errors: list = None, invalid_data: dict = None):
        """Initialize validation error.

        Args:
            message: Error message
            validation_errors: Pydantic validation errors
            invalid_data: Data that failed validation
        """
        self.validation_errors = validation_errors or []
        self.invalid_data = invalid_data

        details = []
        if validation_errors:
            error_msgs = []
            for error in validation_errors:
                loc = ".".join(str(x) for x in error.get("loc", []))
                msg = error.get("msg", "Unknown error")
                error_msgs.append(f"  - {loc}: {msg}")
            details.append("Validation errors:\n" + "\n".join(error_msgs))

        if invalid_data:
            import json
            data_str = json.dumps(invalid_data, indent=2)
            truncated = data_str[:300] + "..." if len(data_str) > 300 else data_str
            details.append(f"Invalid data:\n{truncated}")

        super().__init__(message, "\n".join(details) if details else None)


class LLMProviderError(ParameterGenerationError):
    """Exception raised when LLM provider fails to generate response.

    This wraps underlying LLM provider errors (API failures, timeouts, etc.)
    with additional context about the parameter generation attempt.

    Attributes:
        provider_error: The original provider exception
        provider_name: Name of the LLM provider
        request_info: Information about the failed request
    """

    def __init__(
        self,
        message: str,
        provider_error: Exception = None,
        provider_name: str = None,
        request_info: dict = None
    ):
        """Initialize LLM provider error.

        Args:
            message: Error message
            provider_error: Original provider exception
            provider_name: Name of the provider
            request_info: Request information
        """
        self.provider_error = provider_error
        self.provider_name = provider_name
        self.request_info = request_info

        details = []
        if provider_name:
            details.append(f"Provider: {provider_name}")
        if provider_error:
            details.append(f"Error: {str(provider_error)}")
        if request_info:
            info_str = ", ".join(f"{k}={v}" for k, v in request_info.items())
            details.append(f"Request: {info_str}")

        super().__init__(message, "\n".join(details) if details else None)


class PromptTemplateError(ParameterGenerationError):
    """Exception raised when prompt template loading or formatting fails.

    This occurs when the prompt template file is missing, invalid, or
    cannot be properly formatted with the provided variables.

    Attributes:
        template_version: The prompt template version that failed
        template_error: The original template exception
    """

    def __init__(
        self,
        message: str,
        template_version: str = None,
        template_error: Exception = None
    ):
        """Initialize prompt template error.

        Args:
            message: Error message
            template_version: Template version
            template_error: Original template exception
        """
        self.template_version = template_version
        self.template_error = template_error

        details = []
        if template_version:
            details.append(f"Template version: {template_version}")
        if template_error:
            details.append(f"Error: {str(template_error)}")

        super().__init__(message, "\n".join(details) if details else None)


__all__ = [
    "ParameterGenerationError",
    "JSONParseError",
    "ValidationError",
    "LLMProviderError",
    "PromptTemplateError",
]
