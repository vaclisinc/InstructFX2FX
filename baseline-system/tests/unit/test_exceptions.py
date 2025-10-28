"""Tests for LLM provider exception classes."""

import pytest
from models.llm_judge.utils.exceptions import (
    LLMProviderError,
    ConfigurationError,
    AuthenticationError,
    RateLimitError,
    APIError,
    TimeoutError,
    NetworkError,
    InvalidRequestError,
    ResponseParsingError,
    MaxRetriesExceededError,
    is_retryable_error,
)


class TestExceptionHierarchy:
    """Test exception class hierarchy."""

    def test_all_exceptions_inherit_from_base(self):
        """All custom exceptions should inherit from LLMProviderError."""
        exception_classes = [
            ConfigurationError,
            AuthenticationError,
            RateLimitError,
            APIError,
            TimeoutError,
            NetworkError,
            InvalidRequestError,
            ResponseParsingError,
            MaxRetriesExceededError,
        ]

        for exc_class in exception_classes:
            assert issubclass(exc_class, LLMProviderError)
            assert issubclass(exc_class, Exception)

    def test_base_exception_can_be_raised(self):
        """LLMProviderError can be raised and caught."""
        with pytest.raises(LLMProviderError):
            raise LLMProviderError("Test error")


class TestRateLimitError:
    """Test RateLimitError with retry_after attribute."""

    def test_rate_limit_error_basic(self):
        """RateLimitError can be created with just a message."""
        error = RateLimitError("Rate limit exceeded")
        assert str(error) == "Rate limit exceeded"
        assert error.retry_after is None

    def test_rate_limit_error_with_retry_after(self):
        """RateLimitError stores retry_after value."""
        error = RateLimitError("Rate limit exceeded", retry_after=60.0)
        assert error.retry_after == 60.0

    def test_rate_limit_error_is_retryable(self):
        """RateLimitError should be classified as retryable."""
        error = RateLimitError("Rate limit exceeded")
        assert is_retryable_error(error) is True


class TestAPIError:
    """Test APIError with status code and response body."""

    def test_api_error_basic(self):
        """APIError can be created with just a message."""
        error = APIError("API error occurred")
        assert str(error) == "API error occurred"
        assert error.status_code is None
        assert error.response_body is None

    def test_api_error_with_status_code(self):
        """APIError stores status code."""
        error = APIError("Server error", status_code=500)
        assert error.status_code == 500

    def test_api_error_with_response_body(self):
        """APIError stores response body."""
        body = '{"error": "Internal server error"}'
        error = APIError("Server error", status_code=500, response_body=body)
        assert error.response_body == body

    def test_api_error_is_retryable_5xx(self):
        """APIError with 5xx status should be retryable."""
        error = APIError("Server error", status_code=500)
        assert error.is_retryable() is True

        error = APIError("Bad gateway", status_code=502)
        assert error.is_retryable() is True

        error = APIError("Service unavailable", status_code=503)
        assert error.is_retryable() is True

    def test_api_error_is_retryable_429(self):
        """APIError with 429 status should be retryable."""
        error = APIError("Too many requests", status_code=429)
        assert error.is_retryable() is True

    def test_api_error_not_retryable_4xx(self):
        """APIError with 4xx status (except 429) should not be retryable."""
        error = APIError("Bad request", status_code=400)
        assert error.is_retryable() is False

        error = APIError("Unauthorized", status_code=401)
        assert error.is_retryable() is False

        error = APIError("Forbidden", status_code=403)
        assert error.is_retryable() is False

    def test_api_error_not_retryable_no_status(self):
        """APIError without status code should not be retryable."""
        error = APIError("Unknown error")
        assert error.is_retryable() is False


class TestTimeoutError:
    """Test TimeoutError with timeout duration."""

    def test_timeout_error_basic(self):
        """TimeoutError can be created with just a message."""
        error = TimeoutError("Request timed out")
        assert str(error) == "Request timed out"
        assert error.timeout_seconds is None

    def test_timeout_error_with_duration(self):
        """TimeoutError stores timeout duration."""
        error = TimeoutError("Request timed out", timeout_seconds=30.0)
        assert error.timeout_seconds == 30.0

    def test_timeout_error_is_retryable(self):
        """TimeoutError should be classified as retryable."""
        error = TimeoutError("Request timed out")
        assert is_retryable_error(error) is True


class TestResponseParsingError:
    """Test ResponseParsingError with raw response."""

    def test_response_parsing_error_basic(self):
        """ResponseParsingError can be created with just a message."""
        error = ResponseParsingError("Could not parse response")
        assert str(error) == "Could not parse response"
        assert error.raw_response is None

    def test_response_parsing_error_with_raw_response(self):
        """ResponseParsingError stores raw response."""
        raw = "invalid json {"
        error = ResponseParsingError("Invalid JSON", raw_response=raw)
        assert error.raw_response == raw

    def test_response_parsing_error_is_retryable(self):
        """ResponseParsingError should be classified as retryable."""
        error = ResponseParsingError("Parse error")
        assert is_retryable_error(error) is True


class TestMaxRetriesExceededError:
    """Test MaxRetriesExceededError with attempts and last error."""

    def test_max_retries_error_basic(self):
        """MaxRetriesExceededError requires message and attempts."""
        error = MaxRetriesExceededError("All retries failed", attempts=3)
        assert str(error) == "All retries failed"
        assert error.attempts == 3
        assert error.last_error is None

    def test_max_retries_error_with_last_error(self):
        """MaxRetriesExceededError stores the last error."""
        cause = NetworkError("Connection refused")
        error = MaxRetriesExceededError(
            "All retries failed",
            attempts=3,
            last_error=cause
        )
        assert error.last_error is cause


class TestIsRetryableError:
    """Test is_retryable_error() utility function."""

    def test_retryable_errors(self):
        """Verify retryable error types."""
        retryable = [
            RateLimitError("Rate limit"),
            TimeoutError("Timeout"),
            NetworkError("Network error"),
            ResponseParsingError("Parse error"),
            APIError("Server error", status_code=500),
            APIError("Too many requests", status_code=429),
        ]

        for error in retryable:
            assert is_retryable_error(error) is True, f"{type(error).__name__} should be retryable"

    def test_non_retryable_errors(self):
        """Verify non-retryable error types."""
        non_retryable = [
            AuthenticationError("Invalid API key"),
            ConfigurationError("Missing config"),
            InvalidRequestError("Bad request"),
            APIError("Bad request", status_code=400),
            APIError("Unauthorized", status_code=401),
            APIError("Forbidden", status_code=403),
        ]

        for error in non_retryable:
            assert is_retryable_error(error) is False, f"{type(error).__name__} should not be retryable"

    def test_unknown_error_not_retryable(self):
        """Unknown errors should not be retried by default."""
        error = ValueError("Some random error")
        assert is_retryable_error(error) is False

    def test_base_llm_error_not_retryable(self):
        """Base LLMProviderError should not be retried by default."""
        error = LLMProviderError("Generic error")
        assert is_retryable_error(error) is False


class TestExceptionMessages:
    """Test exception messages are preserved."""

    def test_exception_messages(self):
        """All exceptions should preserve their messages."""
        message = "This is a test error message"

        exceptions = [
            LLMProviderError(message),
            ConfigurationError(message),
            AuthenticationError(message),
            RateLimitError(message),
            APIError(message),
            TimeoutError(message),
            NetworkError(message),
            InvalidRequestError(message),
            ResponseParsingError(message),
            MaxRetriesExceededError(message, attempts=3),
        ]

        for error in exceptions:
            assert str(error) == message


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
