# Testing Documentation

This document explains how to run and interpret tests for the baseline system.

## Test Structure

```
tests/
├── unit/                  # Unit tests for individual components
├── integration/           # Integration tests for full pipeline
├── performance/           # Performance and benchmarking tests
├── mocks/                 # Mock implementations for testing
│   └── mock_provider.py  # Mock LLM provider
├── fixtures/              # Test fixtures and data
│   └── conftest.py       # Pytest fixtures
└── README.md             # This file
```

## Quick Start

### Install Test Dependencies

```bash
# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install test requirements
pip install pytest pytest-cov pytest-asyncio pytest-mock numpy soundfile librosa
```

### Run All Tests

```bash
pytest
```

### Run Specific Test Categories

```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Run only performance tests
pytest -m performance

# Exclude slow tests
pytest -m "not slow"

# Exclude tests requiring API keys
pytest -m "not requires_api_key"
```

### Run Specific Test Files

```bash
# Run specific test file
pytest tests/unit/test_llm_provider.py

# Run specific test class
pytest tests/unit/test_scoring.py::TestScorer

# Run specific test function
pytest tests/unit/test_scoring.py::TestScorer::test_compute_similarity
```

### Run with Coverage

```bash
# Run with coverage report
pytest --cov=src --cov=models --cov-report=html

# Open coverage report in browser
open tests/coverage_html/index.html
```

### Run in Parallel (faster)

```bash
# Install pytest-xdist
pip install pytest-xdist

# Run tests in parallel
pytest -n auto
```

### Verbose Output

```bash
# Show detailed output
pytest -v -s

# Show local variables on failure
pytest --showlocals

# Show full traceback
pytest --tb=long
```

## Test Markers

Tests are categorized using pytest markers:

- `@pytest.mark.unit` - Unit tests (no external dependencies)
- `@pytest.mark.integration` - Integration tests (full pipeline)
- `@pytest.mark.slow` - Slow-running tests
- `@pytest.mark.performance` - Performance benchmarking tests
- `@pytest.mark.requires_api_key` - Tests needing real API keys

## Mock Providers

The `tests/mocks/mock_provider.py` provides `MockLLMProvider` for testing without API costs:

```python
from tests.mocks.mock_provider import MockLLMProvider

# Create mock provider
mock_provider = MockLLMProvider(response_mode="valid")

# Available modes:
# - "valid": Returns proper JSON response
# - "invalid_json": Returns malformed JSON
# - "error": Raises exception
# - "timeout": Simulates timeout
```

## Test Fixtures

Common test fixtures are defined in `tests/fixtures/conftest.py`:

- `test_audio_dir` - Temporary directory for test audio files
- `test_audio_path` - Generated mono test audio file (1 second)
- `test_audio_stereo_path` - Generated stereo test audio file
- `sample_parameters` - Sample effect parameters dictionary
- `sample_description` - Sample audio description string
- `sample_audio_features` - Mock audio feature values
- `sample_scoring_response` - Mock scoring response data

## Coverage Requirements

Tests must maintain **>80% code coverage** across all modules:

```bash
# Check coverage
pytest --cov=src --cov=models --cov-report=term-missing

# Fail if coverage below 80%
pytest --cov-fail-under=80
```

## Writing New Tests

### Unit Test Template

```python
import pytest
from your_module import YourClass

class TestYourClass:
    """Test suite for YourClass."""

    @pytest.fixture
    def instance(self):
        """Create test instance."""
        return YourClass()

    @pytest.mark.unit
    def test_basic_functionality(self, instance):
        """Test basic functionality."""
        result = instance.method()
        assert result is not None
```

### Integration Test Template

```python
import pytest
from tests.mocks.mock_provider import MockLLMProvider

class TestIntegration:
    """Integration test suite."""

    @pytest.fixture
    def mock_provider(self):
        """Create mock LLM provider."""
        return MockLLMProvider(response_mode="valid")

    @pytest.mark.integration
    async def test_full_pipeline(self, mock_provider, test_audio_path):
        """Test complete pipeline execution."""
        # Your integration test here
        pass
```

## Continuous Integration

Tests are configured to run in CI/CD environments. The configuration in `pytest.ini`:

- Sets minimum coverage threshold to 80%
- Marks tests appropriately (unit, integration, slow, etc.)
- Excludes tests requiring API keys by default
- Generates HTML coverage reports

## Troubleshooting

### Tests Fail with Import Errors

Ensure you're in the project root and have installed dependencies:

```bash
cd baseline-system
pip install -e .
pip install pytest pytest-cov pytest-asyncio
```

### Audio Tests Fail

Install audio processing dependencies:

```bash
pip install numpy soundfile librosa
```

### Coverage Report Not Generated

Install pytest-cov:

```bash
pip install pytest-cov
```

### Tests Timeout

Increase timeout for slow tests or mark them as slow:

```python
@pytest.mark.slow
@pytest.mark.timeout(300)  # 5 minutes
def test_long_running():
    pass
```

## Best Practices

1. **Use mocks for external dependencies** - Don't make real API calls in tests
2. **Keep tests fast** - Unit tests should run in milliseconds
3. **Make tests deterministic** - No random failures
4. **Test edge cases** - Invalid inputs, empty data, boundary values
5. **Use descriptive test names** - Clearly describe what is being tested
6. **Maintain high coverage** - Aim for >80% code coverage
7. **Clean up resources** - Use fixtures and context managers

## Performance Tests

Performance tests benchmark critical operations:

```bash
# Run performance tests
pytest -m performance -v

# Example output:
# test_single_experiment_timing ... 2.34s PASSED
# test_batch_throughput ... 5.67s PASSED
```

Performance tests have thresholds and will fail if operations take too long.

## Test Results

Test runs should complete in **<5 minutes** for the full suite:

- Unit tests: ~30 seconds
- Integration tests: ~2 minutes
- Performance tests: ~2 minutes
- Total: ~4-5 minutes

If tests take significantly longer, investigate bottlenecks or mark slow tests appropriately.