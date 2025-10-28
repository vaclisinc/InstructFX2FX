# Integration Tests

This directory contains integration tests that make real API calls to external services.

## Overview

Integration tests verify the system works correctly with real external APIs, including:
- OpenRouter API
- Anthropic Claude API
- Real error handling and retry logic
- Actual token usage and billing

## Requirements

### API Keys

Integration tests require valid API keys set as environment variables:

```bash
# For OpenRouter tests
export OPENROUTER_API_KEY="sk-or-v1-..."

# For Claude tests
export ANTHROPIC_API_KEY="sk-ant-..."
```

**IMPORTANT**: These tests will use real API credits and incur costs. We use lightweight/free models to minimize expenses, but be aware of API usage.

### Setup `.env` File

Create a `.env` file in the project root (not committed to git):

```bash
# .env
OPENROUTER_API_KEY=sk-or-v1-your-key-here
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

The tests will automatically load environment variables from `.env` using `python-dotenv`.

## Running Integration Tests

### Run All Integration Tests

```bash
# Run all integration tests (requires API keys)
pytest tests/integration/ -v -m integration
```

### Run Specific Provider Tests

```bash
# OpenRouter integration tests only
pytest tests/integration/test_openrouter_integration.py -v

# Claude integration tests only
pytest tests/integration/test_claude_integration.py -v
```

### Skip Integration Tests

```bash
# Run all tests EXCEPT integration tests
pytest tests/ -v -m "not integration"

# This is useful for CI/CD or when you don't have API keys
```

### Run Without API Keys

If you don't have API keys set, integration tests will be **automatically skipped**:

```bash
pytest tests/integration/ -v
# Output: 16 skipped in 0.17s
```

## Test Categories

### OpenRouter Integration Tests (`test_openrouter_integration.py`)

Comprehensive tests covering:

1. **Real API Calls**
   - Basic text generation
   - System prompt handling
   - Model selection
   - Token usage tracking

2. **Retry Logic**
   - Rate limit error retry (429)
   - Server error retry (5xx)
   - Max retry limit enforcement
   - Exponential backoff timing

3. **Error Handling**
   - Invalid API key
   - Invalid model name
   - Network errors and timeouts
   - Clear error messages

4. **Advanced Features**
   - OpenRouter-specific headers
   - Temperature variation
   - Max tokens enforcement
   - Model override per request
   - Cumulative token tracking

### Claude Integration Tests (`test_claude_integration.py`)

Similar test coverage for Claude/Anthropic API.

## Test Design Principles

### Cost Minimization

- Use **free models** when possible (e.g., `meta-llama/llama-3.2-3b-instruct:free`)
- Use **small prompts** and low `max_tokens` limits
- **Skip gracefully** when no API key is available
- Mark tests clearly with `@pytest.mark.integration`

### Real Behavior Verification

- **No mocks** - these tests verify real API behavior
- Test actual retry logic with real failures (using mocked client internally)
- Verify actual token counts from API responses
- Test real error codes and messages

### Clear Documentation

Every test includes:
- Descriptive docstring explaining what is verified
- Print statements showing test results
- Clear assertions with meaningful error messages

## Example Test Output

```bash
$ pytest tests/integration/test_openrouter_integration.py::TestOpenRouterRealAPI::test_basic_generation_with_real_api -v -s

tests/integration/test_openrouter_integration.py::TestOpenRouterRealAPI::test_basic_generation_with_real_api
✓ Basic generation test passed
  Model: meta-llama/llama-3.2-3b-instruct:free
  Tokens used: 25
  Response: Hello
PASSED
```

## Writing New Integration Tests

When adding new integration tests:

1. **Mark as integration**:
   ```python
   @pytest.mark.integration
   class TestMyIntegration:
       ...
   ```

2. **Skip if no API key**:
   ```python
   @pytest.mark.skipif(
       not os.getenv("API_KEY_NAME"),
       reason="Requires API_KEY_NAME environment variable"
   )
   ```

3. **Use free/cheap models**:
   ```python
   config = {
       "provider": "openrouter",
       "model": "meta-llama/llama-3.2-3b-instruct:free"
   }
   ```

4. **Keep prompts small**:
   ```python
   request = LLMRequest(
       prompt="Short test prompt",
       max_tokens=10  # Keep small to minimize cost
   )
   ```

5. **Document what you're testing**:
   ```python
   async def test_feature(self):
       """Test feature X works correctly.

       This test verifies:
       - Behavior A
       - Behavior B
       - Error handling for C
       """
       ...
   ```

## Troubleshooting

### Tests Skipped

**Problem**: All integration tests are skipped

**Solution**: Set the required API key environment variable:
```bash
export OPENROUTER_API_KEY="sk-or-v1-..."
pytest tests/integration/test_openrouter_integration.py -v
```

### Authentication Errors

**Problem**: Tests fail with authentication errors

**Solution**: Verify your API key is valid and has credits:
```bash
# Check your API key is set
echo $OPENROUTER_API_KEY

# Verify it starts with the correct prefix
# OpenRouter: sk-or-v1-...
# Anthropic: sk-ant-...
```

### Rate Limit Errors

**Problem**: Tests fail with rate limit errors

**Solution**:
- Wait a few minutes and retry
- Use a different API key with higher rate limits
- Run fewer tests at once

### Network Timeouts

**Problem**: Tests timeout connecting to API

**Solution**:
- Check internet connection
- Increase timeout in provider config
- Verify API endpoint is not blocked by firewall

## CI/CD Integration

For continuous integration, integration tests should be:

1. **Optional**: Run only when API keys are available
2. **Skipped by default**: Use `-m "not integration"` flag
3. **Separate job**: Run in dedicated CI job with secrets

Example GitHub Actions:

```yaml
# .github/workflows/test.yml
- name: Run unit tests
  run: pytest tests/ -m "not integration" -v

- name: Run integration tests
  if: ${{ secrets.OPENROUTER_API_KEY != '' }}
  env:
    OPENROUTER_API_KEY: ${{ secrets.OPENROUTER_API_KEY }}
  run: pytest tests/integration/ -v -m integration
```

## Cost Monitoring

**Estimated costs** (as of 2024):
- Free models: $0.00 per test
- GPT-3.5-turbo: ~$0.0001 per test
- Full test suite: ~$0.01-0.05 per run

Monitor your API usage regularly and set up billing alerts.

## Support

For issues with integration tests:
1. Check API key is valid
2. Verify API service is operational
3. Review test output for specific error messages
4. Check API provider status page
