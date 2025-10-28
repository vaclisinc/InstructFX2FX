# Troubleshooting Guide

Common issues and solutions for the baseline system.

## Setup Issues

### API Key Errors

**Error:** `AuthenticationError: Invalid API key`

**Solution:**
1. Check `.env` file exists in project root:
   ```bash
   ls -la .env
   ```

2. Verify API key format (no extra spaces/newlines):
   ```bash
   cat .env
   # Should show:
   # ANTHROPIC_API_KEY=sk-ant-...
   # OPENROUTER_API_KEY=sk-or-...
   ```

3. Reload environment:
   ```bash
   source .env
   # Or restart terminal
   ```

4. Test API key:
   ```bash
   python -c "import os; from dotenv import load_dotenv; load_dotenv(); print(os.getenv('ANTHROPIC_API_KEY')[:20])"
   ```

### Import Errors

**Error:** `ModuleNotFoundError: No module named 'src'`

**Solution:**
```bash
# Ensure you're in the baseline-system directory
cd baseline-system

# Activate virtual environment
source venv/bin/activate

# Install in development mode
pip install -e .

# Or add to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

### Audio Library Errors

**Error:** `OSError: cannot load library 'portaudio'`

**Solution (macOS):**
```bash
brew install portaudio libsndfile
pip install --upgrade soundfile pyaudio
```

**Solution (Linux):**
```bash
sudo apt-get install portaudio19-dev libsndfile1-dev
pip install --upgrade soundfile pyaudio
```

## Runtime Issues

### Audio Processing Errors

**Error:** `ValueError: Invalid audio file`

**Solutions:**
```bash
# Check file format
file audio.wav
# Should show: RIFF (little-endian) data, WAVE audio

# Check if file is corrupted
ffmpeg -v error -i audio.wav -f null -

# Convert to compatible format
ffmpeg -i input.mp3 -ar 44100 -ac 1 output.wav
```

**Error:** `librosa.util.exceptions.ParameterError: Invalid sample rate`

**Solution:**
```bash
# Resample audio to 44.1kHz
ffmpeg -i input.wav -ar 44100 output.wav
```

### Parameter Generation Errors

**Error:** `JSONDecodeError: Invalid JSON in LLM response`

**Causes & Solutions:**

1. **Temperature too high** → LLM being creative with format
   ```bash
   # Lower temperature for more structured output
   --temperature 0.3
   ```

2. **Response truncated** → Increase max_tokens
   ```bash
   --max-tokens 3000
   ```

3. **Model hallucinating** → Use more capable model
   ```bash
   --model claude-3-opus-20240229
   ```

4. **Prompt issue** → Check system prompt formatting
   ```bash
   # Enable debug logging
   --log-level DEBUG
   # Check logs for actual prompt sent
   ```

### Memory Errors

**Error:** `MemoryError: Unable to allocate array`

**Solutions:**

1. **Reduce batch size:**
   ```yaml
   execution:
     batch_size: 5  # Instead of 20
   ```

2. **Process shorter audio:**
   ```bash
   # Trim audio to 30 seconds
   ffmpeg -i input.wav -t 30 short.wav
   ```

3. **Lower sample rate:**
   ```yaml
   audio:
     sample_rate: 44100  # Instead of 96000
   ```

4. **Reduce parallel workers:**
   ```yaml
   execution:
     parallel_workers: 1  # Instead of 4
   ```

5. **Close other applications** to free memory

### Checkpoint Issues

**Error:** `FileNotFoundError: checkpoint.json not found`

**Solutions:**
```bash
# Check experiment directory exists
ls outputs/batch1/

# Verify permissions
chmod -R 755 outputs/

# Don't delete files during experiment
# Don't run multiple experiments in same output directory
```

**Resume doesn't work:**
```bash
# Check checkpoint file
cat outputs/batch1/checkpoint.json

# Ensure exact same config/parameters
python -m src.runner.cli resume \
  --experiment-dir outputs/batch1
```

## Performance Issues

### Slow Execution

**Diagnosis:**
```bash
# Profile single experiment
python -m cProfile -o profile.stats \
  -m src.runner.cli run-single \
  --description "test" \
  --audio test.wav

# Analyze profile
python -m pstats profile.stats
>>> sort cumtime
>>> stats 20
```

**Common Bottlenecks:**

1. **LLM API calls** (unavoidable)
   - Use faster model (Sonnet vs Opus)
   - Lower max_tokens

2. **Audio processing**
   ```yaml
   audio:
     sample_rate: 44100  # Lower from 96000
   ```

3. **Scoring**
   ```yaml
   scoring:
     method: embedding  # Faster than llm_judge
   ```

4. **I/O operations**
   ```yaml
   output:
     save_audio: false  # Skip if not needed
   ```

**Optimizations:**
```yaml
execution:
  parallel_workers: 4  # Use all CPU cores
  batch_size: 20      # Process more at once
```

### High API Costs

**Strategies:**

1. **Use cheaper models:**
   ```yaml
   llm:
     provider: openrouter
     model: meta-llama/llama-3.1-70b  # Cheaper alternative
   ```

2. **Reduce token usage:**
   ```yaml
   llm:
     temperature: 0.5  # More concise responses
     max_tokens: 1500  # Lower limit
   ```

3. **Use embedding scoring:**
   ```yaml
   scoring:
     method: embedding  # No LLM scoring API calls
   ```

4. **Development mode:**
   ```python
   # Use mock provider for development
   from tests.mocks.mock_provider import MockLLMProvider
   provider = MockLLMProvider(response_mode="valid")
   ```

5. **Cache results:**
   - Don't rerun same prompts
   - Save successful parameters
   - Reuse for similar descriptions

## Debugging

### Enable Debug Logging

```yaml
output:
  log_level: DEBUG
```

Or via CLI:
```bash
python -m src.runner.cli run-single \
  --description "test" \
  --audio test.wav \
  --log-level DEBUG
```

Check logs:
```bash
# Real-time monitoring
tail -f outputs/experiment1/logs/experiment.log

# Search for errors
grep ERROR outputs/experiment1/logs/experiment.log

# View LLM interactions
grep "LLM request\|LLM response" outputs/experiment1/logs/experiment.log
```

### Validate Configuration

```bash
# Dry run (validates without execution)
python -m src.runner.cli run-single \
  --config my_config.yaml \
  --description "test" \
  --audio test.wav \
  --dry-run
```

### Test Individual Components

```bash
# Test LLM provider
python -c "
from src.providers import get_anthropic_client
client = get_anthropic_client()
response = client.messages.create(
    model='claude-3-5-sonnet-20241022',
    max_tokens=100,
    messages=[{'role': 'user', 'content': 'test'}]
)
print(response.content)
"

# Test audio loading
python -c "
import librosa
audio, sr = librosa.load('test.wav', sr=44100)
print(f'Loaded {len(audio)} samples at {sr}Hz')
"

# Test scoring
python -c "
from src.scoring.scorer import ScoringSystem
from tests.mocks.mock_provider import MockLLMProvider
scorer = ScoringSystem(MockLLMProvider())
print('Scorer initialized successfully')
"
```

## Common Workflow Issues

### Batch Gets Stuck

**Check:**
```bash
# Monitor progress
watch -n 5 'ls -lt outputs/batch1/ | head -20'

# Check for hung processes
ps aux | grep python

# Check logs for last activity
tail -20 outputs/batch1/logs/experiment.log
```

**Solutions:**
- Kill and resume with checkpoint
- Reduce timeout in config
- Check network connectivity (for API calls)

### Results Don't Match Expectation

**Debugging:**
```bash
# Listen to outputs
afplay outputs/experiment1/output.wav

# Compare parameters
cat outputs/experiment1/parameters.json | jq .

# Check scores
cat outputs/experiment1/metadata.json | jq .score

# Review prompt that was sent
grep "LLM request" outputs/experiment1/logs/experiment.log
```

**Common Issues:**
- Description too vague → Be more specific
- Temperature too high → Lower for consistency
- Wrong audio input → Check input file
- Scoring mismatch → Verify scoring method

## Getting Help

1. **Check logs first:**
   ```bash
   cat outputs/*/logs/experiment.log | grep ERROR
   ```

2. **Search existing issues:**
   - [GitHub Issues](https://github.com/vaclisinc/text2preset/issues)

3. **Create detailed bug report:**
   - Error message (full stack trace)
   - Configuration used
   - Steps to reproduce
   - System info (OS, Python version)
   - Relevant log excerpts

4. **Ask team:**
   - Share experiment directory
   - Include `metadata.json` and `parameters.json`
   - Describe expected vs actual behavior

## Quick Fixes Checklist

- [ ] `.env` file exists with valid API keys
- [ ] Virtual environment activated
- [ ] In correct directory (`baseline-system/`)
- [ ] Audio file format is valid WAV
- [ ] Sufficient disk space for outputs
- [ ] Network connectivity for API calls
- [ ] Python 3.9+ installed
- [ ] All dependencies installed (`pip install -r requirements.txt`)

## See Also

- [Usage Guide](usage.md) - Correct command syntax
- [Configuration](configuration.md) - Valid config options
- [Examples](examples.md) - Working examples
- [Architecture](architecture.md) - System design
