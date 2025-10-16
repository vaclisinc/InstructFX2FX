# LLM-as-Music-Judge: Baseline System

A research implementation exploring whether Large Language Models can act as a "music judge" that listens, reasons, and scores generative audio like a human would. This baseline system establishes the foundational closed-loop architecture between LLM → Parameter Generation → Audio Processing → Judge Evaluation.

## Overview

This system implements a minimal self-loop for the LLM-as-music-judge research project:

1. **Input**: User provides high-level description (e.g., "after rain campus in October")
2. **LLM Processing**: Generates audio effect parameters in JSON format
3. **Audio Processing**: Applies parameters to input audio sample
4. **Judge Evaluation**: Scores the processed audio against the description
5. **Iterative Refinement**: Feeds score back to LLM for re-prompting and improvement

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Testing](#testing)
- [Development](#development)
- [Logging](#logging)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

- **Python**: 3.9 or higher (3.10+ recommended)
- **Operating System**: macOS, Linux, or Windows
- **Memory**: 8GB RAM minimum (16GB recommended for model inference)
- **Storage**: 2GB free space for dependencies and models

### Audio Processing Dependencies

#### macOS
```bash
brew install portaudio libsndfile
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt-get update
sudo apt-get install portaudio19-dev libsndfile1-dev
```

#### Windows
```bash
# Install through conda (recommended)
conda install -c conda-forge portaudio libsndfile
```

### API Keys

This project requires API keys from LLM providers. You'll need at least one of:

- **Anthropic API Key** (recommended): For Claude models
- **OpenRouter API Key**: For access to multiple LLM providers
- **OpenAI API Key**: For GPT models

Sign up for API access:
- Anthropic: https://console.anthropic.com/
- OpenRouter: https://openrouter.ai/
- OpenAI: https://platform.openai.com/

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/vaclisinc/text2preset.git
cd text2preset/baseline-system
```

### 2. Create Virtual Environment

Creating a virtual environment ensures dependencies are isolated from your system Python.

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

You should see `(venv)` in your terminal prompt indicating the virtual environment is active.

### 3. Install Dependencies

```bash
# Upgrade pip to latest version
pip install --upgrade pip

# Install all required dependencies
pip install -r requirements.txt
```

This will install:
- **LLM Integration**: anthropic, openai
- **Audio Processing**: librosa, soundfile, pedalboard, numpy
- **Configuration & Validation**: pydantic, pyyaml
- **Logging**: structlog
- **Dataset Integration**: datasets
- **Development & Testing**: pytest, pytest-cov, python-dotenv

### 4. Install in Development Mode

Install the package in editable mode for active development:

```bash
pip install -e .
```

This allows you to modify source code and immediately see changes without reinstalling.

### 5. Install Development Tools (Optional)

For code quality and type checking:

```bash
pip install -e ".[dev]"
```

This installs additional tools:
- **black**: Code formatting
- **flake8**: Linting
- **mypy**: Type checking

## Configuration

### 1. Create Environment File

Copy the template and add your API keys:

```bash
cp .env.template .env
```

### 2. Edit .env File

Open `.env` in your editor and add your API keys:

```bash
# LLM API Keys (add at least one)
ANTHROPIC_API_KEY=sk-ant-your-actual-key-here
OPENROUTER_API_KEY=sk-or-your-actual-key-here

# Configuration (optional - defaults are provided)
CONFIG_PATH=configs/default.yaml
LOG_LEVEL=INFO

# Audio Processing (optional - defaults are provided)
SAMPLE_RATE=44100
AUDIO_DIR=./audio_samples
```

**Important**: Never commit the `.env` file to version control. It's already in `.gitignore`.

### 3. Configuration Files

The system uses YAML configuration files in the `configs/` directory:

- **`configs/default.yaml`**: Default settings for all components
- **`configs/experiment.yaml`**: Experiment-specific configuration with detailed settings

Configuration hierarchy (highest to lowest priority):
1. Environment variables (`.env` file)
2. Profile-specific YAML (e.g., `experiment.yaml`)
3. Default YAML (`default.yaml`)
4. Pydantic model defaults

#### Customizing Configuration

Edit `configs/default.yaml` or create your own profile:

```yaml
# Example custom configuration
audio:
  sample_rate: 48000
  audio_dir: ./custom_audio

llm:
  provider: anthropic
  model: claude-3-5-sonnet-20241022
  temperature: 0.7

logging:
  level: DEBUG
  format: json
```

## Quick Start

### 1. Verify Installation

Run the smoke test to ensure everything is set up correctly:

```bash
python smoke_test.py
```

Expected output:
```
Smoke Test Results:
  ✓ Environment setup working
  ✓ Configuration loader functional
  ✓ Logging system initialized
  ✓ All dependencies imported
  ✓ API keys configured

All checks passed! Environment is ready.
```

### 2. Run Your First Experiment

```python
from config.loader import load_config
from utils.logging import configure_logging, get_logger

# Load configuration
config = load_config()

# Initialize logging
configure_logging(
    level=config.logging.level,
    format=config.logging.format,
    output_dir=config.logging.output_dir,
    console_output=config.logging.console_output,
    file_output=config.logging.file_output
)

# Get logger
log = get_logger("experiment")
log.info("experiment_started", name=config.experiment.name)

# Your experiment code here...
```

## Project Structure

```
baseline-system/
├── src/                        # Source code
│   ├── config/                 # Configuration management
│   │   ├── __init__.py
│   │   └── loader.py           # YAML config loading with pydantic validation
│   ├── providers/              # LLM provider abstractions
│   │   └── __init__.py
│   ├── generation/             # Parameter generation
│   │   └── __init__.py
│   ├── processing/             # Audio processing
│   │   └── __init__.py
│   ├── scoring/                # Scoring system
│   │   └── __init__.py
│   └── utils/                  # Utilities
│       ├── __init__.py
│       └── logging.py          # Structured logging with structlog
│
├── tests/                      # Test suite
│   ├── unit/                   # Unit tests
│   ├── integration/            # Integration tests
│   └── fixtures/               # Test fixtures and data
│
├── configs/                    # Configuration files
│   ├── default.yaml            # Default configuration
│   └── experiment.yaml         # Experiment-specific config
│
├── audio_samples/              # Test audio files
│
├── logs/                       # Log files (created at runtime)
│   ├── general.log             # All logs
│   ├── api.log                 # LLM API calls
│   ├── audio.log               # Audio processing
│   └── scoring.log             # Scoring system
│
├── results/                    # Experiment results (created at runtime)
│
├── .env                        # Environment variables (create from .env.template)
├── .env.template               # Template for environment variables
├── .gitignore                  # Git ignore rules
├── requirements.txt            # Python dependencies
├── setup.py                    # Package setup
├── smoke_test.py               # Environment verification script
└── README.md                   # This file
```

## Usage

### Configuration System

```python
from config.loader import load_config, reload_config

# Load configuration (uses default or CONFIG_PATH env var)
config = load_config()

# Access configuration values
print(config.audio.sample_rate)  # 44100
print(config.llm.model)          # claude-3-5-sonnet-20241022

# Reload configuration (useful for hot-reloading)
config = reload_config()

# Load specific config file
config = load_config(config_path="configs/experiment.yaml")
```

### Logging System

```python
from utils.logging import (
    configure_logging,
    get_logger,
    get_api_logger,
    get_audio_logger,
    get_scoring_logger,
    bind_context,
    LogContext
)

# Configure logging once at application startup
configure_logging(
    level="INFO",
    format="json",
    output_dir="./logs",
    console_output=True,
    file_output=True
)

# Get category-specific loggers
api_log = get_api_logger()
audio_log = get_audio_logger()
scoring_log = get_scoring_logger()

# Log with structured data
api_log.info("llm_request",
    provider="anthropic",
    model="claude-3-5-sonnet-20241022",
    tokens=150
)

# Bind context for all subsequent logs
bind_context(experiment_id="exp_001", iteration=1)

# Or use context manager for temporary context
with LogContext(batch_id="batch_123"):
    audio_log.info("processing_started", file="sample.wav")
```

### Environment Variable Overrides

Configuration values can be overridden by environment variables:

```bash
# Override audio settings
export AUDIO_SAMPLE_RATE=48000
export AUDIO_DIR=/path/to/audio

# Override LLM settings
export LLM_PROVIDER=anthropic
export LLM_MODEL=claude-3-opus-20240229
export ANTHROPIC_API_KEY=sk-ant-your-key

# Override logging settings
export LOG_LEVEL=DEBUG
export LOG_FORMAT=json
```

## Testing

### Run All Tests

```bash
pytest
```

### Run with Coverage

```bash
pytest --cov=src --cov-report=html --cov-report=term
```

View coverage report:
```bash
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

### Run Specific Test Categories

```bash
# Unit tests only
pytest tests/unit/

# Integration tests only
pytest tests/integration/

# Specific test file
pytest tests/unit/test_config.py

# Specific test function
pytest tests/unit/test_config.py::test_load_default_config
```

### Run Tests with Verbose Output

```bash
pytest -v  # Verbose
pytest -vv # Very verbose
pytest -s  # Show print statements
```

## Development

### Code Formatting

```bash
# Format all Python files
black src/ tests/

# Check formatting without making changes
black --check src/ tests/
```

### Linting

```bash
# Run flake8 linter
flake8 src/ tests/

# With specific max line length
flake8 --max-line-length=100 src/ tests/
```

### Type Checking

```bash
# Run mypy type checker
mypy src/

# Strict mode
mypy --strict src/
```

### Adding New Dependencies

1. Add dependency to `requirements.txt` with pinned version
2. Install the new dependency: `pip install -r requirements.txt`
3. Update `setup.py` if it's a core dependency

### Development Workflow

1. **Create feature branch**: `git checkout -b feature/your-feature`
2. **Make changes**: Implement your feature or fix
3. **Format code**: `black src/ tests/`
4. **Run tests**: `pytest`
5. **Check linting**: `flake8 src/ tests/`
6. **Commit changes**: `git commit -m "Description of changes"`
7. **Push branch**: `git push origin feature/your-feature`
8. **Create pull request**: Submit PR for review

## Logging

The system uses structured logging with separate log files for different categories:

### Log Files

- **`logs/general.log`**: All application logs
- **`logs/api.log`**: LLM API calls and responses
- **`logs/audio.log`**: Audio processing events
- **`logs/scoring.log`**: Scoring system events

### Log Levels

- **DEBUG**: Detailed information for debugging
- **INFO**: General information about application flow
- **WARNING**: Warning messages for potential issues
- **ERROR**: Error messages for failures
- **CRITICAL**: Critical issues requiring immediate attention

### Log Formats

#### JSON Format (Production)
```json
{
  "event": "llm_request",
  "timestamp": "2025-10-16T12:34:56.789Z",
  "level": "info",
  "logger": "api",
  "provider": "anthropic",
  "model": "claude-3-5-sonnet-20241022",
  "tokens": 150
}
```

#### Console Format (Development)
```
2025-10-16T12:34:56.789Z [info] llm_request provider=anthropic model=claude-3-5-sonnet-20241022 tokens=150
```

## Troubleshooting

### Issue: ImportError for audio libraries

**Error**: `ImportError: No module named 'soundfile'` or `OSError: cannot load library 'libsndfile.so'`

**Solution**: Install system audio libraries (see [Prerequisites](#prerequisites))

### Issue: API key not found

**Error**: `ValueError: API key not configured`

**Solution**:
1. Ensure `.env` file exists and contains your API key
2. Check that the API key format is correct (starts with `sk-ant-` for Anthropic)
3. Verify the environment variable is loaded: `python -c "import os; print(os.getenv('ANTHROPIC_API_KEY'))"`

### Issue: Configuration file not found

**Error**: `FileNotFoundError: Config file not found: configs/default.yaml`

**Solution**: Ensure you're running commands from the `baseline-system/` directory

### Issue: Port audio errors

**Error**: `OSError: PortAudio library not found`

**Solution**: Install portaudio system library (see [Prerequisites](#prerequisites))

### Issue: Permission denied for log files

**Error**: `PermissionError: [Errno 13] Permission denied: './logs/general.log'`

**Solution**:
1. Ensure the logs directory is writable: `chmod 755 logs/`
2. Or disable file logging: Set `file_output: false` in config

### Issue: Tests failing with import errors

**Error**: `ModuleNotFoundError: No module named 'src'`

**Solution**: Install package in development mode: `pip install -e .`

### Issue: Virtual environment not activating

**Solution**:
```bash
# Deactivate any existing environment
deactivate

# Remove old venv
rm -rf venv/

# Create new venv
python -m venv venv

# Activate
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
```

### Getting Help

1. **Check logs**: Review log files in `logs/` directory
2. **Run smoke test**: `python smoke_test.py` to verify environment
3. **GitHub Issues**: Report bugs or request features
4. **Documentation**: See full research proposal in `docs/`

## Research Context

This baseline system is part of the LLM-as-Music-Judge research project by CNMAT Group 2, exploring whether LLMs can evaluate generative audio quality like human judges.

### Research Question
Can LLMs act as a "music judge" that listens, reasons, and scores generative audio like a human would?

### Key Innovation
Unlike prior embedding models that only measure similarity, this framework allows the LLM to reason about tone design and musical quality, transforming static embeddings into actionable evaluation through iterative refinement.

### Evaluation Framework

The judge system uses a three-part evaluation approach:

1. **Quantitative Metrics** (DSP-based)
   - Loudness analysis
   - Spectral balance
   - Objective physical features

2. **Qualitative/Vibe Metrics** (LLM-based)
   - Emotional dimensions (Calm → Energetic)
   - Texture analysis (Warm → Bright)
   - Natural vs Synthetic scoring

3. **Preference Alignment**
   - User preference data collection
   - Calibration of vibe weights
   - Personalization capabilities

### Related Work

- **VibeCheck (ICLR 2025)**: Framework for evaluating LLM "vibes" beyond accuracy
- **CLAP**: Contrastive Language-Audio Pretraining for multimodal understanding
- **MERT**: Music understanding model for emotion and textual alignment
- **FAD**: Fréchet Audio Distance for music quality evaluation

## License

This project is part of academic research at UC Berkeley's CNMAT.

## Contributors

CNMAT Group 2 - UC Berkeley

## Acknowledgments

Special thanks to CNMAT (Center for New Music and Audio Technologies) for supporting this research.
