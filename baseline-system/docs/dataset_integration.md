# SocialFX Dataset Integration Guide

This guide covers the SocialFX dataset integration, which provides reference audio samples and ground-truth parameter configurations for the baseline system. The dataset enables few-shot prompting and serves as evaluation benchmarks for measuring parameter generation performance.

## Table of Contents

- [Overview](#overview)
- [Dataset Structure](#dataset-structure)
- [Installation & Setup](#installation--setup)
- [Basic Usage](#basic-usage)
- [Advanced Features](#advanced-features)
- [API Reference](#api-reference)
- [Error Handling](#error-handling)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

## Overview

The SocialFX dataset contains professionally-crafted audio effect parameters with corresponding text descriptions. It includes:

- **3 Instrument Types**: Guitar, Drums, Piano
- **3 Effect Types**: EQ (Equalizer), Reverb, Compressor
- **Reference Audio Samples**: Original audio files for each instrument
- **Ground-Truth Parameters**: CSV files with validated parameter configurations
- **Text Descriptions**: Natural language descriptions of desired audio effects

### Use Cases

1. **Few-Shot Prompting**: Provide examples to guide LLM parameter generation
2. **Validation Benchmarks**: Test generated parameters against ground truth
3. **Diversity Selection**: Select diverse examples across instruments and effects
4. **Dataset Analysis**: Compute statistics and parameter ranges
5. **Reference Implementation**: Understand expected parameter formats

## Dataset Structure

### Directory Layout

```
data/socialfx/
├── audio/
│   ├── guitar.wav      # Reference guitar sample (44.1 kHz)
│   ├── drums.wav       # Reference drums sample (44.1 kHz)
│   └── piano.wav       # Reference piano sample (44.1 kHz)
└── parameters/
    ├── eq_params.csv          # EQ parameter configurations
    ├── reverb_params.csv      # Reverb parameter configurations
    └── compressor_params.csv  # Compressor parameter configurations
```

### CSV File Formats

#### EQ Parameters (`eq_params.csv`)

EQ parameters support multiple bands with frequency, gain, and Q (bandwidth) values.

```csv
id,description,instrument,band1_freq,band1_gain,band1_q,band2_freq,band2_gain,band2_q,band3_freq,band3_gain,band3_q
1,"warm and intimate",guitar,200,3.0,0.7,3000,-2.0,1.2,8000,1.5,0.8
2,"bright and aggressive",guitar,5000,4.0,1.0,10000,2.0,0.8,100,-3.0,0.5
```

**Column Description:**
- `id`: Unique identifier for the example
- `description`: Text description of the desired audio effect
- `instrument`: Instrument type (guitar, drums, or piano)
- `band{N}_freq`: Center frequency for band N (in Hz)
- `band{N}_gain`: Gain adjustment for band N (in dB)
- `band{N}_q`: Q factor (bandwidth) for band N

#### Reverb Parameters (`reverb_params.csv`)

Reverb parameters control spatial characteristics and ambience.

```csv
id,description,instrument,room_size,damping,wet_level,dry_level,width,freeze_mode
1,"spacious cathedral",piano,0.9,0.3,0.4,0.6,0.9,false
2,"tight room",drums,0.2,0.8,0.2,0.8,0.5,false
```

**Column Description:**
- `id`: Unique identifier
- `description`: Text description
- `instrument`: Instrument type
- `room_size`: Virtual room size (0.0 - 1.0)
- `damping`: High-frequency damping (0.0 - 1.0)
- `wet_level`: Effect level (0.0 - 1.0)
- `dry_level`: Original signal level (0.0 - 1.0)
- `width`: Stereo width (0.0 - 1.0)
- `freeze_mode`: Freeze reverb tail (true/false)

#### Compressor Parameters (`compressor_params.csv`)

Compressor parameters control dynamic range and punch.

```csv
id,description,instrument,threshold,ratio,attack,release,knee,makeup_gain
1,"punchy and controlled",drums,-20,4.0,5,50,3,2.0
2,"smooth leveling",guitar,-30,2.5,10,100,6,3.5
```

**Column Description:**
- `id`: Unique identifier
- `description`: Text description
- `instrument`: Instrument type
- `threshold`: Compression threshold (in dB)
- `ratio`: Compression ratio (e.g., 4.0 = 4:1)
- `attack`: Attack time (in milliseconds)
- `release`: Release time (in milliseconds)
- `knee`: Soft knee width (in dB)
- `makeup_gain`: Output gain compensation (in dB)

## Installation & Setup

### Prerequisites

```bash
# Install required dependencies
pip install pandas pydantic soundfile numpy

# Or install via requirements.txt
pip install -r requirements.txt
```

### Preparing the Dataset

1. **Create Directory Structure**:

```bash
mkdir -p data/socialfx/audio
mkdir -p data/socialfx/parameters
```

2. **Add Audio Samples**:

Place your reference audio files (guitar.wav, drums.wav, piano.wav) in `data/socialfx/audio/`. Ensure they are:
- Sample rate: 44.1 kHz
- Format: WAV (uncompressed)
- Mono or stereo

3. **Add Parameter CSV Files**:

Place the CSV files in `data/socialfx/parameters/`:
- `eq_params.csv`
- `reverb_params.csv`
- `compressor_params.csv`

### Verify Installation

```python
from judge_system.data import SocialFXDataset

# Initialize dataset loader
dataset = SocialFXDataset(data_dir="data/socialfx")

# Load dataset - raises FileNotFoundError if files are missing
dataset.load()

print(f"Loaded {dataset.metadata.total_examples} examples")
print(f"Instruments: {dataset.metadata.instruments}")
print(f"Effect types: {dataset.metadata.effect_types}")
```

## Basic Usage

### Loading the Dataset

```python
from judge_system.data import SocialFXDataset

# Initialize with default directory
dataset = SocialFXDataset()

# Or specify custom directory
dataset = SocialFXDataset(data_dir="path/to/socialfx")

# Load all examples into memory
dataset.load()
```

### Accessing Examples

```python
# Get all examples
all_examples = dataset.examples

# Access individual example
example = all_examples[0]
print(f"ID: {example.id}")
print(f"Description: {example.description}")
print(f"Instrument: {example.instrument}")
print(f"Effect Type: {example.effect_type}")
print(f"Parameters: {example.parameters}")
print(f"Audio Path: {example.audio_path}")
```

### Filtering Examples

```python
# Filter by instrument
guitar_examples = dataset.get_examples(instrument="guitar")
print(f"Found {len(guitar_examples)} guitar examples")

# Filter by effect type
eq_examples = dataset.get_examples(effect_type="eq")
print(f"Found {len(eq_examples)} EQ examples")

# Filter by both
guitar_eq = dataset.get_examples(instrument="guitar", effect_type="eq")
print(f"Found {len(guitar_eq)} guitar EQ examples")

# Limit results
limited = dataset.get_examples(effect_type="reverb", limit=5)
print(f"Retrieved {len(limited)} reverb examples (max 5)")
```

### Loading Audio Files

```python
from judge_system.data import load_audio_sample

# Load audio sample
audio, sample_rate = load_audio_sample(example.audio_path)

print(f"Audio shape: {audio.shape}")
print(f"Sample rate: {sample_rate} Hz")
print(f"Duration: {len(audio) / sample_rate:.2f} seconds")
```

### Formatting for Prompts

```python
from judge_system.data import format_example_for_prompt

# Format example for few-shot prompting
formatted = format_example_for_prompt(example)
print(formatted)

# Output:
# Description: "warm and intimate"
# Instrument: guitar
# Parameters:
# {
#   "bands": [
#     {
#       "frequency": 200,
#       "gain": 3.0,
#       "q": 0.7
#     },
#     ...
#   ]
# }
```

## Advanced Features

### Few-Shot Example Selection

The dataset provides intelligent selection of diverse examples for few-shot prompting:

```python
# Get 3 diverse examples for EQ
examples = dataset.get_few_shot_examples(
    effect_type="eq",
    n_examples=3,
    diverse=True  # Select across different instruments
)

# Build few-shot prompt
prompt_examples = [format_example_for_prompt(ex) for ex in examples]
few_shot_prompt = "\n\n".join(prompt_examples)

print(f"Selected {len(examples)} examples for few-shot prompting")
```

The `diverse=True` option uses round-robin selection across instruments to ensure variety:
- Example 1: Guitar
- Example 2: Drums
- Example 3: Piano
- Example 4: Guitar (cycles back)

### Dataset Statistics

```python
# Access metadata
metadata = dataset.metadata

print(f"Total examples: {metadata.total_examples}")
print(f"Instruments: {metadata.instruments}")
print(f"Effect types: {metadata.effect_types}")

# Example counts per effect type
for effect_type, count in metadata.description_count.items():
    print(f"{effect_type}: {count} examples")

# Parameter ranges
for effect_type, ranges in metadata.parameter_ranges.items():
    print(f"\n{effect_type.upper()} parameter ranges:")
    for param, (min_val, max_val) in ranges.items():
        print(f"  {param}: [{min_val:.2f}, {max_val:.2f}]")
```

Example output:
```
EQ parameter ranges:
  frequency: [20.00, 20000.00]
  gain: [-24.00, 24.00]
  q: [0.30, 2.00]

REVERB parameter ranges:
  room_size: [0.00, 1.00]
  damping: [0.00, 1.00]
  wet_level: [0.00, 1.00]
```

### Custom Filtering Logic

```python
# Filter by description keywords
def filter_by_description(examples, keyword):
    return [ex for ex in examples if keyword.lower() in ex.description.lower()]

bright_examples = filter_by_description(dataset.examples, "bright")
warm_examples = filter_by_description(dataset.examples, "warm")

print(f"Found {len(bright_examples)} examples with 'bright'")
print(f"Found {len(warm_examples)} examples with 'warm'")

# Filter by parameter ranges
def filter_by_reverb_size(examples, min_size=0.5):
    return [
        ex for ex in examples
        if ex.effect_type == "reverb" and ex.parameters.get("room_size", 0) >= min_size
    ]

large_rooms = filter_by_reverb_size(dataset.examples, min_size=0.7)
print(f"Found {len(large_rooms)} examples with large room size")
```

## API Reference

### Core Classes

For detailed API documentation, see the docstrings in:
- `judge_system/data/models.py` - Data models (SocialFXExample, DatasetMetadata)
- `judge_system/data/dataset.py` - Main dataset class (SocialFXDataset)
- `judge_system/data/audio_utils.py` - Audio utilities (load_audio_sample, format_example_for_prompt)

### SocialFXDataset

**Constructor:**
```python
SocialFXDataset(data_dir: str = "data/socialfx")
```

**Methods:**
- `load()` - Load entire dataset into memory
- `get_examples(instrument=None, effect_type=None, limit=None)` - Get filtered examples
- `get_few_shot_examples(effect_type, n_examples=3, diverse=True)` - Get examples for few-shot prompting

**Attributes:**
- `examples: List[SocialFXExample]` - All loaded examples
- `metadata: DatasetMetadata` - Dataset statistics and ranges
- `data_dir: Path` - Root data directory
- `audio_dir: Path` - Audio files directory
- `params_dir: Path` - Parameter CSV files directory

### SocialFXExample

**Fields:**
- `id: int` - Unique identifier (>= 0)
- `description: str` - Text description (min length 1)
- `instrument: str` - Instrument type (guitar, drums, or piano)
- `effect_type: str` - Effect type (min length 1)
- `parameters: Dict[str, Any]` - Effect parameters (non-empty)
- `audio_path: Optional[str]` - Path to audio file (optional)

**Validation:**
- Instrument must be one of: guitar, drums, piano
- Parameters dictionary cannot be empty
- Extra fields are forbidden (strict mode)

### DatasetMetadata

**Fields:**
- `total_examples: int` - Total number of examples (>= 0)
- `instruments: List[str]` - List of instrument types (non-empty)
- `effect_types: List[str]` - List of effect types (non-empty)
- `description_count: Dict[str, int]` - Count per effect type (all >= 0)
- `parameter_ranges: Dict[str, Dict[str, Tuple[float, float]]]` - Min/max ranges per effect type

**Validation:**
- All counts must be non-negative
- Parameter ranges must have min <= max
- Lists cannot be empty

## Error Handling

### Common Errors

#### Missing Dataset Files

```python
from pathlib import Path

try:
    dataset = SocialFXDataset(data_dir="data/socialfx")
    dataset.load()
except FileNotFoundError as e:
    print(f"Dataset files missing: {e}")
    print("Ensure all required files exist:")
    print("  - data/socialfx/audio/guitar.wav")
    print("  - data/socialfx/audio/drums.wav")
    print("  - data/socialfx/audio/piano.wav")
    print("  - data/socialfx/parameters/eq_params.csv")
    print("  - data/socialfx/parameters/reverb_params.csv")
    print("  - data/socialfx/parameters/compressor_params.csv")
```

#### Invalid Data in CSV

```python
from pydantic import ValidationError

try:
    dataset = SocialFXDataset()
    dataset.load()
except ValidationError as e:
    print(f"Data validation failed: {e}")
    print("Check CSV files for:")
    print("  - Invalid instrument names (must be guitar, drums, or piano)")
    print("  - Empty parameter dictionaries")
    print("  - Negative IDs")
    print("  - Empty descriptions")
```

#### Audio File Loading Errors

```python
from judge_system.data import load_audio_sample

try:
    audio, sr = load_audio_sample("path/to/audio.wav")
except FileNotFoundError:
    print("Audio file not found")
except RuntimeError as e:
    print(f"Failed to read audio file: {e}")
    print("Ensure file is valid WAV format")
```

### Validation Best Practices

```python
from pathlib import Path

def validate_dataset_structure(data_dir: str) -> bool:
    """Validate dataset directory structure before loading."""
    data_path = Path(data_dir)

    # Check directories exist
    if not data_path.exists():
        print(f"Data directory not found: {data_dir}")
        return False

    audio_dir = data_path / "audio"
    params_dir = data_path / "parameters"

    if not audio_dir.exists():
        print(f"Audio directory not found: {audio_dir}")
        return False

    if not params_dir.exists():
        print(f"Parameters directory not found: {params_dir}")
        return False

    # Check required files
    required_files = [
        audio_dir / "guitar.wav",
        audio_dir / "drums.wav",
        audio_dir / "piano.wav",
        params_dir / "eq_params.csv",
        params_dir / "reverb_params.csv",
        params_dir / "compressor_params.csv"
    ]

    missing = [f for f in required_files if not f.exists()]
    if missing:
        print("Missing required files:")
        for f in missing:
            print(f"  - {f}")
        return False

    return True

# Use before loading
if validate_dataset_structure("data/socialfx"):
    dataset = SocialFXDataset()
    dataset.load()
else:
    print("Dataset structure validation failed")
```

## Examples

### Example 1: Building Few-Shot Prompts

```python
from judge_system.data import SocialFXDataset, format_example_for_prompt

# Load dataset
dataset = SocialFXDataset()
dataset.load()

# Select diverse EQ examples
examples = dataset.get_few_shot_examples(
    effect_type="eq",
    n_examples=3,
    diverse=True
)

# Build few-shot prompt
prompt = "Generate EQ parameters for the following description.\n\n"
prompt += "Here are some examples:\n\n"

for i, example in enumerate(examples, 1):
    prompt += f"Example {i}:\n"
    prompt += format_example_for_prompt(example)
    prompt += "\n\n"

prompt += "Now generate parameters for:\n"
prompt += 'Description: "crisp and clear"\n'
prompt += "Instrument: guitar\n"
prompt += "Parameters:\n"

print(prompt)
```

### Example 2: Analyzing Parameter Distributions

```python
import numpy as np
from judge_system.data import SocialFXDataset

dataset = SocialFXDataset()
dataset.load()

# Analyze EQ gain distributions
eq_examples = dataset.get_examples(effect_type="eq")
all_gains = []

for example in eq_examples:
    bands = example.parameters.get("bands", [])
    for band in bands:
        all_gains.append(band["gain"])

if all_gains:
    print("EQ Gain Statistics:")
    print(f"  Mean: {np.mean(all_gains):.2f} dB")
    print(f"  Std:  {np.std(all_gains):.2f} dB")
    print(f"  Min:  {np.min(all_gains):.2f} dB")
    print(f"  Max:  {np.max(all_gains):.2f} dB")
```

### Example 3: Validating Generated Parameters

```python
from judge_system.data import SocialFXDataset

dataset = SocialFXDataset()
dataset.load()

def validate_generated_reverb(params: dict) -> bool:
    """Validate generated reverb parameters against dataset ranges."""
    # Get reverb parameter ranges from metadata
    if "reverb" not in dataset.metadata.parameter_ranges:
        print("No reverb examples in dataset")
        return False

    ranges = dataset.metadata.parameter_ranges["reverb"]

    # Check each parameter is within observed ranges
    for param, value in params.items():
        if param not in ranges:
            print(f"Warning: Unknown parameter '{param}'")
            continue

        min_val, max_val = ranges[param]
        if not (min_val <= value <= max_val):
            print(f"Parameter '{param}' = {value} outside range [{min_val}, {max_val}]")
            return False

    return True

# Test with generated parameters
generated_params = {
    "room_size": 0.7,
    "damping": 0.4,
    "wet_level": 0.3,
    "dry_level": 0.7,
    "width": 0.8
}

if validate_generated_reverb(generated_params):
    print("Generated parameters are valid")
else:
    print("Generated parameters are out of range")
```

### Example 4: Batch Processing Audio Samples

```python
from judge_system.data import SocialFXDataset, load_audio_sample
import numpy as np

dataset = SocialFXDataset()
dataset.load()

# Load all audio samples
audio_samples = {}

for instrument in dataset.metadata.instruments:
    examples = dataset.get_examples(instrument=instrument, limit=1)
    if examples:
        audio, sr = load_audio_sample(examples[0].audio_path)
        audio_samples[instrument] = {
            "audio": audio,
            "sample_rate": sr,
            "duration": len(audio) / sr
        }

# Print audio information
print("Loaded Audio Samples:")
for instrument, info in audio_samples.items():
    print(f"  {instrument}:")
    print(f"    Duration: {info['duration']:.2f}s")
    print(f"    Sample Rate: {info['sample_rate']} Hz")
    print(f"    Shape: {info['audio'].shape}")
```

### Example 5: Dataset Statistics Report

```python
from judge_system.data import SocialFXDataset

dataset = SocialFXDataset()
dataset.load()

print("=" * 70)
print("SocialFX Dataset Report")
print("=" * 70)

# Overall statistics
print(f"\nTotal Examples: {dataset.metadata.total_examples}")
print(f"Instruments: {', '.join(dataset.metadata.instruments)}")
print(f"Effect Types: {', '.join(dataset.metadata.effect_types)}")

# Examples per effect type
print("\nExamples per Effect Type:")
for effect_type, count in sorted(dataset.metadata.description_count.items()):
    print(f"  {effect_type.upper()}: {count}")

# Examples per instrument
print("\nExamples per Instrument:")
for instrument in dataset.metadata.instruments:
    count = len(dataset.get_examples(instrument=instrument))
    print(f"  {instrument.capitalize()}: {count}")

# Parameter ranges
print("\nParameter Ranges:")
for effect_type in sorted(dataset.metadata.effect_types):
    if effect_type in dataset.metadata.parameter_ranges:
        print(f"\n  {effect_type.upper()}:")
        ranges = dataset.metadata.parameter_ranges[effect_type]
        for param, (min_val, max_val) in sorted(ranges.items()):
            print(f"    {param}: [{min_val:.2f}, {max_val:.2f}]")

print("\n" + "=" * 70)
```

## Troubleshooting

### Issue: "Missing required files" error

**Cause**: Dataset directory structure is incomplete or files are in wrong location.

**Solution**:
```bash
# Verify directory structure
ls -R data/socialfx/

# Should show:
# data/socialfx/audio/guitar.wav
# data/socialfx/audio/drums.wav
# data/socialfx/audio/piano.wav
# data/socialfx/parameters/eq_params.csv
# data/socialfx/parameters/reverb_params.csv
# data/socialfx/parameters/compressor_params.csv
```

### Issue: "Instrument must be one of ['guitar', 'drums', 'piano']" error

**Cause**: CSV file contains invalid instrument name.

**Solution**: Check CSV files and ensure `instrument` column only contains: `guitar`, `drums`, or `piano` (lowercase).

### Issue: "Parameters dictionary cannot be empty" error

**Cause**: CSV row has no parameter columns or all parameter values are null.

**Solution**: Ensure each row has at least one valid parameter value. For EQ, at least `band1_freq`, `band1_gain`, and `band1_q` must be present.

### Issue: "Sample rate {X} != 44100" warning

**Cause**: Audio file has different sample rate than expected.

**Solution**: This is a warning, not an error. The audio will still load, but you may need to resample:

```python
import librosa

# Load and resample to 44.1 kHz
audio, sr = librosa.load("audio.wav", sr=44100)
```

### Issue: No examples returned from filtering

**Cause**: Filter criteria too restrictive or typo in parameter values.

**Solution**:
```python
# Check available values
print("Available instruments:", dataset.metadata.instruments)
print("Available effect types:", dataset.metadata.effect_types)

# Use correct lowercase values
examples = dataset.get_examples(instrument="guitar")  # Not "Guitar"
examples = dataset.get_examples(effect_type="eq")      # Not "EQ"
```

### Issue: Memory usage is high

**Cause**: Loading large dataset into memory.

**Solution**: Use filtering and limits to reduce memory footprint:

```python
# Don't load all examples at once
dataset = SocialFXDataset()
dataset.load()

# Instead, filter immediately
eq_examples = dataset.get_examples(effect_type="eq", limit=10)

# Or process in batches
for effect_type in ["eq", "reverb", "compressor"]:
    examples = dataset.get_examples(effect_type=effect_type, limit=5)
    # Process examples
    del examples  # Free memory
```

## Additional Resources

- **Source Code**: See `judge_system/data/` for complete implementation
- **Tests**: See `tests/unit/test_judge_system/test_data_models.py` for usage examples
- **Examples**: See `examples/` directory for practical use cases
- **Main Documentation**: See `README.md` for project overview

## Contributing

When adding new examples to the dataset:

1. Follow the CSV format exactly
2. Ensure instrument names are lowercase (guitar, drums, piano)
3. Validate parameter ranges are reasonable
4. Add corresponding test cases
5. Update documentation if adding new effect types

For questions or issues, please open an issue on the project repository.
