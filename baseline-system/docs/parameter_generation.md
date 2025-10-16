# Parameter Generation Module

## Overview

The Parameter Generation module translates high-level textual descriptions of desired audio characteristics into structured JSON audio effect parameters. It leverages Large Language Models (LLMs) combined with prompt engineering techniques (including few-shot examples from Sony LLM2Fx research) to generate valid effect parameters for EQ, Reverb, and Compressor effects.

The module ensures 100% valid JSON output through:
- Schema validation using Pydantic models
- Parameter range enforcement
- Retry logic with correction prompts
- Automatic normalization of out-of-range values

## Architecture

The system consists of four main components:

### 1. Parameter Schemas (`src/models/parameters/`)
Pydantic models defining effect parameter structures with built-in validation:
- `EQParameters`: Parametric equalizer with 3-10 bands
- `ReverbParameters`: Room reverb with size, damping, and wet/dry controls
- `CompressorParameters`: Dynamics compressor with threshold, ratio, attack, and release
- `EffectChain`: Container for multiple effects with execution order

### 2. Prompt Engineering (`src/prompts/`, `configs/prompts/`)
Template-based system for LLM prompting:
- System prompts with audio engineering expertise
- Few-shot examples demonstrating parameter patterns
- User prompt templates with variable substitution
- Versioning system for A/B testing different prompt strategies

### 3. Parameter Generation (`src/generation/parameter_generator.py`)
Core orchestration class that:
- Formats prompts from templates
- Calls LLM provider to generate parameters
- Parses JSON output (handles markdown code blocks)
- Validates against Pydantic schemas
- Retries with correction prompts on failure

### 4. Validation & Normalization (`src/generation/validator.py`, `src/generation/normalizer.py`)
Pre- and post-validation utilities:
- **Validator**: Checks parameter structure and ranges before/after generation
- **Normalizer**: Corrects out-of-range values while preserving intent

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Create .env file with API key
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
# OR
echo "OPENROUTER_API_KEY=sk-or-..." > .env
```

**IMPORTANT**: Never commit API keys to git! Always use `.env` files.

### Basic Usage

```python
import asyncio
from dotenv import load_dotenv
from models.llm_judge import create_provider, AnthropicConfig
from src.generation import ParameterGenerator

# Load API keys
load_dotenv()

# Create LLM provider
provider = create_provider("anthropic", {
    "api_key": os.getenv("ANTHROPIC_API_KEY"),
    "model": "claude-3-5-sonnet-20241022"
})

# Create generator
generator = ParameterGenerator(
    llm_provider=provider,
    prompt_version="v1"
)

# Generate parameters
async def generate():
    chain = await generator.generate_parameters(
        description="warm and intimate vocal sound",
        effects=["eq", "reverb"]
    )

    print(f"Generated {len(chain.effects)} effects:")
    print(chain.to_dict())

asyncio.run(generate())
```

## API Reference

### ParameterGenerator Class

#### Constructor

```python
ParameterGenerator(
    llm_provider: LLMProvider,
    prompt_version: str = "v1",
    prompts_dir: Optional[Path] = None,
    max_correction_attempts: int = 3
)
```

**Parameters:**
- `llm_provider`: LLM provider instance (from `models.llm_judge`)
- `prompt_version`: Prompt template version (default: "v1")
- `prompts_dir`: Custom prompts directory (default: "configs/prompts/")
- `max_correction_attempts`: Max retry attempts for invalid output (default: 3)

**Raises:**
- `PromptTemplateError`: If template loading fails

#### Methods

##### generate_parameters()

Main entry point for parameter generation.

```python
async def generate_parameters(
    description: str,
    effects: Optional[List[str]] = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    include_examples: bool = True,
    num_examples: Optional[int] = None
) -> EffectChain
```

**Parameters:**
- `description`: High-level description of desired audio characteristics
- `effects`: List of effect types (default: `["eq", "reverb", "compressor"]`)
- `temperature`: LLM sampling temperature 0.0-2.0 (default: 0.7)
  - Lower (0.3-0.5): More focused, deterministic output
  - Medium (0.6-0.8): Balanced creativity and consistency
  - Higher (0.9-1.5): More creative, varied output
- `max_tokens`: Maximum tokens to generate (default: 2048)
- `include_examples`: Whether to include few-shot examples (default: True)
- `num_examples`: Number of examples to include (None = all available)

**Returns:**
- `EffectChain`: Validated effect chain with all effects

**Raises:**
- `ParameterGenerationError`: If generation fails after all retries
- `LLMProviderError`: If LLM provider fails
- `ValidationError`: If output cannot be validated after corrections
- `ValueError`: If invalid effect types provided

**Example:**

```python
chain = await generator.generate_parameters(
    description="bright and energetic guitar sound with punchy dynamics",
    effects=["eq", "compressor"],
    temperature=0.5,
    include_examples=True
)

# Access generated effects
for effect in chain.effects:
    print(f"{effect.effect_type}: {effect.to_dict()}")
```

##### parse_and_validate()

Parse JSON string and validate against schemas.

```python
def parse_and_validate(
    json_str: str,
    expected_effects: Optional[List[str]] = None,
    description: Optional[str] = None
) -> EffectChain
```

**Parameters:**
- `json_str`: JSON string from LLM (may include markdown code blocks)
- `expected_effects`: Expected effect types for validation
- `description`: Original description (used if missing in output)

**Returns:**
- `EffectChain`: Validated effect chain

**Raises:**
- `JSONParseError`: If JSON parsing fails
- `ValidationError`: If schema validation fails

**Example:**

```python
# Useful for testing or custom LLM outputs
json_output = '{"description": "test", "effects": [...]}'
chain = generator.parse_and_validate(json_output)
```

### Validator Functions

#### validate_effect_structure()

Pre-validate effect data before Pydantic parsing.

```python
from src.generation.validator import validate_effect_structure

result = validate_effect_structure({
    "type": "eq",
    "bands": [
        {"frequency": 1000, "gain": 3.0, "q": 1.0}
    ]
})

if result.is_valid:
    print("Valid!")
else:
    print(result.format_report())
```

**Returns:**
- `ValidationResult`: Object with `is_valid`, `issues`, and helper methods

#### validate_effect_chain_structure()

Pre-validate complete effect chain structure.

```python
from src.generation.validator import validate_effect_chain_structure

result = validate_effect_chain_structure({
    "description": "test sound",
    "effects": [...]
})
```

#### validate_effect_parameter()

Post-validate a Pydantic model instance.

```python
from src.generation.validator import validate_effect_parameter
from src.models.parameters import EQParameters

eq = EQParameters(bands=[...])
result = validate_effect_parameter(eq)
```

### Normalizer Functions

#### normalize_effect()

Normalize effect parameters to valid ranges.

```python
from src.generation.normalizer import normalize_effect

# Input with out-of-range values
effect_data = {
    "type": "compressor",
    "threshold": -100,  # Out of range!
    "ratio": 50,        # Out of range!
    "attack": 5.0,
    "release": 100.0,
    "knee": 3.0,
    "makeup_gain": 6.0
}

# Normalize (clamps to valid ranges)
normalized = normalize_effect(effect_data)
# normalized["threshold"] = -60 (clamped)
# normalized["ratio"] = 20 (clamped)
```

#### normalize_effect_chain_data()

Normalize complete effect chain data dictionary.

```python
from src.generation.normalizer import normalize_effect_chain_data

chain_data = {
    "description": "test",
    "effects": [...]
}

normalized = normalize_effect_chain_data(chain_data)
```

#### Utility Functions

```python
from src.generation.normalizer import clamp, safe_float, safe_bool

# Clamp value to range
value = clamp(150, min_val=0, max_val=100)  # Returns 100

# Safe type conversion with fallback
freq = safe_float("1000.5", default=440)    # Returns 1000.5
enabled = safe_bool("true", default=False)   # Returns True
```

## Effect Parameter Schemas

### EQ Parameters

Parametric equalizer with 3-10 frequency bands.

```python
from src.models.parameters import EQParameters, EQBand

eq = EQParameters(
    bands=[
        EQBand(frequency=200, gain=3.0, q=0.7),
        EQBand(frequency=1000, gain=0.0, q=1.0),
        EQBand(frequency=8000, gain=-2.0, q=1.2)
    ],
    eq_type="parametric"  # or "graphic", "shelving"
)
```

**Band Parameters:**
- `frequency`: Center frequency in Hz (20-20000)
- `gain`: Gain adjustment in dB (-12 to +12)
- `q`: Q factor/bandwidth (0.1 to 10)
  - 0.5-2.0: Broad, musical curves
  - 2.0-10.0: Narrow, surgical cuts/boosts

**Validation:**
- Minimum 3 bands, maximum 10 bands
- Bands automatically sorted by frequency
- Minimum 10% spacing between adjacent bands

### Reverb Parameters

Room reverb effect with spatial control.

```python
from src.models.parameters import ReverbParameters

reverb = ReverbParameters(
    room_size=0.4,      # Room size (0=small, 1=large)
    damping=0.6,        # High-frequency damping (0=none, 1=full)
    wet_level=0.3,      # Reverb signal level
    dry_level=0.8,      # Direct signal level
    width=0.7,          # Stereo width (0=mono, 1=full stereo)
    freeze_mode=False   # Infinite reverb tail
)
```

**Parameters:**
- `room_size`: 0.0-1.0
  - 0.2-0.4: Intimate spaces
  - 0.5-0.7: Medium rooms
  - 0.7-0.9: Large halls
- `damping`: 0.0-1.0
  - 0.3-0.5: Bright, lively
  - 0.5-0.7: Natural
  - 0.7-0.9: Dark, warm
- `wet_level`: 0.0-1.0 (reverb amount)
- `dry_level`: 0.0-1.0 (direct signal)
- `width`: 0.0-1.0 (stereo spread)
- `freeze_mode`: Boolean (infinite reverb)

**Validation:**
- All levels 0.0-1.0
- Combined wet+dry should not exceed 2.0

### Compressor Parameters

Dynamics compressor for level control.

```python
from src.models.parameters import CompressorParameters

compressor = CompressorParameters(
    threshold=-20.0,    # Compression threshold in dB
    ratio=4.0,          # Compression ratio (4:1)
    attack=5.0,         # Attack time in ms
    release=80.0,       # Release time in ms
    knee=3.0,           # Knee width in dB
    makeup_gain=6.0     # Gain compensation in dB
)
```

**Parameters:**
- `threshold`: -60 to 0 dB (compression trigger level)
- `ratio`: 1 to 20 (compression amount)
  - 1:1: No compression
  - 2:1-4:1: Gentle, transparent
  - 6:1-10:1: Aggressive, obvious
  - 10:1-20:1: Limiting
- `attack`: 0.1 to 100 ms
  - Fast (0.1-5 ms): Catch transients
  - Medium (5-20 ms): Balanced
  - Slow (20-100 ms): Preserve transients
- `release`: 10 to 1000 ms (should be longer than attack)
- `knee`: 0 to 12 dB
  - 0-3 dB: Hard knee (obvious)
  - 6-12 dB: Soft knee (transparent)
- `makeup_gain`: 0 to 24 dB (compensate for reduction)

**Validation:**
- Attack time must be shorter than release time
- All parameters within specified ranges

### Effect Chain

Container for multiple effects with execution order.

```python
from src.models.parameters import EffectChain

chain = EffectChain(
    description="warm and intimate vocal sound",
    effects=[eq, compressor, reverb],  # Effect instances
    order=["eq", "compressor", "reverb"]  # Execution order
)

# Access effects
for effect in chain.effects:
    print(effect.effect_type)

# Get specific effect types
eqs = chain.get_effect_by_type("eq")

# Convert to dictionary
data = chain.to_dict()
```

**Parameters:**
- `description`: Human-readable description (required)
- `effects`: List of 1-10 effect parameter instances
- `order`: Effect execution order (must match effects list)

**Validation:**
- Order length must match effects length
- Order types must match actual effect types
- At least 1 effect, maximum 10 effects

## Error Handling

### Exception Hierarchy

```
ParameterGenerationError (base)
├── JSONParseError
│   ├── raw_output (str)
│   └── parse_error (Exception)
├── ValidationError
│   ├── validation_errors (List[dict])
│   └── invalid_data (Any)
├── LLMProviderError
│   ├── provider_error (Exception)
│   ├── provider_name (str)
│   └── request_info (dict)
└── PromptTemplateError
    ├── template_version (str)
    └── template_error (Exception)
```

### Error Handling Example

```python
from src.generation import (
    ParameterGenerator,
    JSONParseError,
    ValidationError,
    LLMProviderError
)

try:
    chain = await generator.generate_parameters(
        description="warm vocal",
        effects=["eq", "reverb"]
    )
except JSONParseError as e:
    print(f"Failed to parse JSON: {e}")
    print(f"Raw output: {e.raw_output[:200]}")

except ValidationError as e:
    print(f"Validation failed: {e}")
    print(f"Errors: {e.validation_errors}")

except LLMProviderError as e:
    print(f"LLM provider failed: {e}")
    print(f"Provider: {e.provider_name}")

except ValueError as e:
    print(f"Invalid input: {e}")
```

### Automatic Correction

The generator automatically attempts to correct invalid output:

1. **Initial Generation**: LLM generates parameters
2. **Validation**: Check against Pydantic schemas
3. **Correction Prompt**: If invalid, send error details back to LLM
4. **Retry**: Up to `max_correction_attempts` (default: 3)
5. **Final Validation**: Return corrected parameters or raise error

```python
generator = ParameterGenerator(
    llm_provider=provider,
    max_correction_attempts=5  # Increase retry attempts
)
```

## Advanced Usage

### Custom Prompt Versions

Create new prompt versions for A/B testing:

1. Create new template file:
```bash
configs/prompts/parameter_generation_v2.yaml
```

2. Load custom version:
```python
generator = ParameterGenerator(
    llm_provider=provider,
    prompt_version="v2"  # Use v2 instead of v1
)
```

### Retry Logic

Control LLM provider retry behavior:

```python
from models.llm_judge import create_provider, AnthropicConfig

config = AnthropicConfig(
    api_key=api_key,
    model="claude-3-5-sonnet-20241022",
    retry={
        "max_attempts": 5,        # More retries
        "initial_delay": 2.0,     # Longer initial delay
        "max_delay": 60.0,        # Max backoff delay
        "exponential_base": 2.0   # Backoff multiplier
    }
)

provider = create_provider("anthropic", config.model_dump())
```

### Custom Validators

Add custom validation logic:

```python
from src.generation.validator import ValidationResult, ValidationIssue, ValidationLevel

def validate_custom_rules(chain):
    issues = []

    # Custom rule: Check for excessive EQ boosts
    for effect in chain.effects:
        if effect.effect_type == "eq":
            for band in effect.bands:
                if band.gain > 6.0:
                    issues.append(ValidationIssue(
                        level=ValidationLevel.WARNING,
                        field=f"eq.band.{band.frequency}Hz",
                        message="Excessive gain may cause distortion",
                        current_value=band.gain,
                        expected_value="≤6 dB"
                    ))

    return ValidationResult(
        is_valid=len(issues) == 0,
        issues=issues
    )

# Use custom validator
chain = await generator.generate_parameters(description="...")
result = validate_custom_rules(chain)
if not result.is_valid:
    print(result.format_report())
```

### Normalization Workflow

Apply normalization to fix out-of-range values:

```python
from src.generation import ParameterGenerator
from src.generation.normalizer import normalize_effect_chain

# Generate (may have out-of-range values)
chain = await generator.generate_parameters(description="...")

# Normalize to valid ranges
normalized_chain = normalize_effect_chain(chain)

# Validate normalized chain
from src.generation.validator import validate_effect_chain
result = validate_effect_chain(normalized_chain)
assert result.is_valid
```

### Temperature Experimentation

Different temperatures for different use cases:

```python
# Consistent, safe parameters
chain = await generator.generate_parameters(
    description="clean vocal",
    temperature=0.3  # Low temperature
)

# Balanced creativity
chain = await generator.generate_parameters(
    description="experimental soundscape",
    temperature=0.7  # Default
)

# Highly creative, varied output
chain = await generator.generate_parameters(
    description="unique texture",
    temperature=1.2  # High temperature
)
```

## Testing

### Unit Tests

Test individual components:

```python
# Test EQ schema validation
from src.models.parameters import EQParameters, EQBand

eq = EQParameters(
    bands=[
        EQBand(frequency=1000, gain=3.0, q=1.0),
        EQBand(frequency=200, gain=-2.0, q=0.7),
        EQBand(frequency=8000, gain=1.0, q=1.5)
    ]
)

assert len(eq.bands) == 3
assert eq.bands[0].frequency == 200  # Sorted!
```

### Integration Tests

Test complete generation pipeline:

```python
import pytest
from src.generation import ParameterGenerator

@pytest.mark.asyncio
async def test_parameter_generation(generator):
    chain = await generator.generate_parameters(
        description="warm vocal",
        effects=["eq", "reverb"]
    )

    assert len(chain.effects) == 2
    assert chain.effects[0].effect_type == "eq"
    assert chain.effects[1].effect_type == "reverb"

    # Validate parameters
    eq = chain.effects[0]
    assert len(eq.bands) >= 3
    for band in eq.bands:
        assert 20 <= band.frequency <= 20000
        assert -12 <= band.gain <= 12
```

Run tests:

```bash
# Run all tests
pytest tests/test_generation/

# Run specific test file
pytest tests/test_generation/test_generator.py -v

# Run with coverage
pytest tests/test_generation/ --cov=src/generation --cov-report=html
```

## Performance

### Optimization Tips

1. **Cache Prompt Templates**: Templates are cached automatically
2. **Reuse Generator Instance**: Avoid recreating generator for each request
3. **Batch Requests**: Use async/await for parallel generation
4. **Control Examples**: Use `num_examples` to limit prompt size
5. **Temperature**: Lower temperature (0.3-0.5) = faster, more deterministic

### Timing Example

```python
import time

# Measure generation time
start = time.time()
chain = await generator.generate_parameters(description="...")
elapsed = time.time() - start

print(f"Generated in {elapsed:.2f}s")
# Typical: 2-5 seconds depending on LLM provider and parameters
```

### Async Batch Generation

Generate multiple parameter sets in parallel:

```python
import asyncio

async def batch_generate(descriptions):
    tasks = [
        generator.generate_parameters(description=desc)
        for desc in descriptions
    ]

    chains = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter successful results
    results = []
    for i, chain in enumerate(chains):
        if isinstance(chain, Exception):
            print(f"Failed: {descriptions[i]} - {chain}")
        else:
            results.append(chain)

    return results

# Generate for multiple descriptions
descriptions = [
    "warm and intimate",
    "bright and energetic",
    "dark and atmospheric"
]

results = await batch_generate(descriptions)
```

## Common Patterns

### Pattern 1: Generate with Validation

```python
chain = await generator.generate_parameters(description="...")

# Validate
from src.generation.validator import validate_effect_chain
result = validate_effect_chain(chain)

if result.has_warnings():
    print("Warnings:", result.get_warnings())

assert result.is_valid
```

### Pattern 2: Generate and Save

```python
import json

chain = await generator.generate_parameters(description="...")

# Save to file
output_file = "output/parameters.json"
with open(output_file, 'w') as f:
    json.dump(chain.to_dict(), f, indent=2)

print(f"Saved to {output_file}")
```

### Pattern 3: Iterative Refinement

```python
# Generate initial parameters
chain = await generator.generate_parameters(description="warm vocal")

# Modify based on feedback
eq = chain.effects[0]
eq.bands[0].gain += 1.5  # Boost low-end more

# Re-validate
from src.generation.validator import validate_effect_parameter
result = validate_effect_parameter(eq)
assert result.is_valid
```

### Pattern 4: A/B Testing Prompts

```python
# Test multiple prompt versions
versions = ["v1", "v2"]
results = {}

for version in versions:
    generator = ParameterGenerator(
        llm_provider=provider,
        prompt_version=version
    )

    chain = await generator.generate_parameters(description="...")
    results[version] = chain

# Compare results
for version, chain in results.items():
    print(f"\n{version}: {len(chain.effects)} effects")
    print(chain.to_dict())
```

## Troubleshooting

### Issue: Missing API Key

**Error:**
```
ValueError: ANTHROPIC_API_KEY not found in environment
```

**Solution:**
1. Create `.env` file in project root
2. Add API key: `ANTHROPIC_API_KEY=sk-ant-...`
3. Run `load_dotenv()` before creating provider

### Issue: JSON Parse Error

**Error:**
```
JSONParseError: Failed to parse JSON from LLM output
```

**Solution:**
- Check LLM output format
- Ensure system prompt instructs JSON-only output
- Increase `max_correction_attempts`
- Use lower temperature for more structured output

### Issue: Validation Errors

**Error:**
```
ValidationError: Failed to validate eq parameters
```

**Solution:**
- Check validation errors: `e.validation_errors`
- Use normalizer to fix out-of-range values
- Adjust prompt to emphasize parameter ranges
- Review few-shot examples for correct patterns

### Issue: Rate Limiting

**Error:**
```
LLMProviderError: Rate limit exceeded
```

**Solution:**
```python
config = AnthropicConfig(
    api_key=api_key,
    rate_limit={
        "requests_per_minute": 30,  # Lower limit
        "tokens_per_minute": 80000
    }
)
```

## Best Practices

1. **Always use `.env` files** for API keys - never hardcode
2. **Start with default temperature (0.7)** - adjust based on results
3. **Include few-shot examples** - dramatically improves output quality
4. **Validate all generated parameters** - use built-in validators
5. **Handle errors gracefully** - use try/except blocks
6. **Cache generator instances** - don't recreate for each request
7. **Use async/await** - for better performance in batch operations
8. **Log generation attempts** - helps debug issues
9. **Version prompt templates** - allows A/B testing
10. **Test with diverse descriptions** - ensure robustness

## References

- **Sony LLM2Fx Paper**: Few-shot example patterns
- **Pydantic Documentation**: Schema validation
- **Anthropic API**: Claude model usage
- **OpenRouter**: Multi-model LLM access

## See Also

- [Prompt Engineering Guide](prompt_engineering_guide.md) - Detailed prompt strategies
- [examples/single_effect_example.py](../examples/single_effect_example.py) - Single effect generation
- [examples/effect_chain_example.py](../examples/effect_chain_example.py) - Multi-effect chains
- [examples/batch_generation.py](../examples/batch_generation.py) - Batch processing
